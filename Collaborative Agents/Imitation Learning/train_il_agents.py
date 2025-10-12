"""
Imitation Learning Trainer (Behavior Cloning) for Fireboy/Watergirl
- Loads NPZ episodes created by data_converter.py
- Trains a classifier to predict discrete actions (0..5) from state vectors
- Now robust to variable state dimensions (e.g., old 21-D and new 33-D with hazard features):
    * Automatically pads per-episode states with zeros to a common target_dim
    * Default target_dim is the maximum dimensionality found across all episodes
- Saves best checkpoints to models/<agent>_<level|all>_best.pt

Usage (CLI):
  python train_il_agents.py --agent both --epochs 50
  python train_il_agents.py --agent fire --level Tutorial --epochs 75
  python train_il_agents.py --data_dir training_data
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


ACTIONS = 6  # 0=none, 1=left, 2=right, 3=jump, 4=left+jump, 5=right+jump


@dataclass
class TrainerConfig:
    agent: str = "both"                  # 'fire', 'water', or 'both'
    level_name: Optional[str] = None     # filter by level; None = all
    data_dir: str = "training_data"      # where episode_*.npz live
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 50
    val_split: float = 0.2
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    target_dim: Optional[int] = None     # if None, use max dim across files


def _load_files(data_dir: str) -> List[Path]:
    path = Path(data_dir)
    files = sorted(path.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No converted NPZ files found in {data_dir}. Run data_converter.py first.")
    return files


def _peek_state_dim(file: Path) -> int:
    arr = np.load(file, allow_pickle=True)
    states = arr["states"]
    if states.ndim != 2 or states.shape[0] == 0:
        return 0
    return states.shape[1]


def _determine_target_dim(files: List[Path], explicit: Optional[int]) -> int:
    if explicit is not None:
        return int(explicit)
    dims = [d for d in (_peek_state_dim(f) for f in files) if d > 0]
    if not dims:
        raise RuntimeError("Could not infer state dimension from any NPZ episode.")
    return max(dims)


class ILEpisodeDataset(Dataset):
    """Flattened per-frame dataset from NPZ episodes, with zero-padding to target_dim."""
    def __init__(self, files: List[Path], agent: str, level_name: Optional[str], target_dim: int):
        assert agent in ("fire", "water")
        self.X: List[np.ndarray] = []
        self.y: List[np.ndarray] = []

        kept = 0
        for f in files:
            data = np.load(f, allow_pickle=True)
            states = data["states"]                     # shape [T, D]
            fire_actions = data["fire_actions"]         # shape [T]
            water_actions = data["water_actions"]       # shape [T]
            meta = json.loads(str(data["metadata"]))

            if level_name and meta.get("level_name") != level_name:
                continue
            if states.ndim != 2 or states.shape[0] == 0:
                continue

            # Pad or truncate features to target_dim (pad at the end with zeros)
            T, D = states.shape
            if D < target_dim:
                pad_width = ((0, 0), (0, target_dim - D))
                states = np.pad(states, pad_width, mode="constant", constant_values=0.0)
            elif D > target_dim:
                states = states[:, :target_dim]

            acts = fire_actions if agent == "fire" else water_actions

            # Drop NaNs if any (match indices)
            mask = ~np.isnan(states).any(axis=1)
            if len(acts) != len(states):
                # defensive: ensure actions length matches T (should in our converter)
                min_len = min(len(acts), len(states))
                states = states[:min_len]
                acts = acts[:min_len]
                mask = mask[:min_len]

            states = states[mask]
            acts = np.asarray(acts, dtype=np.int64)[mask]

            if len(states) == 0:
                continue

            self.X.append(states.astype(np.float32))
            self.y.append(acts.astype(np.int64))
            kept += 1

        if kept == 0:
            raise RuntimeError(f"No episodes found for agent={agent} (level={level_name or 'ALL'})")

        # Concatenate across episodes
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)

        # Final state_dim is target_dim
        self.state_dim = int(target_dim)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BehaviorCloningAgent(nn.Module):
    """Simple MLP for behavior cloning."""
    def __init__(self, state_dim: int, n_actions: int = ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_actions),
        )

    @torch.no_grad()
    def get_action(self, state: torch.Tensor, deterministic: bool = True) -> int:
        """
        state: [D] or [B, D] tensor on any device
        returns: int action id
        """
        if state.ndim == 1:
            state = state.unsqueeze(0)
        logits = self.net(state)
        if deterministic:
            return int(logits.argmax(dim=-1).item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs.squeeze(0), num_samples=1).item())

    def forward(self, x):
        return self.net(x)


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()


def _split_dataset(dataset: Dataset, val_split: float, seed: int) -> Tuple[Dataset, Dataset]:
    n = len(dataset)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val
    return random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))


def train_one_head(cfg: TrainerConfig, agent: str) -> Path:
    """Train one agent head and return checkpoint path."""
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    files = _load_files(cfg.data_dir)
    target_dim = _determine_target_dim(files, cfg.target_dim)
    print(f"[INFO] Using state_dim={target_dim}")

    ds = ILEpisodeDataset(files, agent=agent, level_name=cfg.level_name, target_dim=target_dim)
    train_ds, val_ds = _split_dataset(ds, cfg.val_split, cfg.seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = BehaviorCloningAgent(state_dim=ds.state_dim, n_actions=ACTIONS).to(cfg.device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Compute class weights to handle imbalance (action=0 is often overrepresented)
    action_counts = torch.bincount(torch.from_numpy(ds.y), minlength=ACTIONS).float()
    
    # Only compute weights for actions that actually exist
    mask = action_counts > 0
    class_weights = torch.ones(ACTIONS)
    
    if mask.sum() > 1:  # Need at least 2 classes
        total = action_counts[mask].sum()
        n_classes_present = mask.sum()
        # Inverse frequency for existing actions only
        class_weights[mask] = total / (n_classes_present * action_counts[mask])
        
        # BOOST non-zero movement actions even more (3x)
        for i in range(1, ACTIONS):
            if mask[i]:  # Only boost if action exists
                class_weights[i] *= 3.0
    
    class_weights = class_weights.to(cfg.device)
    print(f"[INFO] Class weights (boosted): {class_weights.cpu().numpy()}")
    
    crit = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = -1.0
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    tag = f"{agent}_{(cfg.level_name or 'all')}_best.pt"
    ckpt_path = save_dir / tag

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss, running_acc, batches = 0.0, 0.0, 0
        for X, y in train_loader:
            X = X.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            logits = model(X)
            loss = crit(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += float(loss.item())
            running_acc += _accuracy(logits, y)
            batches += 1

        # Validation
        model.eval()
        val_loss, val_acc, vb = 0.0, 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(cfg.device, non_blocking=True)
                y = y.to(cfg.device, non_blocking=True)
                logits = model(X)
                loss = crit(logits, y)
                val_loss += float(loss.item())
                val_acc += _accuracy(logits, y)
                vb += 1

        train_loss = running_loss / max(1, batches)
        train_acc = running_acc / max(1, batches)
        val_loss = val_loss / max(1, vb)
        val_acc = val_acc / max(1, vb)

        print(f"Epoch {epoch:03d}/{cfg.epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": best_acc,
                "config": vars(cfg),
                "state_dim": ds.state_dim,   # persisted so BC runtime knows input size
                "agent": agent,
                "level_name": cfg.level_name,
            }, ckpt_path)
            print(f"  Saved new best to {ckpt_path} (val_acc={best_acc:.3f})")

    return ckpt_path


def train_agent(agent: str = "both", level_name: Optional[str] = None, epochs: int = 50,
                data_dir: str = "training_data", batch_size: int = 64, lr: float = 1e-3,
                target_dim: Optional[int] = None) -> None:
    """
    Public API used by quick_start.py
    """
    if agent not in ("fire", "water", "both"):
        raise ValueError("agent must be one of: 'fire', 'water', 'both'")

    cfg = TrainerConfig(agent=agent, level_name=level_name, epochs=epochs,
                        data_dir=data_dir, batch_size=batch_size, lr=lr,
                        target_dim=target_dim)

    if agent in ("fire", "both"):
        print("\n" + "="*70 + "\nTRAINING: FIRE\n" + "="*70)
        train_one_head(cfg, agent="fire")

    if agent in ("water", "both"):
        print("\n" + "="*70 + "\nTRAINING: WATER\n" + "="*70)
        train_one_head(cfg, agent="water")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Behavior Cloning Trainer for Fireboy/Watergirl")
    p.add_argument("--agent", type=str, default="both", choices=["fire", "water", "both"],
                   help="which agent to train")
    p.add_argument("--level", type=str, default=None, help="train only on a single level name")
    p.add_argument("--data_dir", type=str, default="training_data",
                   help="directory with episode_*.npz")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target_dim", type=int, default=None,
                   help="force input feature dim; default uses max across episodes")
    args = p.parse_args()

    train_agent(agent=args.agent, level_name=args.level, epochs=args.epochs,
                data_dir=args.data_dir, batch_size=args.batch_size, lr=args.lr,
                target_dim=args.target_dim)


if __name__ == "__main__":
    main()

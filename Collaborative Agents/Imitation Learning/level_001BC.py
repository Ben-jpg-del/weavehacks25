"""
play_with_bc_v3.py — Fireboy & Watergirl with Behavior-Cloning control (hazard-aware + staged objectives)

What's new vs your v2:
  - Adds FULL hazard sensing matching the training data (33-D state vector):
      * Main LOS (center → exit) for walls.
      * Secondary, parallel LOS from the sprite bottom ("probe") to detect water/lava directly ahead/underfoot.
      * Exposure rule: If PROBE sees hazard and MAIN sees no wall, hazard is EXPOSED (no safe bridge).
        If PROBE sees hazard but MAIN sees a wall, treat as covered (bridge present) → no hazard warning.
  - Draws both LOS lines and hazard outlines on screen.
  - Logs the same hazard signals into position_history for parity with human logs.
  - STAGED OBJECTIVES (NEW):
      * Before plates activated: agents move toward pressure plates
      * After plates activated: agents move toward final exits
      * This fixes oscillation - agents now know to activate switches first!
  - Builds a FULL 33-D state vector matching training data:
      * Base 21-D (positions, velocities, distances, switches, etc.)
      * Fire hazards 6-D: los_haz, hazard_exposed, bottom_haz, bottom_exposed, unsafe_bottom, bridge_ahead
      * Water hazards 6-D: los_haz, hazard_exposed, bottom_haz, bottom_exposed, unsafe_bottom, bridge_ahead
    (Total dims = 21 + 6 + 6 = 33.)
  - Automatically pads/truncates input to match the model's expected state_dim from the checkpoint.

Requirements:
  - pygame, numpy, torch
  - Trained checkpoints saved as:
        models/fire_all_best.pt
        models/water_all_best.pt

Controls:
  - A/D/W: Fire (manual)
  - ←/→/↑: Water (manual)
  - B: Toggle AI globally ON/OFF
  - 1: Toggle Fire AI ON/OFF
  - 2: Toggle Water AI ON/OFF
  - R: Reset current level
  - N / P: Next / Previous level
  - E: Export JSON session log
"""

import pygame as pg
import json
from datetime import datetime
import numpy as np
import torch

# Try to import BehaviorCloningAgent from your trainer; fallback if missing.
try:
    from train_il_agents import BehaviorCloningAgent
except Exception:
    import torch.nn as nn
    class BehaviorCloningAgent(nn.Module):
        def __init__(self, state_dim: int, n_actions: int = 6):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, n_actions),
            )
        @torch.no_grad()
        def get_action(self, state: torch.Tensor, deterministic: bool = True) -> int:
            if state.ndim == 1:
                state = state.unsqueeze(0)
            logits = self.net(state)
            if deterministic:
                return int(logits.argmax(dim=-1).item())
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs.squeeze(0), 1).item())

pg.init()

# ---- Display settings ----
W, H = 960, 540
SCALE = 0.8
PANEL_W = 300
SW, SH = int(W*SCALE) + PANEL_W, int(H*SCALE)

S = pg.display.set_mode((SW, SH))
pg.display.set_caption("Fireboy & Watergirl — BC Agent Demo (v3 hazard-aware 33-D)")
G = pg.Surface((W, H))
C = pg.time.Clock()
F = pg.font.SysFont(None, 22)
F_SMALL = pg.font.SysFont(None, 18)

RED, BLUE, GREEN, GRAY, WHITE, BLACK = (220,60,60), (60,120,255), (60,200,120), (170,170,170), (240,240,240), (15,15,20)
YELLOW, PURPLE = (255,220,60), (180,60,200)

W_CONST, H_CONST = 960.0, 540.0

# ---- Data Logging System (now logs hazard signals too) ----
class GameLogger:
    def __init__(self):
        self.reset()
    def reset(self):
        self.keystrokes = []
        self.position_history = []
        self.switch_events = []
        self.frame_data = []
        self.start_time = pg.time.get_ticks()
        self.last_log_time = self.start_time
    def log_keystroke(self, agent, keys_pressed):
        timestamp = pg.time.get_ticks() - self.start_time
        self.keystrokes.append({
            'time_ms': timestamp,
            'agent': agent,
            'keys': list(keys_pressed)
        })
    def log_positions(self, fire_pos, water_pos, fire_exit, water_exit,
                      fire_rect, water_rect, solids,
                      water_pool=None, lava_pool=None):
        timestamp = pg.time.get_ticks() - self.start_time
        # Distances
        fire_dist = ((fire_pos[0] - fire_exit.centerx)**2 + (fire_pos[1] - fire_exit.centery)**2)**0.5
        water_dist = ((water_pos[0] - water_exit.centerx)**2 + (water_pos[1] - water_exit.centery)**2)**0.5
        # Walls on main LOS
        fire_walls = count_walls_between(fire_rect, fire_exit, solids)
        water_walls = count_walls_between(water_rect, water_exit, solids)
        # Main LOS hazard cross (geometry only)
        fire_los_hz = count_threats_between(fire_rect, fire_exit, [water_pool] if water_pool else [])
        water_los_hz = count_threats_between(water_rect, water_exit, [lava_pool] if lava_pool else [])
        # Bottom-parallel probe → hazard hit?
        fire_probe = probe_hazard_on_parallel(fire_rect, fire_exit, water_pool) if water_pool else 0
        water_probe = probe_hazard_on_parallel(water_rect, water_exit, lava_pool) if lava_pool else 0
        # Exposure rule (matches human recorder):
        fire_exposed = bool(fire_probe and fire_walls == 0)
        water_exposed = bool(water_probe and water_walls == 0)

        self.position_history.append({
            'time_ms': timestamp,
            'fire': {
                'x': fire_pos[0],
                'y': fire_pos[1],
                'dist_to_exit': fire_dist,
                'walls_blocking': fire_walls,
                'path_clear': fire_walls == 0,
                'los_hazard': int(fire_los_hz),
                'probe_hazard': int(fire_probe),
                'hazard_exposed': fire_exposed
            },
            'water': {
                'x': water_pos[0],
                'y': water_pos[1],
                'dist_to_exit': water_dist,
                'walls_blocking': water_walls,
                'path_clear': water_walls == 0,
                'los_hazard': int(water_los_hz),
                'probe_hazard': int(water_probe),
                'hazard_exposed': water_exposed
            },
            'combined_dist': fire_dist + water_dist,
            'total_walls_blocking': fire_walls + water_walls
        })
    def log_switch_event(self, switch_name, activated_by, state):
        timestamp = pg.time.get_ticks() - self.start_time
        self.switch_events.append({
            'time_ms': timestamp,
            'switch': switch_name,
            'agent': activated_by,
            'state': state
        })
    def log_frame(self, game_state):
        timestamp = pg.time.get_ticks() - self.start_time
        self.frame_data.append({
            'time_ms': timestamp,
            'state': game_state.copy()
        })
    def export_session(self, level_name, success, retries):
        total_time = (pg.time.get_ticks() - self.start_time) / 1000
        data = {
            'timestamp': datetime.now().isoformat(),
            'level': level_name,
            'success': success,
            'time_seconds': total_time,
            'retries': retries,
            'keystrokes': self.keystrokes,
            'position_history': self.position_history[-100:],  # last 100 samples
            'switch_events': self.switch_events,
            'summary': {
                'total_keystrokes': len(self.keystrokes),
                'total_frames': len(self.frame_data),
                'final_distance': self.position_history[-1]['combined_dist'] if self.position_history else None
            }
        }
        filename = f"game_log_{level_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Session exported to {filename}")
        return data

# ---- Level Definitions ----
class Level:
    def __init__(self, name, fire_start, water_start, geometry):
        self.name = name
        self.fire_start = fire_start
        self.water_start = water_start
        self.geometry = geometry
    def get_solids(self, bridge_up=False, gate_open=False):
        s = list(self.geometry['base_solids'])
        if bridge_up and 'bridge' in self.geometry:
            s.append(self.geometry['bridge'])
        if not gate_open and 'gate' in self.geometry:
            s.append(self.geometry['gate'])
        return s

LEVEL_1 = Level(
    "Tutorial",
    (420, 380),
    (60, 384),
    {
        'base_solids': [
            pg.Rect(0, 500, int(W_CONST), 40),  # floor
            pg.Rect(40, 420, 200, 20),          # left platform
            pg.Rect(400, 420, 120, 20),         # midL
            pg.Rect(700, 420, 120, 20),         # midR
            pg.Rect(40, 340, 240, 20)           # upper platform
        ],
        'bridge': pg.Rect(520, 400, 180, 20),
        'gate': pg.Rect(280, 340, 20, 120),
        'plateA': pg.Rect(180, 404, 40, 16),
        'plateB': pg.Rect(820, 404, 40, 16),
        'water_pool': pg.Rect(520, 420, 180, 80),
        'lava_pool': pg.Rect(280, 500, 120, 40),
        'exitW': pg.Rect(60, 308, 36, 36),
        'exitF': pg.Rect(880, 388, 36, 36)
    }
)
LEVELS = [LEVEL_1]

# ---- Game State ----
class GameState:
    def __init__(self):
        self.current_level_idx = 0
        self.reset_level()
        self.logger = GameLogger()
        self.retries = 0
        self.done = False
        self.last_bridge_state = False
        self.last_gate_state = False
    def reset_level(self):
        level = LEVELS[self.current_level_idx]
        self.fire = self.mkplayer(*level.fire_start, RED)
        self.water = self.mkplayer(*level.water_start, BLUE)
        self.bridge_up = False
        self.gate_open = False
        self.t0 = pg.time.get_ticks()
        self.done = False
        if hasattr(self, 'logger'):
            self.logger.reset()
            self.last_bridge_state = False
            self.last_gate_state = False
    def mkplayer(self, x, y, color):
        return {"r": pg.Rect(x, y, 28, 36), "v": [0.0, 0.0], "g": False, "c": color}
    def get_level(self):
        return LEVELS[self.current_level_idx]
    def next_level(self):
        if self.current_level_idx < len(LEVELS) - 1:
            self.current_level_idx += 1
            self.retries = 0
            self.reset_level()
            return True
        return False
    def prev_level(self):
        if self.current_level_idx > 0:
            self.current_level_idx -= 1
            self.retries = 0
            self.reset_level()
            return True
        return False

# ---- Physics / movement ----
def move(p, keys, left, right, jump, solids, agent_name, logger):
    VMAX, GRAV = 3.4, 0.5
    keys_pressed = []
    if keys[left]: keys_pressed.append('left')
    if keys[right]: keys_pressed.append('right')
    if keys[jump] and p["g"]: keys_pressed.append('jump')
    if keys_pressed:
        logger.log_keystroke(agent_name, keys_pressed)

    p["v"][0] = (-VMAX if keys[left] else VMAX if keys[right] else 0)
    if p["g"] and keys[jump]: p["v"][1] = -13
    p["v"][1] += GRAV
    p["v"][1] = min(p["v"][1], 12)

    # Horizontal
    p["r"].x += int(p["v"][0])
    for s in solids:
        if p["r"].colliderect(s):
            if p["v"][0] > 0: p["r"].right = s.left
            elif p["v"][0] < 0: p["r"].left = s.right

    # Vertical
    p["r"].y += int(p["v"][1])
    p["g"] = False
    for s in solids:
        if p["r"].colliderect(s):
            if p["v"][1] > 0:
                p["r"].bottom = s.top
                p["g"] = True
            else:
                p["r"].top = s.bottom
            p["v"][1] = 0

# ---- Geometry helpers (shared with human version) ----
def line_segments_intersect(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return False
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
    return 0 <= t <= 1 and 0 <= u <= 1

def segment_intersection_t(p1, p2, q1, q2):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = q1; x4, y4 = q2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return t
    return None

def first_intersection_t_with_rect(p1, p2, rect):
    if rect is None:
        return None
    if rect.collidepoint(p1):
        return 0.0
    edges = [
        ((rect.left, rect.top), (rect.right, rect.top)),
        ((rect.right, rect.top), (rect.right, rect.bottom)),
        ((rect.right, rect.bottom), (rect.left, rect.bottom)),
        ((rect.left, rect.bottom), (rect.left, rect.top))
    ]
    best_t = None
    for a, b in edges:
        t = segment_intersection_t(p1, p2, a, b)
        if t is not None:
            best_t = t if best_t is None else min(best_t, t)
    return best_t

def line_intersects_rect(p1, p2, rect):
    if rect is None:
        return False
    if rect.collidepoint(p1) or rect.collidepoint(p2):
        return True
    edges = [
        ((rect.left, rect.top), (rect.right, rect.top)),
        ((rect.right, rect.top), (rect.right, rect.bottom)),
        ((rect.right, rect.bottom), (rect.left, rect.bottom)),
        ((rect.left, rect.bottom), (rect.left, rect.top))
    ]
    return any(line_segments_intersect(p1, p2, a, b) for a, b in edges)

def count_walls_between(start_rect, end_rect, solids):
    p1 = start_rect.center
    p2 = end_rect.center
    return sum(1 for s in solids if line_intersects_rect(p1, p2, s))

def count_threats_between(start_rect, end_rect, hazard_rects):
    p1 = start_rect.center
    p2 = end_rect.center
    hits = 0
    for hz in hazard_rects:
        if hz and line_intersects_rect(p1, p2, hz):
            hits += 1
    return hits

def parallel_segment_points(start_rect, end_rect):
    """Create a second LOS parallel to the main line, offset downward from the agent center to its bottom."""
    offset_y = start_rect.height//2 - 1
    p1_main = start_rect.center
    p2_main = end_rect.center
    p1b = (p1_main[0], p1_main[1] + offset_y)
    p2b = (p2_main[0], p2_main[1] + offset_y)
    return p1b, p2b

def probe_hazard_on_parallel(start_rect, end_rect, hazard_rect):
    """Return 1 if the parallel (bottom) LOS intersects the hazard rect, else 0."""
    if hazard_rect is None:
        return 0
    p1b, p2b = parallel_segment_points(start_rect, end_rect)
    return 1 if line_intersects_rect(p1b, p2b, hazard_rect) else 0

# ---- AI Controller (Behavior Cloning) ----
ACTIONS = 6  # 0 none, 1 L, 2 R, 3 J, 4 L+J, 5 R+J

class KeyProxy:
    """Mimic pygame key state object for a small set of pressed keys."""
    def __init__(self, pressed_keys):
        self._pressed = set(pressed_keys)
    def __getitem__(self, keycode):
        return keycode in self._pressed

class CombinedKeyProxy:
    """Union of human and AI keys: True if either is pressed."""
    def __init__(self, human_keys, ai_keys_or_none):
        self.human = human_keys
        self.ai = ai_keys_or_none
    def __getitem__(self, keycode):
        return self.human[keycode] or (self.ai[keycode] if self.ai is not None else False)

class BCController:
    def __init__(self,
                 fire_ckpt="models/fire_all_best.pt",
                 water_ckpt="models/water_all_best.pt",
                 epsilon_idle=0.15,
                 stuck_frames=15):
        self.fire = None
        self.water = None
        self.ready_fire = False
        self.ready_water = False

        self.fire_dim = None
        self.water_dim = None

        self.prev_fx_norm = None
        self.prev_fy_norm = None
        self.prev_wx_norm = None
        self.prev_wy_norm = None
        self.last_level_idx = None

        # Diagnostics
        self.last_action_fire = -1
        self.last_action_water = -1
        self.idle_count_fire = 0
        self.idle_count_water = 0

        # Params
        self.epsilon_idle = float(epsilon_idle)
        self.stuck_frames = int(stuck_frames)

        # Try load agents
        self.ready_fire = self._load_one('fire', fire_ckpt)
        self.ready_water = self._load_one('water', water_ckpt)

        # Toggles
        self.enabled_global = True
        self.enabled_fire = True
        self.enabled_water = True

    def _load_one(self, which, path):
        try:
            ckpt = torch.load(path, map_location="cpu")
            state_dim = int(ckpt.get("state_dim", 21))
            model = BehaviorCloningAgent(state_dim=state_dim)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            if which == 'fire':
                self.fire = model
                self.fire_dim = state_dim
            else:
                self.water = model
                self.water_dim = state_dim
            print(f"[BC] Loaded {which} model: {path} (state_dim={state_dim})")
            return True
        except Exception as e:
            print(f"[BC] Could not load {which} model at {path}: {e}")
            return False

    def reset_vel_baseline(self):
        self.prev_fx_norm = self.prev_fy_norm = None
        self.prev_wx_norm = self.prev_wy_norm = None
        self.idle_count_fire = 0
        self.idle_count_water = 0

    def _align_dim(self, x: torch.Tensor, want: int) -> torch.Tensor:
        """Pad with zeros or truncate to match checkpoint input size."""
        got = x.shape[-1]
        if got == want:
            return x
        if got < want:
            pad = torch.zeros((want - got,), dtype=x.dtype)
            return torch.cat([x, pad], dim=-1)
        return x[:want]

    def _check_bottom_exposed(self, agent_rect, exit_rect, solids, hazard_rect) -> float:
        """
        Check if hazard is exposed on bottom-parallel LOS.
        Returns 1.0 if exposed (no solid before hazard), 0.0 otherwise.
        Matches logic from level_001.py exposed_hazard_along_bottom_line.
        """
        if not hazard_rect:
            return 0.0
        
        # Create bottom-parallel line (same as probe)
        p1b, p2b = parallel_segment_points(agent_rect, exit_rect)
        
        # Check if hazard is hit
        if not line_intersects_rect(p1b, p2b, hazard_rect):
            return 0.0
        
        # Get t-value for hazard intersection
        t_hz = first_intersection_t_with_rect(p1b, p2b, hazard_rect)
        if t_hz is None:
            return 0.0
        
        # Check if any solid comes before the hazard
        t_solid = None
        for s in solids:
            t = first_intersection_t_with_rect(p1b, p2b, s)
            if t is not None:
                t_solid = t if t_solid is None else min(t_solid, t)
        
        # Exposed if no solid or solid comes after hazard
        exposed = (t_solid is None) or (t_solid > t_hz + 1e-6)
        return 1.0 if exposed else 0.0

    def _build_state(self, state: 'GameState') -> torch.Tensor:
        """
        Build the full 33-D state vector matching training data:
          base 21-D +
          fire hazards (6-D): los_haz, hazard_exposed, bottom_haz, bottom_exposed, unsafe_bottom, bridge_ahead
          water hazards (6-D): los_haz, hazard_exposed, bottom_haz, bottom_exposed, unsafe_bottom, bridge_ahead
        """
        level = state.get_level()
        geo = level.geometry
        solids = level.get_solids(state.bridge_up, state.gate_open)

        # positions normalized (top-left)
        fx = state.fire["r"].x; fy = state.fire["r"].y
        wx = state.water["r"].x; wy = state.water["r"].y
        fxn, fyn = fx/W_CONST, fy/H_CONST
        wxn, wyn = wx/W_CONST, wy/H_CONST

        # STAGED OBJECTIVES: distance to plate first, then to exit
        # If plates not active yet, measure distance to plate
        # Once plates active, measure distance to exit
        if state.bridge_up or state.gate_open:
            # Plates activated - go to exit
            fire_dist = ((fx - geo['exitF'].centerx)**2 + (fy - geo['exitF'].centery)**2) ** 0.5
            water_dist = ((wx - geo['exitW'].centerx)**2 + (wy - geo['exitW'].centery)**2) ** 0.5
        else:
            # Plates not active - go to plate first
            if 'plateB' in geo:
                fire_dist = ((fx - geo['plateB'].centerx)**2 + (fy - geo['plateB'].centery)**2) ** 0.5
            else:
                fire_dist = ((fx - geo['exitF'].centerx)**2 + (fy - geo['exitF'].centery)**2) ** 0.5
            
            if 'plateA' in geo:
                water_dist = ((wx - geo['plateA'].centerx)**2 + (wy - geo['plateA'].centery)**2) ** 0.5
            else:
                water_dist = ((wx - geo['exitW'].centerx)**2 + (wy - geo['exitW'].centery)**2) ** 0.5

        # walls / LOS
        fire_walls = float(count_walls_between(state.fire["r"], geo['exitF'], solids))
        water_walls = float(count_walls_between(state.water["r"], geo['exitW'], solids))
        fire_clear = 1.0 if fire_walls == 0.0 else 0.0
        water_clear = 1.0 if water_walls == 0.0 else 0.0

        # switches
        plateA_active = 1.0 if ('plateA' in geo and (state.fire["r"].colliderect(geo['plateA']) or state.water["r"].colliderect(geo['plateA']))) else 0.0
        plateB_active = 1.0 if ('plateB' in geo and (state.fire["r"].colliderect(geo['plateB']) or state.water["r"].colliderect(geo['plateB']))) else 0.0
        bridge_up = 1.0 if state.bridge_up else 0.0
        gate_open = 1.0 if state.gate_open else 0.0

        # relatives
        rel_x = (fx - wx) / W_CONST
        rel_y = (fy - wy) / H_CONST
        combined_dist = (fire_dist + water_dist) / 1500.0

        # velocities (approx from normalized pos deltas)
        if self.prev_fx_norm is None:
            fvx = fvy = wvx = wvy = 0.0
        else:
            fvx = (fxn - self.prev_fx_norm) * W_CONST / 10.0
            fvy = (fyn - self.prev_fy_norm) * H_CONST / 20.0
            wvx = (wxn - self.prev_wx_norm) * W_CONST / 10.0
            wvy = (wyn - self.prev_wy_norm) * H_CONST / 20.0
        self.prev_fx_norm, self.prev_fy_norm = fxn, fyn
        self.prev_wx_norm, self.prev_wy_norm = wxn, wyn

        # --- Full 12-D hazard features (matches data_converter.py) ---
        water_pool = geo.get('water_pool')
        lava_pool  = geo.get('lava_pool')
        
        # Fire hazards (6-D)
        fire_los_haz = float(count_threats_between(state.fire["r"], geo['exitF'], [water_pool] if water_pool else []))
        fire_bottom_haz = float(probe_hazard_on_parallel(state.fire["r"], geo['exitF'], water_pool) if water_pool else 0)
        # Compute bottom_exposed using helper (more accurate than just checking probe)
        fire_bottom_exposed = self._check_bottom_exposed(state.fire["r"], geo['exitF'], solids, water_pool)
        fire_hazard_exposed = 1.0 if (fire_los_haz > 0 and fire_walls == 0.0) else 0.0
        fire_unsafe_bottom = 1.0 if (fire_bottom_exposed and fire_walls == 0.0) else 0.0
        fire_bridge_ahead = 1.0 if (fire_bottom_haz > 0 and fire_walls > 0.0) else 0.0
        
        # Water hazards (6-D)
        water_los_haz = float(count_threats_between(state.water["r"], geo['exitW'], [lava_pool] if lava_pool else []))
        water_bottom_haz = float(probe_hazard_on_parallel(state.water["r"], geo['exitW'], lava_pool) if lava_pool else 0)
        water_bottom_exposed = self._check_bottom_exposed(state.water["r"], geo['exitW'], solids, lava_pool)
        water_hazard_exposed = 1.0 if (water_los_haz > 0 and water_walls == 0.0) else 0.0
        water_unsafe_bottom = 1.0 if (water_bottom_exposed and water_walls == 0.0) else 0.0
        water_bridge_ahead = 1.0 if (water_bottom_haz > 0 and water_walls > 0.0) else 0.0

        v = np.array([
            # base 21-D
            fxn, fyn, fvx, fvy, fire_dist/1000.0, fire_walls, fire_clear,
            wxn, wyn, wvx, wvy, water_dist/1000.0, water_walls, water_clear,
            bridge_up, gate_open, plateA_active, plateB_active,
            rel_x, rel_y, combined_dist,
            # fire hazards (6-D): indices 21-26
            fire_los_haz, fire_hazard_exposed, fire_bottom_haz, fire_bottom_exposed,
            fire_unsafe_bottom, fire_bridge_ahead,
            # water hazards (6-D): indices 27-32
            water_los_haz, water_hazard_exposed, water_bottom_haz, water_bottom_exposed,
            water_unsafe_bottom, water_bridge_ahead
        ], dtype=np.float32)

        return torch.from_numpy(v)

    def _action_to_keys(self, action_id, left_key, right_key, jump_key):
        if action_id == 1:
            return KeyProxy([left_key])
        elif action_id == 2:
            return KeyProxy([right_key])
        elif action_id == 3:
            return KeyProxy([jump_key])
        elif action_id == 4:
            return KeyProxy([left_key, jump_key])
        elif action_id == 5:
            return KeyProxy([right_key, jump_key])
        else:
            return KeyProxy([])

    def _nudge_toward_exit(self, which, state):
        geo = state.get_level().geometry
        if which == 'fire':
            ex = geo['exitF'].centerx; px = state.fire["r"].centerx
            return 2 if ex > px else 1
        else:
            ex = geo['exitW'].centerx; px = state.water["r"].centerx
            return 2 if ex > px else 1

    def act(self, state: 'GameState'):
        if self.last_level_idx != state.current_level_idx:
            self.reset_vel_baseline()
            self.last_level_idx = state.current_level_idx

        x_full = self._build_state(state)

        fire_keys = None
        water_keys = None

        if self.enabled_global and self.enabled_fire and self.ready_fire:
            xf = self._align_dim(x_full, self.fire_dim or x_full.shape[-1])
            a_f = self.fire.get_action(xf, deterministic=True)
            if a_f == 0 and np.random.rand() < self.epsilon_idle:
                a_f = self.fire.get_action(xf, deterministic=False)
            self.idle_count_fire = self.idle_count_fire + 1 if a_f == 0 else 0
            if a_f == 0 and self.idle_count_fire >= self.stuck_frames:
                a_f = self._nudge_toward_exit('fire', state)
                self.idle_count_fire = 0
            self.last_action_fire = a_f
            fire_keys = self._action_to_keys(a_f, pg.K_a, pg.K_d, pg.K_w)

        if self.enabled_global and self.enabled_water and self.ready_water:
            xw = self._align_dim(x_full, self.water_dim or x_full.shape[-1])
            a_w = self.water.get_action(xw, deterministic=True)
            if a_w == 0 and np.random.rand() < self.epsilon_idle:
                a_w = self.water.get_action(xw, deterministic=False)
            self.idle_count_water = self.idle_count_water + 1 if a_w == 0 else 0
            if a_w == 0 and self.idle_count_water >= self.stuck_frames:
                a_w = self._nudge_toward_exit('water', state)
                self.idle_count_water = 0
            self.last_action_water = a_w
            water_keys = self._action_to_keys(a_w, pg.K_LEFT, pg.K_RIGHT, pg.K_UP)

        return fire_keys, water_keys

# ---- Drawing (with main LOS + parallel probe + hazard overlays) ----
def draw(state, bot):
    G.fill(BLACK)
    level = state.get_level()
    geo = level.geometry

    # Hazards
    if 'water_pool' in geo:
        pg.draw.rect(G, (30, 90, 200), geo['water_pool'])
    if 'lava_pool' in geo:
        pg.draw.rect(G, (200, 60, 30), geo['lava_pool'])

    # Solids
    solids = level.get_solids(state.bridge_up, state.gate_open)
    for s in solids:
        pg.draw.rect(G, GRAY, s)

    # Plates
    if 'plateA' in geo: pg.draw.rect(G, BLUE, geo['plateA'])
    if 'plateB' in geo: pg.draw.rect(G, RED,  geo['plateB'])

    # Exits
    pg.draw.rect(G, BLUE, geo['exitW'], 2)
    pg.draw.rect(G, RED,  geo['exitF'], 2)

    # Players
    pg.draw.rect(G, state.fire["c"], state.fire["r"])
    pg.draw.rect(G, state.water["c"], state.water["r"])

    # LOS
    fire_walls = count_walls_between(state.fire["r"],  geo['exitF'], solids)
    water_walls = count_walls_between(state.water["r"], geo['exitW'], solids)

    # Main LOS color (green if clear; tinted if blocked)
    line_color_f = (255, 100, 100) if fire_walls > 0 else (100, 255, 100)
    line_color_w = (100, 100, 255) if water_walls > 0 else (100, 255, 100)
    pg.draw.line(G, line_color_f, state.fire["r"].center,  geo['exitF'].center, 1)
    pg.draw.line(G, line_color_w, state.water["r"].center, geo['exitW'].center, 1)

    # Parallel probe (bottom) lines
    f_p1b, f_p2b = parallel_segment_points(state.fire["r"],  geo['exitF'])
    w_p1b, w_p2b = parallel_segment_points(state.water["r"], geo['exitW'])
    pg.draw.line(G, PURPLE, f_p1b, f_p2b, 1)
    pg.draw.line(G, PURPLE, w_p1b, w_p2b, 1)

    # Hazard probe + exposure
    fire_probe = probe_hazard_on_parallel(state.fire["r"],  geo['exitF'], geo.get('water_pool'))
    water_probe = probe_hazard_on_parallel(state.water["r"], geo['exitW'], geo.get('lava_pool'))
    fire_exposed = bool(fire_probe and fire_walls == 0)
    water_exposed = bool(water_probe and water_walls == 0)

    # Overlay hazard warning only if exposed (no covering wall on main LOS)
    if fire_exposed:
        pg.draw.line(G, YELLOW, state.fire["r"].center, geo['exitF'].center, 3)
        if 'water_pool' in geo: pg.draw.rect(G, YELLOW, geo['water_pool'], 2)
    if water_exposed:
        pg.draw.line(G, YELLOW, state.water["r"].center, geo['exitW'].center, 3)
        if 'lava_pool' in geo: pg.draw.rect(G, YELLOW, geo['lava_pool'], 2)

    # HUD
    tt = (pg.time.get_ticks() - state.t0) / 1000 if not state.done else state.final_time
    hud = f"Level {state.current_level_idx+1}: {level.name} | Time {tt:.2f}s | Retries {state.retries}"
    G.blit(F.render(hud, True, WHITE), (12, 10))
    G.blit(F.render("A/D/W Fire | ←/→/↑ Water | R reset | N/P level | E export | B AI | 1/2 per-agent", True, WHITE), (12, 34))

    # AI status + last actions
    status = f"AI:{'ON' if bot.enabled_global else 'OFF'}  Fire:{'ON' if bot.enabled_fire else 'OFF'}  Water:{'ON' if bot.enabled_water else 'OFF'}"
    G.blit(F_SMALL.render(status, True, WHITE), (12, 54))
    G.blit(F_SMALL.render(f"Last actions — Fire:{bot.last_action_fire}  Water:{bot.last_action_water}", True, WHITE), (12, 74))

    # Composite to window
    S.fill(BLACK)
    scaled = pg.transform.smoothscale(G, (int(W*SCALE), int(H*SCALE)))
    S.blit(scaled, (0, 0))

    # Right panel
    panel_x = int(W*SCALE)
    panel = pg.Rect(panel_x, 0, PANEL_W, SH)
    pg.draw.rect(S, (28, 28, 42), panel)

    S.blit(F.render("LIVE DATA LOG", True, WHITE), (panel.x+12, 12))
    y = 40

    fire_dist = ((state.fire["r"].centerx - geo['exitF'].centerx)**2 +
                 (state.fire["r"].centery - geo['exitF'].centery)**2)**0.5
    water_dist = ((state.water["r"].centerx - geo['exitW'].centerx)**2 +
                  (state.water["r"].centery - geo['exitW'].centery)**2)**0.5

    S.blit(F_SMALL.render(f"Fire Pos: ({state.fire['r'].x}, {state.fire['r'].y})", True, RED), (panel.x+12, y)); y += 20
    S.blit(F_SMALL.render(f"Exit Dist: {fire_dist:.1f}px", True, RED), (panel.x+12, y)); y += 18
    S.blit(F_SMALL.render(f"Walls: {fire_walls} {'(BLOCKED)' if fire_walls > 0 else '(CLEAR)'}", True, RED if fire_walls>0 else GREEN), (panel.x+12, y)); y += 18
    S.blit(F_SMALL.render(f"Hazard Probe: {'YES' if fire_probe else 'NO'} (WATER)", True, YELLOW if fire_probe else GRAY), (panel.x+12, y)); y += 18
    S.blit(F_SMALL.render(f"Hazard Exposed: {'YES' if fire_exposed else 'NO'}", True, YELLOW if fire_exposed else GRAY), (panel.x+12, y)); y += 25

    S.blit(F_SMALL.render(f"Water Pos: ({state.water['r'].x}, {state.water['r'].y})", True, BLUE), (panel.x+12, y)); y += 20
    S.blit(F_SMALL.render(f"Exit Dist: {water_dist:.1f}px", True, BLUE), (panel.x+12, y)); y += 18
    S.blit(F_SMALL.render(f"Walls: {water_walls} {'(BLOCKED)' if water_walls > 0 else '(CLEAR)'}", True, RED if water_walls>0 else GREEN), (panel.x+12, y)); y += 18
    S.blit(F_SMALL.render(f"Hazard Probe: {'YES' if water_probe else 'NO'} (LAVA)", True, YELLOW if water_probe else GRAY), (panel.x+12, y)); y += 18
    S.blit(F_SMALL.render(f"Hazard Exposed: {'YES' if water_exposed else 'NO'}", True, YELLOW if water_exposed else GRAY), (panel.x+12, y)); y += 25

    # Switches
    S.blit(F_SMALL.render("SWITCHES:", True, YELLOW), (panel.x+12, y)); y += 20
    if 'plateA' in geo:
        plateA_active = (state.fire["r"].colliderect(geo['plateA']) or state.water["r"].colliderect(geo['plateA']))
        S.blit(F_SMALL.render(f"Plate A: {'ON' if plateA_active else 'OFF'}", True, GREEN if plateA_active else GRAY), (panel.x+12, y)); y += 18
    if 'plateB' in geo:
        plateB_active = (state.fire["r"].colliderect(geo['plateB']) or state.water["r"].colliderect(geo['plateB']))
        S.blit(F_SMALL.render(f"Plate B: {'ON' if plateB_active else 'OFF'}", True, GREEN if plateB_active else GRAY), (panel.x+12, y)); y += 18

    y += 10
    S.blit(F_SMALL.render(f"Bridge: {'UP' if state.bridge_up else 'DOWN'}", True, GREEN if state.bridge_up else GRAY), (panel.x+12, y)); y += 18
    S.blit(F_SMALL.render(f"Gate: {'OPEN' if state.gate_open else 'CLOSED'}", True, GREEN if state.gate_open else GRAY), (panel.x+12, y)); y += 25

    S.blit(F_SMALL.render("STATISTICS:", True, YELLOW), (panel.x+12, y)); y += 20
    S.blit(F_SMALL.render(f"Keystrokes: {len(state.logger.keystrokes)}", True, WHITE), (panel.x+12, y)); y += 18
    S.blit(F_SMALL.render(f"Switch Events: {len(state.logger.switch_events)}", True, WHITE), (panel.x+12, y)); y += 18
    S.blit(F_SMALL.render(f"Positions Logged: {len(state.logger.position_history)}", True, WHITE), (panel.x+12, y))

    # Win banner
    if state.done:
        box = pg.Rect(0, 0, 420, 90); box.center = (W//2, 90)
        pg.draw.rect(G, (35, 35, 50), box, 0, 8)
        pg.draw.rect(G, GREEN, box, 2, 8)
        G.blit(F.render(f"SUCCESS: {state.final_time:.2f}s, Retries {state.retries}", True, WHITE), (box.x+14, box.y+28))

    pg.display.flip()

# ---- Main Loop ----
def main():
    state = GameState()
    bot = BCController(
        fire_ckpt="models/fire_all_best.pt",
        water_ckpt="models/water_all_best.pt",
        epsilon_idle=0.0,  # Disable randomness for cleaner behavior
        stuck_frames=10    # Nudge sooner to avoid long pauses
    )

    running = True
    while running:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                running = False
            if e.type == pg.KEYDOWN:
                if e.key == pg.K_b:
                    bot.enabled_global = not bot.enabled_global
                    print(f"[BC] Global AI: {'ON' if bot.enabled_global else 'OFF'}")
                if e.key == pg.K_1:
                    bot.enabled_fire = not bot.enabled_fire
                    print(f"[BC] Fire AI: {'ON' if bot.enabled_fire else 'OFF'}")
                if e.key == pg.K_2:
                    bot.enabled_water = not bot.enabled_water
                    print(f"[BC] Water AI: {'ON' if bot.enabled_water else 'OFF'}")

                if e.key == pg.K_r:
                    state.retries += 1
                    state.reset_level()
                    bot.reset_vel_baseline()
                if e.key == pg.K_n:
                    if state.next_level():
                        bot.reset_vel_baseline()
                if e.key == pg.K_p:
                    if state.prev_level():
                        bot.reset_vel_baseline()
                if e.key == pg.K_e:
                    state.logger.export_session(
                        state.get_level().name,
                        state.done,
                        state.retries
                    )

        if not state.done:
            level = state.get_level()
            geo = level.geometry
            solids = level.get_solids(state.bridge_up, state.gate_open)

            # Plate logic (single-plate style for this level)
            if 'plateA' in geo:
                plateA_on = (state.fire["r"].colliderect(geo['plateA']) or state.water["r"].colliderect(geo['plateA']))
                if plateA_on != state.bridge_up:
                    agent = 'fire' if state.fire["r"].colliderect(geo['plateA']) else 'water'
                    state.logger.log_switch_event('plateA', agent, plateA_on)
                state.bridge_up = plateA_on
            if 'plateB' in geo:
                plateB_on = (state.fire["r"].colliderect(geo['plateB']) or state.water["r"].colliderect(geo['plateB']))
                if plateB_on != state.gate_open:
                    agent = 'fire' if state.fire["r"].colliderect(geo['plateB']) else 'water'
                    state.logger.log_switch_event('plateB', agent, plateB_on)
                state.gate_open = plateB_on

            # Player movement (merge AI + manual)
            keys_human = pg.key.get_pressed()
            fire_keys, water_keys = bot.act(state)

            fire_input  = CombinedKeyProxy(keys_human, fire_keys)
            water_input = CombinedKeyProxy(keys_human, water_keys)

            move(state.fire,  fire_input,  pg.K_a,   pg.K_d,   pg.K_w,  solids, 'fire',  state.logger)
            move(state.water, water_input, pg.K_LEFT,pg.K_RIGHT,pg.K_UP, solids, 'water', state.logger)

            # Log positions every frame (include hazard rects for parity)
            state.logger.log_positions(
                state.fire["r"].topleft, state.water["r"].topleft,
                geo['exitF'], geo['exitW'],
                state.fire["r"], state.water["r"],
                solids,
                water_pool=geo.get('water_pool'),
                lava_pool=geo.get('lava_pool')
            )

            # Death checks
            if 'water_pool' in geo and state.fire["r"].colliderect(geo['water_pool']):
                state.retries += 1
                state.reset_level()
                bot.reset_vel_baseline()
            if 'lava_pool' in geo and state.water["r"].colliderect(geo['lava_pool']):
                state.retries += 1
                state.reset_level()
                bot.reset_vel_baseline()

            # Win check
            fire_in = geo['exitF'].collidepoint(state.fire["r"].center)
            water_in = geo['exitW'].collidepoint(state.water["r"].center)
            if fire_in and water_in:
                state.done = True
                state.final_time = (pg.time.get_ticks() - state.t0) / 1000
                state.logger.export_session(level.name, True, state.retries)

        draw(state, bot)
        C.tick(60)

    pg.quit()

if __name__ == "__main__":
    main()

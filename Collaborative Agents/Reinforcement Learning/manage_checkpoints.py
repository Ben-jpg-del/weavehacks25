#!/usr/bin/env python3
"""
Checkpoint Management Utility
Helps manage, view, and clean up training checkpoints
"""

import os
import json
import torch
from datetime import datetime
from pathlib import Path

def get_file_size(filepath):
    """Get file size in MB"""
    if not os.path.exists(filepath):
        return 0
    return os.path.getsize(filepath) / (1024 * 1024)

def list_checkpoints(models_dir="models"):
    """List all checkpoints and their metadata"""
    print("\n" + "="*80)
    print("CHECKPOINT STATUS")
    print("="*80)
    
    latest_checkpoint = f"{models_dir}/cooperative_agent_checkpoint_latest.pt"
    latest_buffers = f"{models_dir}/cooperative_agent_buffers_latest.pkl"
    
    # Check if latest checkpoint exists
    if os.path.exists(latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        print(f"\n✓ Latest Checkpoint Found")
        print(f"  Episode: {checkpoint['episode']}")
        print(f"  Timestamp: {checkpoint.get('timestamp', 'Unknown')}")
        print(f"  Best Avg Reward: {checkpoint.get('best_avg_reward', 'N/A'):.2f}")
        print(f"  Fire Epsilon: {checkpoint['fire_epsilon']:.3f}")
        print(f"  Water Epsilon: {checkpoint['water_epsilon']:.3f}")
        print(f"  Architecture: {checkpoint.get('architecture', 'Unknown')}")
        print(f"  State Dim: {checkpoint.get('state_dim', 'Unknown')}")
        print(f"  File Size: {get_file_size(latest_checkpoint):.2f} MB")
        
        if os.path.exists(latest_buffers):
            print(f"  Buffer Size: {get_file_size(latest_buffers):.2f} MB")
        else:
            print(f"  ⚠ Buffers Not Found")
    else:
        print("\n✗ No checkpoint found - will start fresh training")
    
    # List best models
    print(f"\n{'='*80}")
    print("BEST MODELS (Lightweight)")
    print("="*80)
    
    best_models = [
        f"{models_dir}/cooperative_agent_shared_backbone.pt",
        f"{models_dir}/cooperative_agent_fire_head.pt",
        f"{models_dir}/cooperative_agent_water_head.pt"
    ]
    
    for model_file in best_models:
        if os.path.exists(model_file):
            size = get_file_size(model_file)
            print(f"✓ {os.path.basename(model_file):45s} {size:6.2f} MB")
        else:
            print(f"✗ {os.path.basename(model_file):45s} Not Found")
    
    # List historical checkpoints
    print(f"\n{'='*80}")
    print("HISTORICAL CHECKPOINTS")
    print("="*80)
    
    checkpoint_files = []
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.startswith("cooperative_agent_checkpoint_ep") and file.endswith(".pt"):
                checkpoint_files.append(file)
    
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split("_ep")[1].split(".")[0]))
        total_size = 0
        for checkpoint_file in checkpoint_files:
            filepath = f"{models_dir}/{checkpoint_file}"
            episode = checkpoint_file.split("_ep")[1].split(".")[0]
            size = get_file_size(filepath)
            total_size += size
            
            buffer_file = f"{models_dir}/cooperative_agent_buffers_ep{episode}.pkl"
            buffer_size = get_file_size(buffer_file)
            total_size += buffer_size
            
            buffer_status = f"({buffer_size:.1f} MB buffers)" if buffer_size > 0 else "(no buffers)"
            print(f"  Episode {episode:5s}: {size:6.2f} MB {buffer_status}")
        
        print(f"\n  Total: {len(checkpoint_files)} checkpoints, {total_size:.2f} MB")
    else:
        print("  No historical checkpoints found")
    
    # Training history
    print(f"\n{'='*80}")
    print("TRAINING HISTORY")
    print("="*80)
    
    history_file = f"{models_dir}/training_logs/training_history.jsonl"
    if os.path.exists(history_file):
        entries = []
        with open(history_file, 'r') as f:
            for line in f:
                entries.append(json.loads(line))
        
        if entries:
            print(f"✓ Training history found: {len(entries)} episodes logged")
            print(f"  File size: {get_file_size(history_file):.2f} MB")
            print(f"  First episode: {entries[0]['episode']} at {entries[0]['timestamp']}")
            print(f"  Last episode: {entries[-1]['episode']} at {entries[-1]['timestamp']}")
            print(f"  Latest success rate: {entries[-1]['success_rate_100']:.1%}")
            print(f"  Latest avg reward: {entries[-1]['avg_reward_100']:.2f}")
        else:
            print("✓ Training history file exists but is empty")
    else:
        print("✗ No training history found")
    
    print(f"\n{'='*80}\n")

def clean_old_checkpoints(models_dir="models", keep_latest_n=3):
    """Clean up old episode-specific checkpoints, keeping only the latest N"""
    print(f"\nCleaning old checkpoints (keeping latest {keep_latest_n})...")
    
    checkpoint_files = []
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.startswith("cooperative_agent_checkpoint_ep") and file.endswith(".pt"):
                episode = int(file.split("_ep")[1].split(".")[0])
                checkpoint_files.append((episode, file))
    
    if not checkpoint_files:
        print("No historical checkpoints to clean")
        return
    
    # Sort by episode number
    checkpoint_files.sort(reverse=True)
    
    # Keep latest N, delete the rest
    deleted_count = 0
    freed_space = 0
    
    for episode, checkpoint_file in checkpoint_files[keep_latest_n:]:
        checkpoint_path = f"{models_dir}/{checkpoint_file}"
        buffer_path = f"{models_dir}/cooperative_agent_buffers_ep{episode}.pkl"
        
        # Calculate space to be freed
        freed_space += get_file_size(checkpoint_path)
        freed_space += get_file_size(buffer_path)
        
        # Delete files
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            deleted_count += 1
        
        if os.path.exists(buffer_path):
            os.remove(buffer_path)
        
        print(f"  Deleted checkpoint at episode {episode}")
    
    print(f"\nCleaned {deleted_count} old checkpoints, freed {freed_space:.2f} MB")

def view_recent_history(models_dir="models", last_n=20):
    """View recent training history"""
    history_file = f"{models_dir}/training_logs/training_history.jsonl"
    
    if not os.path.exists(history_file):
        print(f"No training history found at {history_file}")
        return
    
    print(f"\n{'='*100}")
    print(f"RECENT TRAINING HISTORY (Last {last_n} episodes)")
    print(f"{'='*100}")
    print(f"{'Episode':>8} | {'Reward':>8} | {'Avg100':>8} | {'Success':>8} | "
          f"{'FireSucc':>9} | {'WaterSucc':>10} | {'Coop':>4} | {'Epsilon':>7}")
    print(f"{'-'*100}")
    
    entries = []
    with open(history_file, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
    
    for entry in entries[-last_n:]:
        print(f"{entry['episode']:8d} | "
              f"{entry['total_reward']:8.1f} | "
              f"{entry['avg_reward_100']:8.1f} | "
              f"{entry['success_rate_100']:7.1%} | "
              f"{entry['fire_success_rate_100']:8.1%} | "
              f"{entry['water_success_rate_100']:9.1%} | "
              f"{entry['cooperation_events']:4d} | "
              f"{entry['fire_epsilon']:7.3f}")
    
    print(f"{'='*100}\n")

def export_training_data(models_dir="models", output_file="training_analysis.csv"):
    """Export training history to CSV for analysis"""
    history_file = f"{models_dir}/training_logs/training_history.jsonl"
    
    if not os.path.exists(history_file):
        print(f"No training history found at {history_file}")
        return
    
    print(f"Exporting training data to {output_file}...")
    
    import csv
    
    entries = []
    with open(history_file, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
    
    if not entries:
        print("No data to export")
        return
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        fieldnames = entries[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)
    
    print(f"Exported {len(entries)} episodes to {output_file}")

def main():
    """Main menu"""
    while True:
        print("\n" + "="*80)
        print("CHECKPOINT MANAGEMENT UTILITY")
        print("="*80)
        print("1. View checkpoint status")
        print("2. View recent training history")
        print("3. Clean old checkpoints (keep latest 3)")
        print("4. Export training data to CSV")
        print("5. Exit")
        print("="*80)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            list_checkpoints()
        elif choice == "2":
            try:
                n = int(input("How many recent episodes to show? (default 20): ").strip() or "20")
            except ValueError:
                n = 20
            view_recent_history(last_n=n)
        elif choice == "3":
            try:
                keep_n = int(input("How many recent checkpoints to keep? (default 3): ").strip() or "3")
            except ValueError:
                keep_n = 3
            confirm = input(f"This will delete old checkpoints, keeping only the latest {keep_n}. Continue? (y/N): ")
            if confirm.lower() == 'y':
                clean_old_checkpoints(keep_latest_n=keep_n)
        elif choice == "4":
            output_file = input("Output filename (default 'training_analysis.csv'): ").strip() or "training_analysis.csv"
            export_training_data(output_file=output_file)
        elif choice == "5":
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice, please try again")

if __name__ == "__main__":
    main()


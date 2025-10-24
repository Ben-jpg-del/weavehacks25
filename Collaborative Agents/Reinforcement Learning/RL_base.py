"""
Fire & Water Game - Clean Framework
Core game logic without RL components
"""

# ---- 1. Imports and Setup ----
import pygame as pg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random
import wandb
import json
import pickle
from datetime import datetime
from collections import deque

# Get script directory for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

# ---- 2. Display Settings and Global Variables ----
W, H = 960, 540
SCALE = 0.8
PANEL_W = 400
SW, SH = int(W * SCALE) + PANEL_W, int(H * SCALE)

S, G, C, FONT_MAIN, FONT_SMALL = None, None, None, None, None
RED, BLUE, GREEN, GRAY, WHITE, BLACK = (220, 60, 60), (60, 120, 255), (60, 200, 120), (170, 170, 170), (240, 240, 240), (15, 15, 20)
YELLOW = (255, 220, 60)
PURPLE = (180, 60, 200)

def init_display(render_mode):
    global S, G, C, FONT_MAIN, FONT_SMALL, pg
    if not render_mode:
        return
    pg.init()
    S = pg.display.set_mode((SW, SH))
    G = pg.Surface((W, H))
    C = pg.time.Clock()
    FONT_MAIN = pg.font.SysFont(None, 22)
    FONT_SMALL = pg.font.SysFont(None, 18)
    pg.display.set_caption("Fire & Water Game")

# ---- 3. Physics-Based Trajectory Prediction ----
def predict_landing_trajectory(agent_rect, initial_velocity, gravity=0.5, max_steps=100):
    """
    Simulate agent's future trajectory to predict landing position.
    Returns: (trajectory_points, landing_pos, steps_to_land)
    """
    trajectory = []
    pos = [float(agent_rect.centerx), float(agent_rect.centery)]
    vel = list(initial_velocity)
    
    for step in range(max_steps):
        # Apply gravity
        vel[1] += gravity
        vel[1] = min(vel[1], 12)  # Terminal velocity
        
        # Update position
        pos[0] += vel[0]
        pos[1] += vel[1]
        
        trajectory.append((pos[0], pos[1]))
        
        # Check if below screen (landed or fell off)
        if pos[1] >= H:
            return trajectory, (pos[0], pos[1]), step + 1
    
    return trajectory, (pos[0], pos[1]), max_steps

def predict_action_landing(agent_rect, agent_velocity, action_id, solids, hazards, agent_type='fire'):
    """
    Predict landing position and safety for a given action.
    Returns: (safe_to_execute, landing_pos, hazards_hit, trajectory)
    """
    # Map action to velocity changes
    action_velocities = {
        0: (0, agent_velocity[1]),           # idle
        1: (-3.4, agent_velocity[1]),        # left
        2: (3.4, agent_velocity[1]),         # right
        3: (agent_velocity[0], -15.73),      # jump (if grounded)
        4: (-3.4, -15.73),                   # left + jump
        5: (3.4, -15.73)                     # right + jump
    }
    
    action_vel = action_velocities.get(action_id, (0, agent_velocity[1]))
    
    # Simulate trajectory
    trajectory, landing_pos, steps = predict_landing_trajectory(
        agent_rect, action_vel, gravity=0.5, max_steps=100
    )
    
    # Check for hazard collisions
    hazards_hit = []
    for point in trajectory:
        for hazard_name, hazard_rect in hazards.items():
            if hazard_rect and hazard_rect.collidepoint(point):
                if agent_type == 'fire' and hazard_name == 'water_pool':
                    hazards_hit.append(hazard_name)
                elif agent_type == 'water' and hazard_name == 'lava_pool':
                    hazards_hit.append(hazard_name)
    
    # Check for solid collisions (landing safety)
    safe_landing = True
    for point in trajectory:
        for solid in solids:
            if solid.collidepoint(point):
                safe_landing = True
                break
        else:
            # No solid collision - check if falling into void
            if point[1] > H:
                safe_landing = False
                break
    
    return safe_landing, landing_pos, hazards_hit, trajectory

# ---- 4. Ray Casting and Clearance Detection ----
def first_intersection_t_with_rect(start, end, rect):
    """Find intersection parameter t for ray-rectangle intersection"""
    import math
    
    # Ray direction
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    if dx == 0 and dy == 0:
        return None
    
    # Ray parameterization: point = start + t * direction
    # For each edge of the rectangle, find intersection
    t_values = []
    
    # Left edge (x = rect.left)
    if dx != 0:
        t = (rect.left - start[0]) / dx
        if t >= 0:
            y = start[1] + t * dy
            if rect.top <= y <= rect.bottom:
                t_values.append(t)
    
    # Right edge (x = rect.right)
    if dx != 0:
        t = (rect.right - start[0]) / dx
        if t >= 0:
            y = start[1] + t * dy
            if rect.top <= y <= rect.bottom:
                t_values.append(t)
    
    # Top edge (y = rect.top)
    if dy != 0:
        t = (rect.top - start[1]) / dy
        if t >= 0:
            x = start[0] + t * dx
            if rect.left <= x <= rect.right:
                t_values.append(t)
    
    # Bottom edge (y = rect.bottom)
    if dy != 0:
        t = (rect.bottom - start[1]) / dy
        if t >= 0:
            x = start[0] + t * dx
            if rect.left <= x <= rect.right:
                t_values.append(t)
    
    return min(t_values) if t_values else None

def get_radial_clearance(agent_rect, solids, hazards=None, max_distance=500, num_rays=18):
    """Get radial clearance distances in multiple directions"""
    import math
    center = agent_rect.center
    clearances = []
    
    obstacles = list(solids)
    if hazards:
        obstacles.extend([h for h in hazards if h is not None])
    
    angle_step = 360.0 / num_rays
    
    for i in range(num_rays):
        angle_degrees = i * angle_step
        angle_radians = math.radians(angle_degrees)
        
        dx = math.cos(angle_radians)
        dy = math.sin(angle_radians)
        
        start = center
        end = (center[0] + dx * max_distance, center[1] + dy * max_distance)
        
        min_distance = max_distance
        closest_point = end
        
        for obstacle in obstacles:
            t = first_intersection_t_with_rect(start, end, obstacle)
            if t is not None and t > 0:
                distance = t * max_distance
                if distance < min_distance:
                    min_distance = distance
                    closest_point = (start[0] + dx * distance, start[1] + dy * distance)
        
        clearances.append((angle_degrees, min_distance, closest_point))
    
    return clearances

# ---- 5. Physics and Movement ----
def move(p, keys, left, right, jump, solids, hazards=None, agent_type='fire'):
    """Apply physics and movement to agent"""
    VMAX, GRAV = 3.4, 0.5
    
    p["v"][0] = (-VMAX if keys.get(left) else VMAX if keys.get(right) else 0)
    if p["g"] and keys.get(jump): 
        p["v"][1] = -15.73
    p["v"][1] += GRAV
    p["v"][1] = min(p["v"][1], 12)
    
    p["r"].x += int(p["v"][0])
    for s in solids:
        if p["r"].colliderect(s):
            if p["v"][0] > 0: p["r"].right = s.left
            elif p["v"][0] < 0: p["r"].left = s.right
    
    p["r"].y += int(p["v"][1])
    p["g"] = False
    for s in solids:
        if p["r"].colliderect(s):
            if p["v"][1] > 0: p["r"].bottom, p["g"] = s.top, True
            else: p["r"].top = s.bottom
            p["v"][1] = 0
    
    # Check for hazard collisions (fire can't stand on water)
    if hazards:
        for hazard_name, hazard_rect in hazards.items():
            if hazard_rect and p["r"].colliderect(hazard_rect):
                if agent_type == 'fire' and hazard_name == 'water_pool':
                    # Fire dies in water - reset to start position
                    p["r"].x = 420  # Reset to start
                    p["r"].y = 380
                    p["v"] = [0.0, 0.0]
                    p["g"] = False

# ---- 6. RL Framework Skeleton (No Specific Rewards) ----
class BaseRLAgent:
    """Base RL agent class - implement your own reward logic"""
    def __init__(self, name, partner_name, state_dim=52):
        self.name = name
        self.partner_name = partner_name
        self.state_dim = state_dim
        self.epsilon = 0.1
        self.plate_activations = 0
        
    def get_enhanced_state_vector(self, agent, partner, level, solids, switches):
        """Get state vector - implement your own state representation"""
        # Basic state vector - customize as needed
        state = np.zeros(self.state_dim)
        
        # Agent position (normalized)
        state[0] = agent["r"].centerx / W
        state[1] = agent["r"].centery / H
        state[2] = agent["v"][0] / 10.0  # velocity x
        state[3] = agent["v"][1] / 10.0  # velocity y
        state[4] = 1.0 if agent["g"] else 0.0  # grounded
        
        # Partner position
        state[5] = partner["r"].centerx / W
        state[6] = partner["r"].centery / H
        state[7] = partner["v"][0] / 10.0
        state[8] = partner["v"][1] / 10.0
        state[9] = 1.0 if partner["g"] else 0.0
        
        # Game state
        state[10] = 1.0 if switches.get('bridge_up', False) else 0.0
        state[11] = 1.0 if switches.get('gate_open', False) else 0.0
        
        # Add clearance data (18 rays)
        clearances = get_radial_clearance(agent["r"], solids, num_rays=18)
        for i, (angle, distance, endpoint) in enumerate(clearances):
            if i < 18:  # Ensure we don't exceed array bounds
                state[12 + i] = distance / 500.0  # Normalize clearance
        
        return state
    
    def get_partner_info(self, partner):
        """Get partner information for cooperation"""
        return {
            'pos': partner["r"].center,
            'vel': partner["v"],
            'grounded': partner["g"]
        }
    
    def select_action(self, state):
        """Select action - implement your own action selection"""
        # Placeholder - implement your RL algorithm here
        return random.randint(0, 5)  # 6 actions: idle, left, right, jump, left+jump, right+jump
    
    def update(self, state, action, reward, next_state, done):
        """Update agent - implement your own learning algorithm"""
        # Placeholder - implement your learning update here
        pass

class BaseTrainer:
    """Base trainer class - implement your own training logic"""
    def __init__(self, fire_agent, water_agent):
        self.fire_agent = fire_agent
        self.water_agent = water_agent
        
    def train_step(self, fire_state, fire_action, fire_reward, fire_next_state, fire_done,
                   water_state, water_action, water_reward, water_next_state, water_done):
        """Single training step - implement your own training logic"""
        # Placeholder - implement your training step here
        pass
    
    def save_checkpoint(self, path, episode, training_state):
        """Save training checkpoint"""
        checkpoint_data = {
            'episode': episode,
            'fire_agent_state': self.fire_agent.__dict__,
            'water_agent_state': self.water_agent.__dict__,
            'training_state': training_state
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(f"{path}_checkpoint.pkl", 'wb') as f:
            pickle.dump(checkpoint_data, f)
        print(f"Checkpoint saved: {path}_checkpoint.pkl")
    
    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint_path = f"{path}_checkpoint.pkl"
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            print(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
        return None

# ---- 7. Level Definition ----
class Level:
    def __init__(self):
        self.name = "Tutorial"
        self.fire_start = (420, 380)
        self.water_start = (60, 384)
        self.geometry = {
            'base_solids': [
                # Boundary walls (10px thick)
                pg.Rect(-10, 0, 10, H),              # Left wall
                pg.Rect(W, 0, 10, H),                # Right wall
                pg.Rect(0, -10, W, 10),              # Top wall
                pg.Rect(0, H, W, 10),                # Bottom wall
                # Platform solids
                pg.Rect(0, 500, W, 40),
                pg.Rect(40, 420, 240, 20),
                pg.Rect(400, 420, 120, 20),
                pg.Rect(700, 420, 260, 20),
                pg.Rect(40, 340, 240, 20),
                pg.Rect(35, 340, 5, 160),
                pg.Rect(925, 0, 5, 420)
            ],
            'black_barrier': pg.Rect(925, 0, 5, 420),
            'bridge': pg.Rect(520, 400, 180, 20), 
            'gate': pg.Rect(280, 340, 20, 120),
            'plateA': pg.Rect(220, 404, 40, 16),
            'plateB': pg.Rect(840, 404, 40, 16),
            'water_pool': pg.Rect(520, 420, 180, 80),
            'exitW': pg.Rect(60, 308, 36, 36), 
            'exitF': pg.Rect(880, 388, 36, 36)
        }
    
    def get_solids(self, bridge_up=False, gate_open=False):
        """Get list of solid obstacles based on current state"""
        s = list(self.geometry['base_solids'])
        if bridge_up: s.append(self.geometry['bridge'])
        if not gate_open: s.append(self.geometry['gate'])
        return s

# ---- 7. Game State Management ----
class GameState:
    def __init__(self):
        self.level = Level()
        self.reset()
        
    def reset(self):
        """Reset game to initial state"""
        self.fire = {"r": pg.Rect(*self.level.fire_start, 28, 36), "v": [0.0, 0.0], "g": False, "c": RED}
        self.water = {"r": pg.Rect(*self.level.water_start, 28, 36), "v": [0.0, 0.0], "g": False, "c": BLUE}
        self.bridge_up = False
        self.gate_open = False

# ---- 8. Rendering Functions ----
def draw_game(state, episode=None, total_episodes=None, metrics=None):
    """Render the game state"""
    if S is None: return
    
    G.fill(BLACK)
    geo = state.level.geometry
    
    # Draw static elements
    for s in geo['base_solids']: pg.draw.rect(G, GRAY, s)
    if 'black_barrier' in geo: pg.draw.rect(G, BLACK, geo['black_barrier'])
    if state.bridge_up: pg.draw.rect(G, GREEN, geo['bridge'])
    if not state.gate_open: pg.draw.rect(G, GREEN, geo['gate'])
    
    # Draw pressure plates with state-based colors
    plate_a_color = (100, 150, 255) if state.bridge_up else (50, 75, 150)
    plate_b_color = (255, 100, 100) if state.gate_open else (150, 50, 50)
    pg.draw.rect(G, plate_a_color, geo['plateA'])
    pg.draw.rect(G, plate_b_color, geo['plateB'])
    
    # Draw special areas
    pg.draw.rect(G, BLUE, geo['water_pool'])
    pg.draw.rect(G, (255, 100, 100), geo['exitF'])
    pg.draw.rect(G, (100, 100, 255), geo['exitW'])
    
    # Draw clearance rays for visualization
    solids = state.level.get_solids(state.bridge_up, state.gate_open)
    hazards = [geo.get('water_pool')]
    
    fire_clearance = get_radial_clearance(state.fire["r"], solids, hazards, num_rays=18)
    water_clearance = get_radial_clearance(state.water["r"], solids, hazards, num_rays=18)
    
    # Draw clearance rays
    for angle, distance, endpoint in fire_clearance:
        color = (200, 100, 50, 128)
        pg.draw.line(G, color[:3], state.fire["r"].center, endpoint, 1)
    
    for angle, distance, endpoint in water_clearance:
        color = (50, 150, 200, 128)
        pg.draw.line(G, color[:3], state.water["r"].center, endpoint, 1)
    
    # Draw center lines to exits
    pg.draw.line(G, (255, 100, 100), state.fire["r"].center, geo['exitF'].center, 1)
    pg.draw.line(G, (100, 100, 255), state.water["r"].center, geo['exitW'].center, 1)
    
    # Draw agents
    pg.draw.rect(G, state.fire["c"], state.fire["r"])
    pg.draw.rect(G, state.water["c"], state.water["r"])
    
    # Render to screen
    S.fill((30, 30, 35))
    scaled_game = pg.transform.scale(G, (int(W*SCALE), int(H*SCALE)))
    S.blit(scaled_game, (0, 0))
    
    # Draw info panel
    panel_x = int(W*SCALE) + 12
    y = 12
    
    S.blit(FONT_MAIN.render("Fire & Water Game", True, YELLOW), (panel_x, y)); y += 30
    
    if episode is not None and total_episodes is not None:
        S.blit(FONT_SMALL.render(f"Episode: {episode}/{total_episodes}", True, WHITE), (panel_x, y)); y += 20
    
    if metrics:
        if 'success_rate' in metrics:
            S.blit(FONT_SMALL.render(f"Success Rate: {metrics.get('success_rate', 0):.1%}", True, WHITE), (panel_x, y)); y += 20
        if 'coop_events' in metrics:
            S.blit(FONT_SMALL.render(f"Cooperation Events: {metrics.get('coop_events', 0)}", True, GREEN), (panel_x, y)); y += 25
    
    # Fire agent data
    fire_dist = ((state.fire["r"].centerx - geo['exitF'].centerx)**2 + 
                 (state.fire["r"].centery - geo['exitF'].centery)**2)**0.5
    S.blit(FONT_SMALL.render(f"Fire Agent", True, RED), (panel_x, y)); y += 18
    S.blit(FONT_SMALL.render(f"Exit Dist: {fire_dist:.1f}px", True, RED), (panel_x, y)); y += 25
    
    # Water agent data
    water_dist = ((state.water["r"].centerx - geo['exitW'].centerx)**2 + 
                  (state.water["r"].centery - geo['exitW'].centery)**2)**0.5
    S.blit(FONT_SMALL.render(f"Water Agent", True, BLUE), (panel_x, y)); y += 18
    S.blit(FONT_SMALL.render(f"Exit Dist: {water_dist:.1f}px", True, BLUE), (panel_x, y)); y += 25
    
    S.blit(FONT_SMALL.render("Press H: Human Play", True, GREEN), (panel_x, y)); y += 20
    S.blit(FONT_SMALL.render("Press R: Reset Episode", True, YELLOW), (panel_x, y)); y += 20
    
    pg.display.flip()

# ---- 9. Human Play Mode ----
def human_play_mode():
    """Human-controlled game mode"""
    init_display(True)
    game = GameState()
    
    print("\n" + "=" * 50)
    print("HUMAN PLAY MODE")
    print("=" * 50)
    print("Controls:")
    print("  Fire (Red):   A/D = move, W = jump")
    print("  Water (Blue): Arrow Keys = move/jump")
    print("  Press 'R' to reset level")
    print("  Press ESC to quit")
    print("=" * 50 + "\n")
    
    running = True
    clock = pg.time.Clock()
    
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                elif event.key == pg.K_r:
                    print("\n[R pressed] Resetting level...")
                    game.reset()
        
        keys = pg.key.get_pressed()
        fire_keys = {pg.K_a: keys[pg.K_a], pg.K_d: keys[pg.K_d], pg.K_w: keys[pg.K_w]}
        water_keys = {pg.K_LEFT: keys[pg.K_LEFT], pg.K_RIGHT: keys[pg.K_RIGHT], pg.K_UP: keys[pg.K_UP]}
        
        solids = game.level.get_solids(game.bridge_up, game.gate_open)
        hazards = {'water_pool': game.level.geometry['water_pool']}
        move(game.fire, fire_keys, pg.K_a, pg.K_d, pg.K_w, solids, hazards, 'fire')
        move(game.water, water_keys, pg.K_LEFT, pg.K_RIGHT, pg.K_UP, solids, hazards, 'water')
        
        geo = game.level.geometry
        # Permanent pressure plates - once activated, they stay activated
        if game.water["r"].colliderect(geo['plateA']):
            game.bridge_up = True
        if game.fire["r"].colliderect(geo['plateB']):
            game.gate_open = True
        
        # Check win/lose conditions
        fire_died = game.fire["r"].colliderect(geo['water_pool'])
        fire_won = geo['exitF'].collidepoint(game.fire["r"].center)
        water_won = geo['exitW'].collidepoint(game.water["r"].center)
        
        if fire_died:
            print("Fire died in water! Resetting...")
            game.reset()
        
        if fire_won and water_won:
            print("\nYOU WIN! Both agents reached their exits!\n")
            game.reset()
        
        # Render the game
        draw_game(game)
        clock.tick(60)
    
    pg.quit()
    return "quit"

# ---- 10. Training Loop Skeleton ----
def train_agents(num_episodes=5000, render=True, use_wandb=True, checkpoint_frequency=50):
    """Training loop skeleton - implement your own training logic"""
    
    init_display(render)
    
    # Create base agents
    fire_agent = BaseRLAgent("fire", "water", state_dim=52)
    water_agent = BaseRLAgent("water", "fire", state_dim=52)
    
    # Create trainer
    trainer = BaseTrainer(fire_agent, water_agent)
    
    # Try to load checkpoint
    print("\nChecking for existing checkpoint...")
    checkpoint_path = os.path.join(MODEL_DIR, "base_agent")
    training_state = trainer.load_checkpoint(checkpoint_path)
    
    if training_state:
        start_episode = training_state['episode'] + 1
        episode_rewards = training_state['training_state'].get('episode_rewards', deque(maxlen=100))
        success_tracker = training_state['training_state'].get('success_tracker', deque(maxlen=100))
        print(f"\n{'='*60}")
        print(f"RESUMING TRAINING from episode {start_episode}")
        print(f"{'='*60}")
    else:
        start_episode = 0
        episode_rewards = deque(maxlen=100)
        success_tracker = deque(maxlen=100)
        print("\nNo checkpoint found. Starting fresh training")
    
    # Initialize W&B
    if use_wandb:
        wandb.init(
            project="firewater-base-framework",
            config={
                "episodes": num_episodes,
                "learning_rate": 3e-4,
                "architecture": "base_framework",
                "state_dim": 52,
                "checkpoint_frequency": checkpoint_frequency
            }
        )
    
    game = GameState()
    
    print("Starting training with base framework...")
    print(f"Checkpoints will be saved every {checkpoint_frequency} episodes")
    print("Press 'S' during training to manually save checkpoint")
    
    for episode in range(start_episode, num_episodes):
        game.reset()
        episode_reward_fire, episode_reward_water = 0, 0
        cooperation_events = 0
        
        for step in range(3000):
            if render:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        print("Window closed by user. Shutting down.")
                        return "quit"
                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_h:
                            print("\nSwitching to Human Play Mode...")
                            if use_wandb and wandb.run:
                                wandb.finish()
                            pg.quit()
                            return "human"
                        elif event.key == pg.K_r:
                            print("\n[R pressed] Resetting current episode...")
                            game.reset()
                            episode_reward_fire, episode_reward_water = 0, 0
                            break
                        elif event.key == pg.K_s:
                            print(f"\n[S pressed] Manually saving checkpoint at episode {episode}...")
                            current_training_state = {
                                'episode_rewards': episode_rewards,
                                'success_tracker': success_tracker
                            }
                            trainer.save_checkpoint(checkpoint_path, episode, current_training_state)
                            print(f"[Manual checkpoint saved successfully at episode {episode}]")
            
            # Get game state
            solids = game.level.get_solids(game.bridge_up, game.gate_open)
            switches = {
                'plateA': game.bridge_up, 
                'plateB': game.gate_open, 
                'bridge_up': game.bridge_up, 
                'gate_open': game.gate_open
            }
            
            # Get state vectors
            fire_state = fire_agent.get_enhanced_state_vector(
                game.fire, game.water, game.level, solids, switches)
            water_state = water_agent.get_enhanced_state_vector(
                game.water, game.fire, game.level, solids, switches)
            
            # Select actions
            fire_action = fire_agent.select_action(fire_state)
            water_action = water_agent.select_action(water_state)
            
            # Apply actions (convert to movement)
            fire_keys = _action_to_keys(fire_action)
            water_keys = _action_to_keys(water_action)
            
            move(game.fire, fire_keys, pg.K_a, pg.K_d, pg.K_w, solids)
            move(game.water, water_keys, pg.K_LEFT, pg.K_RIGHT, pg.K_UP, solids)
            
            # Update game state
            geo = game.level.geometry
            if game.water["r"].colliderect(geo['plateA']):
                game.bridge_up = True
            if game.fire["r"].colliderect(geo['plateB']):
                game.gate_open = True
            
            # Check win/lose conditions
            fire_died = game.fire["r"].colliderect(geo['water_pool'])
            fire_won = geo['exitF'].collidepoint(game.fire["r"].center)
            water_won = geo['exitW'].collidepoint(game.water["r"].center)
            
            # Calculate rewards (implement your own reward logic)
            fire_reward = 0.0  # Implement your reward function here
            water_reward = 0.0  # Implement your reward function here
            
            if fire_died:
                fire_reward = -100.0
                print("Fire died in water!")
                break
            
            if fire_won and water_won:
                fire_reward = 100.0
                water_reward = 100.0
                print("Both agents won!")
                success_tracker.append(1)
                break
            else:
                success_tracker.append(0)
            
            # Get next states
            next_solids = game.level.get_solids(game.bridge_up, game.gate_open)
            fire_next_state = fire_agent.get_enhanced_state_vector(
                game.fire, game.water, game.level, next_solids, switches)
            water_next_state = water_agent.get_enhanced_state_vector(
                game.water, game.fire, game.level, next_solids, switches)
            
            # Update agents (implement your learning algorithm)
            fire_agent.update(fire_state, fire_action, fire_reward, fire_next_state, fire_died or (fire_won and water_won))
            water_agent.update(water_state, water_action, water_reward, water_next_state, fire_died or (fire_won and water_won))
            
            # Training step
            trainer.train_step(fire_state, fire_action, fire_reward, fire_next_state, fire_died or (fire_won and water_won),
                              water_state, water_action, water_reward, water_next_state, fire_died or (fire_won and water_won))
            
            episode_reward_fire += fire_reward
            episode_reward_water += water_reward
            
            # Render
            if render:
                metrics = {
                    'success_rate': np.mean(success_tracker) if success_tracker else 0,
                    'coop_events': cooperation_events
                }
                draw_game(game, episode, num_episodes, metrics)
        
        # Episode logging
        episode_rewards.append(episode_reward_fire + episode_reward_water)
        
        if use_wandb and wandb.run:
            wandb.log({
                "episode": episode,
                "fire_reward": episode_reward_fire,
                "water_reward": episode_reward_water,
                "total_reward": episode_reward_fire + episode_reward_water,
                "success_rate": np.mean(success_tracker) if success_tracker else 0,
                "cooperation_events": cooperation_events
            })
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            success_rate = np.mean(success_tracker) if success_tracker else 0
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Success Rate = {success_rate:.2%}")
        
        # Save checkpoint
        if episode % checkpoint_frequency == 0 and episode > 0:
            current_training_state = {
                'episode_rewards': episode_rewards,
                'success_tracker': success_tracker
            }
            trainer.save_checkpoint(checkpoint_path, episode, current_training_state)
    
    if use_wandb and wandb.run:
        wandb.finish()
    
    pg.quit()
    return "completed"

def _action_to_keys(action):
    """Convert action ID to key dictionary"""
    action_map = {
        0: {},  # idle
        1: {pg.K_LEFT: True},  # left
        2: {pg.K_RIGHT: True},  # right
        3: {pg.K_UP: True},  # jump
        4: {pg.K_LEFT: True, pg.K_UP: True},  # left + jump
        5: {pg.K_RIGHT: True, pg.K_UP: True}  # right + jump
    }
    return action_map.get(action, {})

# ---- 11. Main Entry Point ----
def main():
    """Main entry point for the game"""
    print("Fire & Water Game - Clean Framework")
    print("Choose mode:")
    print("1. Human Play Mode")
    print("2. Training Mode (RL Framework)")
    
    while True:
        try:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == "1":
                print("Starting Human Play Mode...")
                human_play_mode()
                break
            elif choice == "2":
                print("Starting Training Mode...")
                train_agents(num_episodes=1000, render=True, use_wandb=True, checkpoint_frequency=50)
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
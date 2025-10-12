"""
Cooperative Reinforcement Learning Agent for Fire & Water Game
Hybrid Parameter Sharing with Cooperative Penalties - v3.0

NEW FEATURES:
- Shared backbone network synchronized between agents
- Partner-aware state vectors (32-dimensional)
- Cooperative training with mutual penalties
- 18-directional clearance integration
- Synchronized weight updates
"""

# ---- 1. Imports and Setup ----
import pygame as pg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from collections import deque
import random
import wandb

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
    pg.display.set_caption("Cooperative RL Training")

# ---- 3. NEW: Physics-Based Trajectory Prediction ----
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
    
    # Check for hazard collisions along trajectory
    hazards_hit = []
    safe = True
    
    for point in trajectory:
        point_rect = pg.Rect(int(point[0]) - 14, int(point[1]) - 18, 28, 36)
        
        for hazard_name, hazard_rect in hazards.items():
            if hazard_rect and point_rect.colliderect(hazard_rect):
                # Fire agents die in water, water agents die in lava
                if (agent_type == 'fire' and 'water' in hazard_name) or \
                   (agent_type == 'water' and 'lava' in hazard_name):
                    hazards_hit.append(hazard_name)
                    safe = False
                    break
        
        if not safe:
            break
    
    return safe, landing_pos, hazards_hit, trajectory

def get_directional_hazard_info(agent_rect, hazards, agent_type='fire', num_directions=8):
    """
    Calculate hazard proximity in 8 cardinal/ordinal directions.
    Returns array of [distance_to_hazard] in each direction (0-1 normalized).
    """
    import math
    center = agent_rect.center
    max_check_distance = 300.0
    
    hazard_proximities = []
    angle_step = 360.0 / num_directions
    
    # Filter hazards relevant to this agent type
    relevant_hazards = []
    for hazard_name, hazard_rect in hazards.items():
        if hazard_rect:
            if (agent_type == 'fire' and 'water' in hazard_name) or \
               (agent_type == 'water' and 'lava' in hazard_name):
                relevant_hazards.append(hazard_rect)
    
    for i in range(num_directions):
        angle_degrees = i * angle_step
        angle_radians = math.radians(angle_degrees)
        
        dx = math.cos(angle_radians)
        dy = math.sin(angle_radians)
        
        min_distance = max_check_distance
        
        # Check distance to hazards in this direction
        for hazard in relevant_hazards:
            # Simple ray-box intersection
            t = first_intersection_t_with_rect(
                center,
                (center[0] + dx * max_check_distance, center[1] + dy * max_check_distance),
                hazard
            )
            if t is not None:
                distance = t * max_check_distance
                min_distance = min(min_distance, distance)
        
        # Normalize to 0-1 (1 = far/safe, 0 = close/danger)
        normalized_distance = min_distance / max_check_distance
        hazard_proximities.append(normalized_distance)
    
    return np.array(hazard_proximities, dtype=np.float32)

def calculate_hazard_proximity(agent_rect, hazards, agent_type='fire'):
    """
    Calculate minimum distance to deadly hazards.
    Returns normalized distance (0 = touching hazard, 1 = far away).
    """
    min_distance = 1000.0
    
    for hazard_name, hazard_rect in hazards.items():
        if hazard_rect is not None:  # Check for None instead of using truthiness
            # Fire dies in water, water dies in lava
            if (agent_type == 'fire' and 'water' in hazard_name) or \
               (agent_type == 'water' and 'lava' in hazard_name):
                # Calculate distance from agent center to hazard
                dx = agent_rect.centerx - hazard_rect.centerx
                dy = agent_rect.centery - hazard_rect.centery
                distance = np.sqrt(dx**2 + dy**2)
                min_distance = min(min_distance, distance)
    
    # Normalize (0-1, where 1 is safe and far)
    return min(min_distance / 500.0, 1.0)

# ---- 4. NEW: Safety-Aware Neural Network ----
class SharedBackboneNetwork(nn.Module):
    """Shared feature extraction backbone for cooperative learning"""
    def __init__(self, state_dim=52, hidden_dim=256):  # Updated from 32 to 52
        super(SharedBackboneNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
    
    def forward(self, state):
        return self.shared_layers(state)

class CooperativeActorCritic(nn.Module):
    """Agent network with shared backbone + individual decision heads + safety awareness"""
    def __init__(self, agent_name, shared_backbone, state_dim=52):  # Updated from 30 to 52
        super(CooperativeActorCritic, self).__init__()
        self.agent_name = agent_name
        self.shared_backbone = shared_backbone
        
        # Partner information encoder
        self.partner_encoder = nn.Linear(8, 8)
        
        # Individual decision-making head
        self.agent_head = nn.Linear(128 + 8, 64)
        
        # Policy and value heads
        self.actor = nn.Linear(64, 6)  # 6 actions
        self.critic = nn.Linear(64, 1)  # Value estimation
    
    def forward(self, state, partner_info):
        # Shared feature extraction
        shared_features = self.shared_backbone(state)
        
        # Partner-aware processing
        partner_features = F.relu(self.partner_encoder(partner_info))
        combined = torch.cat([shared_features, partner_features], dim=-1)
        
        # Agent-specific decision making
        agent_features = F.relu(self.agent_head(combined))
        
        action_logits = self.actor(agent_features)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.critic(agent_features)
        
        return action_probs, value

# ---- 4. Enhanced Replay Buffer ----
class CooperativeReplayBuffer:
    """Stores experiences with partner information"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, partner_info, action, reward, next_state, next_partner_info, done):
        self.buffer.append((state, partner_info, action, reward, next_state, next_partner_info, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, partner_infos, actions, rewards, next_states, next_partner_infos, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(partner_infos)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(next_partner_infos)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)

# ---- 5. NEW: Safety-Aware RL Agent ----
class CooperativeRLAgent:
    """RL Agent with partner awareness, cooperative training, and safety prediction"""
    def __init__(self, agent_name, partner_name, shared_backbone, state_dim=52):  # Updated from 30 to 52
        self.agent_name = agent_name
        self.partner_name = partner_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network with shared backbone
        self.policy_net = CooperativeActorCritic(agent_name, shared_backbone, state_dim).to(self.device)
        
        # Note: Optimizer will be managed by CooperativeTrainer
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.replay_buffer = CooperativeReplayBuffer()
        
        # Cooperation metrics
        self.cooperation_events = 0
        self.plate_activations = 0
        
        # Safety metrics
        self.safety_violations = 0
        self.safe_actions_taken = 0
    
    def get_safe_action_mask(self, player, solids, hazards):
        """
        Generate binary mask for safe actions (1 = safe, 0 = unsafe).
        Returns 6-element array for each action.
        """
        mask = np.ones(6, dtype=np.float32)
        
        for action_id in range(6):
            safe, landing_pos, hazards_hit, trajectory = predict_action_landing(
                player["r"], player["v"], action_id, solids, hazards, self.agent_name
            )
            if not safe:
                mask[action_id] = 0.0  # Mask out unsafe actions
        
        return mask
    
    def get_action_safety_predictions(self, player, solids, hazards):
        """
        Get safety prediction for each action (6 dimensions).
        Returns: [safety_score_action_0, ..., safety_score_action_5]
        where 1.0 = safe, 0.0 = definitely unsafe
        """
        safety_predictions = []
        
        for action_id in range(6):
            safe, landing_pos, hazards_hit, trajectory = predict_action_landing(
                player["r"], player["v"], action_id, solids, hazards, self.agent_name
            )
            safety_score = 1.0 if safe else 0.0
            safety_predictions.append(safety_score)
        
        return np.array(safety_predictions, dtype=np.float32)
    
    def get_partner_info(self, partner, level, solids, switches):
        """Extract 8-dimensional partner information vector"""
        geo = level.geometry
        partner_exit_key = 'exitW' if self.partner_name == 'water' else 'exitF'
        partner_exit = geo[partner_exit_key]
        
        # Partner goal progress
        partner_dist = np.sqrt((partner["r"].centerx - partner_exit.centerx)**2 + 
                               (partner["r"].centery - partner_exit.centery)**2)
        partner_progress = 1.0 - min(partner_dist / 1000.0, 1.0)
        
        # Cooperation urgency (distance to needed plates)
        if self.agent_name == 'fire':
            # Fire needs water to activate plateA for bridge
            urgency_dist = np.sqrt((partner["r"].centerx - geo['plateA'].centerx)**2 + 
                                   (partner["r"].centery - geo['plateA'].centery)**2)
        else:
            # Water needs fire to activate plateB for gate
            urgency_dist = np.sqrt((partner["r"].centerx - geo['plateB'].centerx)**2 + 
                                   (partner["r"].centery - geo['plateB'].centery)**2)
        cooperation_urgency = 1.0 - min(urgency_dist / 500.0, 1.0)
        
        # Partner can help (proximity and capability)
        can_help = 1.0 if urgency_dist < 150 else 0.0
        
        # Relative efficiency (simplified)
        relative_efficiency = partner_progress
        
        partner_info = np.array([
            partner["r"].centerx / W,
            partner["r"].centery / H,
            partner["v"][0] / 10.0,
            partner["v"][1] / 20.0,
            partner_progress,
            cooperation_urgency,
            can_help,
            relative_efficiency
        ], dtype=np.float32)
        
        return partner_info
    
    def process_clearance_data(self, clearances):
        """Convert 18-directional clearance into 8 summary features"""
        if not clearances:
            return np.zeros(8, dtype=np.float32)
        
        angles, distances, _ = zip(*clearances)
        distances = np.array(distances)
        angles = np.array(angles)
        
        # Group by cardinal/ordinal directions (8 sectors)
        sectors = []
        for i in range(8):
            sector_start = i * 45
            sector_end = (i + 1) * 45
            mask = (angles >= sector_start) & (angles < sector_end)
            sector_clearance = np.mean(distances[mask]) if np.any(mask) else 500.0
            sectors.append(min(sector_clearance / 500.0, 1.0))
        
        return np.array(sectors, dtype=np.float32)
    
    def get_enhanced_state_vector(self, player, partner, level, solids, switches):
        """
        Enhanced state vector with safety predictions and hazard awareness.
        Total: 30 (base) + 6 (safety) + 8 (hazard directions) = 44 dimensions
        """
        geo = level.geometry
        exit_key = 'exitF' if self.agent_name == 'fire' else 'exitW'
        exit_rect = geo[exit_key]
        
        # Get assigned plate for this agent
        plate_key = 'plateB' if self.agent_name == 'fire' else 'plateA'
        plate_rect = geo[plate_key]
        
        # DEBUG: Print plate assignment (only occasionally to avoid spam)
        if random.random() < 0.001:  # Print 0.1% of the time
            print(f"{self.agent_name} agent -> plate: {plate_key} at x={plate_rect.centerx}, player at x={player['r'].centerx}")
        
        # Collect hazards
        hazards = {
            'water_pool': geo.get('water_pool'),
            'lava_pool': geo.get('lava_pool')
        }
        
        # Calculate directional vectors to goal
        dx_to_goal = (exit_rect.centerx - player["r"].centerx) / W
        dy_to_goal = (exit_rect.centery - player["r"].centery) / H
        dist_to_goal = np.sqrt(dx_to_goal**2 + dy_to_goal**2)
        
        # Calculate directional vectors to assigned plate (PRIORITY TASK)
        dx_to_plate = (plate_rect.centerx - player["r"].centerx) / W
        dy_to_plate = (plate_rect.centery - player["r"].centery) / H
        dist_to_plate = np.sqrt(dx_to_plate**2 + dy_to_plate**2)
        
        # Calculate angles
        angle_to_goal = np.arctan2(dy_to_goal, dx_to_goal) / np.pi
        angle_to_plate = np.arctan2(dy_to_plate, dx_to_plate) / np.pi
        
        # Check if both plates are active
        both_plates_active = switches.get('plateA', False) and switches.get('plateB', False)
        
        # Calculate hazard proximity
        hazard_proximity = calculate_hazard_proximity(player["r"], hazards, self.agent_name)
        
        # Base state (30 dimensions - reduced from 32 to fit safety additions)
        base_state = np.array([
            player["r"].centerx / W,
            player["r"].centery / H,
            player["v"][0] / 10.0,
            player["v"][1] / 20.0,
            dist_to_plate,  # PRIORITY: Distance to assigned plate
            float(player["g"]),  # On ground indicator
            dx_to_plate,  # PRIORITY: Directional X to plate
            dy_to_plate,  # PRIORITY: Directional Y to plate
            angle_to_plate,  # PRIORITY: Angle to plate
            1.0 if switches.get('plateA', False) else 0.0,
            1.0 if switches.get('plateB', False) else 0.0,
            1.0 if both_plates_active else 0.0,
            dist_to_goal,  # Secondary: Distance to goal
            angle_to_goal,  # Secondary: Angle to goal
            min(self.count_walls_between(player["r"], plate_rect, solids) / 5.0, 1.0),
            abs(player["v"][0]) / 10.0,  # Speed magnitude
            hazard_proximity,  # Overall hazard proximity
            dx_to_goal,  # Direction to goal X
            dy_to_goal,  # Direction to goal Y
            float(player["r"].left < 50 or player["r"].right > W - 50),  # Near edge
            float(player["r"].top < 50),  # Near top
            float(player["r"].bottom > H - 100),  # Near bottom
            # Padding to reach 30
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        # Safety predictions per action (6 dimensions)
        safety_predictions = self.get_action_safety_predictions(player, solids, hazards)
        
        # Directional hazard info (8 dimensions)
        hazard_directions = get_directional_hazard_info(player["r"], hazards, self.agent_name, num_directions=8)
        
        # Partner information (8 dimensions)
        partner_info = self.get_partner_info(partner, level, solids, switches)
        
        # Combine all features: 30 + 6 + 8 + 8 = 52 dimensions total
        enhanced_state = np.concatenate([base_state, safety_predictions, hazard_directions, partner_info])
        
        return enhanced_state
    
    def count_walls_between(self, start_rect, end_rect, solids):
        """Count walls blocking direct path"""
        start_center = start_rect.center
        end_center = end_rect.center
        wall_count = 0
        for solid in solids:
            if line_intersects_rect(start_center, end_center, solid):
                wall_count += 1
        return wall_count
    
    def action_to_keys(self, action_idx):
        actions = {
            0: [], 1: ['left'], 2: ['right'], 3: ['jump'],
            4: ['left', 'jump'], 5: ['right', 'jump']
        }
        return actions.get(action_idx, [])

    def select_safe_action(self, state, partner_info, state_data, training=True):
        """
        Select action with safety masking.
        state_data: dict with 'player', 'solids', 'hazards'
        """
        # Get safety mask
        safety_mask = self.get_safe_action_mask(
            state_data['player'], 
            state_data['solids'], 
            state_data['hazards']
        )
        
        # Epsilon-greedy with safety constraints
        if training and random.random() < self.epsilon:
            # Random action from safe actions only
            safe_actions = np.where(safety_mask > 0)[0]
            if len(safe_actions) > 0:
                action = np.random.choice(safe_actions)
                self.safe_actions_taken += 1
                return action
            else:
                # No safe actions, pick least bad (this shouldn't happen often)
                self.safety_violations += 1
                return random.randint(0, 5)
        
        # Policy-based action with safety masking
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            partner_tensor = torch.FloatTensor(partner_info).unsqueeze(0).to(self.device)
            action_probs, _ = self.policy_net(state_tensor, partner_tensor)
            
            # Apply safety mask to probabilities
            action_probs_np = action_probs.cpu().numpy().flatten()
            masked_probs = action_probs_np * safety_mask
            
            # Renormalize if we have any safe actions
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
                action = np.random.choice(6, p=masked_probs)
                self.safe_actions_taken += 1
            else:
                # Fallback: use original policy (all actions unsafe)
                action = torch.multinomial(action_probs, 1).item()
                self.safety_violations += 1
            
            return action

    def select_action(self, state, partner_info, training=True):
        """Legacy method for backward compatibility"""
        if training and random.random() < self.epsilon:
            return random.randint(0, 5)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            partner_tensor = torch.FloatTensor(partner_info).unsqueeze(0).to(self.device)
            action_probs, _ = self.policy_net(state_tensor, partner_tensor)
            action = torch.multinomial(action_probs, 1).item()
            return action

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ---- 6. NEW: Cooperative Trainer ----
class CooperativeTrainer:
    """Synchronized training system with mutual penalties"""
    def __init__(self, fire_agent, water_agent, lr=3e-4):
        self.fire_agent = fire_agent
        self.water_agent = water_agent
        self.device = fire_agent.device
        
        # Get shared backbone parameters using ID-based filtering
        shared_backbone_ids = {id(p) for p in fire_agent.policy_net.shared_backbone.parameters()}
        
        # Shared backbone optimizer (updates both agents simultaneously)
        shared_params = list(fire_agent.policy_net.shared_backbone.parameters())
        self.shared_optimizer = optim.Adam(shared_params, lr=lr)
        
        # Individual head optimizers - filter by parameter ID
        fire_individual_params = [p for p in fire_agent.policy_net.parameters() 
                                  if id(p) not in shared_backbone_ids]
        water_individual_params = [p for p in water_agent.policy_net.parameters() 
                                   if id(p) not in shared_backbone_ids]
        
        self.fire_optimizer = optim.Adam(fire_individual_params, lr=lr)
        self.water_optimizer = optim.Adam(water_individual_params, lr=lr)
        
        # Penalty weights
        self.COOPERATION_PENALTY = 0.3
        self.MUTUAL_FAILURE_PENALTY = 0.5
        self.SUCCESS_DISPARITY_PENALTY = 0.2
    
    def calculate_agent_loss(self, agent, states, partner_infos, actions, rewards, 
                            next_states, next_partner_infos, dones):
        """Calculate actor-critic loss for one agent"""
        action_probs, values = agent.policy_net(states, partner_infos)
        values = values.squeeze()
        
        with torch.no_grad():
            _, next_values = agent.policy_net(next_states, next_partner_infos)
            next_values = next_values.squeeze()
            target_values = rewards + agent.gamma * next_values * (1 - dones)
        
        advantages = target_values - values
        
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-9)
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, target_values)
        
        return actor_loss + 0.5 * critic_loss
    
    def calculate_cooperation_penalty(self, cooperation_metrics):
        """Calculate mutual penalty for poor cooperation"""
        penalty = torch.tensor(0.0, device=self.device)
        
        # Penalty for success disparity (one agent consistently failing)
        if 'success_disparity' in cooperation_metrics:
            penalty += self.SUCCESS_DISPARITY_PENALTY * cooperation_metrics['success_disparity']
        
        # Penalty for not activating needed plates
        if cooperation_metrics.get('bridge_needed', False) and not cooperation_metrics.get('plate_a_activated', False):
            penalty += self.COOPERATION_PENALTY
            
        if cooperation_metrics.get('gate_needed', False) and not cooperation_metrics.get('plate_b_activated', False):
            penalty += self.COOPERATION_PENALTY
        
        # Penalty for mutual failure
        if cooperation_metrics.get('both_failed', False):
            penalty += self.MUTUAL_FAILURE_PENALTY
        
        return penalty
    
    def synchronized_training_step(self, batch_size=64, cooperation_metrics=None):
        """Joint training with mutual penalties through shared backbone"""
        if len(self.fire_agent.replay_buffer) < batch_size or len(self.water_agent.replay_buffer) < batch_size:
            return None
        
        if cooperation_metrics is None:
            cooperation_metrics = {}
        
        # Sample batches for both agents
        fire_batch = self.fire_agent.replay_buffer.sample(batch_size)
        water_batch = self.water_agent.replay_buffer.sample(batch_size)
        
        fire_states, fire_partner_infos, fire_actions, fire_rewards, fire_next_states, fire_next_partner_infos, fire_dones = fire_batch
        water_states, water_partner_infos, water_actions, water_rewards, water_next_states, water_next_partner_infos, water_dones = water_batch
        
        # Move to device
        fire_states = fire_states.to(self.device)
        fire_partner_infos = fire_partner_infos.to(self.device)
        fire_actions = fire_actions.to(self.device)
        fire_rewards = fire_rewards.to(self.device)
        fire_next_states = fire_next_states.to(self.device)
        fire_next_partner_infos = fire_next_partner_infos.to(self.device)
        fire_dones = fire_dones.to(self.device)
        
        water_states = water_states.to(self.device)
        water_partner_infos = water_partner_infos.to(self.device)
        water_actions = water_actions.to(self.device)
        water_rewards = water_rewards.to(self.device)
        water_next_states = water_next_states.to(self.device)
        water_next_partner_infos = water_next_partner_infos.to(self.device)
        water_dones = water_dones.to(self.device)
        
        # Calculate individual losses
        fire_loss = self.calculate_agent_loss(self.fire_agent, fire_states, fire_partner_infos, 
                                              fire_actions, fire_rewards, fire_next_states, 
                                              fire_next_partner_infos, fire_dones)
        water_loss = self.calculate_agent_loss(self.water_agent, water_states, water_partner_infos,
                                               water_actions, water_rewards, water_next_states,
                                               water_next_partner_infos, water_dones)
        
        # Calculate cooperation penalty
        coop_penalty = self.calculate_cooperation_penalty(cooperation_metrics)
        
        # Joint loss (both agents penalized through shared backbone)
        total_loss = fire_loss + water_loss + coop_penalty
        
        # Zero all gradients before backward pass
        self.shared_optimizer.zero_grad()
        self.fire_optimizer.zero_grad()
        self.water_optimizer.zero_grad()
        
        # Single backward pass on total loss (no retain_graph needed)
        total_loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.fire_agent.policy_net.shared_backbone.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.fire_optimizer.param_groups[0]['params'], 1.0)
        torch.nn.utils.clip_grad_norm_(self.water_optimizer.param_groups[0]['params'], 1.0)
        
        # Update all parameters simultaneously
        self.shared_optimizer.step()
        self.fire_optimizer.step()
        self.water_optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'fire_loss': fire_loss.item(),
            'water_loss': water_loss.item(),
            'coop_penalty': coop_penalty.item()
        }
    
    def save_models(self, filepath_prefix):
        """Save both agent models and shared backbone"""
        os.makedirs(os.path.dirname(filepath_prefix) or '.', exist_ok=True)
        
        # Save shared backbone
        torch.save(self.fire_agent.policy_net.shared_backbone.state_dict(), 
                  f"{filepath_prefix}_shared_backbone.pt")
        
        # Save individual agent heads
        torch.save({
            'partner_encoder': self.fire_agent.policy_net.partner_encoder.state_dict(),
            'agent_head': self.fire_agent.policy_net.agent_head.state_dict(),
            'actor': self.fire_agent.policy_net.actor.state_dict(),
            'critic': self.fire_agent.policy_net.critic.state_dict()
        }, f"{filepath_prefix}_fire_head.pt")
        
        torch.save({
            'partner_encoder': self.water_agent.policy_net.partner_encoder.state_dict(),
            'agent_head': self.water_agent.policy_net.agent_head.state_dict(),
            'actor': self.water_agent.policy_net.actor.state_dict(),
            'critic': self.water_agent.policy_net.critic.state_dict()
        }, f"{filepath_prefix}_water_head.pt")
        
        print(f"Cooperative models saved to {filepath_prefix}_*.pt")
    
    def load_models(self, filepath_prefix):
        """Load both agent models and shared backbone"""
        try:
            # Load shared backbone
            backbone_path = f"{filepath_prefix}_shared_backbone.pt"
            if os.path.exists(backbone_path):
                # Check if the saved model has the correct dimensions
                saved_state = torch.load(backbone_path, map_location=self.device)
                first_layer_key = 'shared_layers.0.weight'
                if first_layer_key in saved_state:
                    saved_input_dim = saved_state[first_layer_key].shape[1]
                    current_input_dim = self.fire_agent.policy_net.shared_backbone.shared_layers[0].in_features
                    
                    if saved_input_dim != current_input_dim:
                        print(f"WARNING: Saved model has input dimension {saved_input_dim}, but current model expects {current_input_dim}")
                        print(f"Skipping model loading - starting fresh training with new architecture")
                        return False
                
                self.fire_agent.policy_net.shared_backbone.load_state_dict(saved_state)
                print(f"Loaded shared backbone from {backbone_path}")
            else:
                print(f"No saved shared backbone found at {backbone_path} - starting fresh")
                return False
            
            # Load fire agent head
            fire_head_path = f"{filepath_prefix}_fire_head.pt"
            if os.path.exists(fire_head_path):
                fire_state = torch.load(fire_head_path, map_location=self.device)
                self.fire_agent.policy_net.partner_encoder.load_state_dict(fire_state['partner_encoder'])
                self.fire_agent.policy_net.agent_head.load_state_dict(fire_state['agent_head'])
                self.fire_agent.policy_net.actor.load_state_dict(fire_state['actor'])
                self.fire_agent.policy_net.critic.load_state_dict(fire_state['critic'])
                print(f"Loaded fire agent head from {fire_head_path}")
            else:
                print(f"No saved fire head found - starting fresh")
                return False
            
            # Load water agent head
            water_head_path = f"{filepath_prefix}_water_head.pt"
            if os.path.exists(water_head_path):
                water_state = torch.load(water_head_path, map_location=self.device)
                self.water_agent.policy_net.partner_encoder.load_state_dict(water_state['partner_encoder'])
                self.water_agent.policy_net.agent_head.load_state_dict(water_state['agent_head'])
                self.water_agent.policy_net.actor.load_state_dict(water_state['actor'])
                self.water_agent.policy_net.critic.load_state_dict(water_state['critic'])
                print(f"Loaded water agent head from {water_head_path}")
            else:
                print(f"No saved water head found - starting fresh")
                return False
            
            # Reset epsilon for evaluation
            self.fire_agent.epsilon = self.fire_agent.epsilon_min
            self.water_agent.epsilon = self.water_agent.epsilon_min
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Starting fresh training with new architecture")
            return False

# ---- 7. Game Logic (Keep existing functions) ----
class RewardCalculator:
    def __init__(self):
        self.DEATH_PENALTY = -200.0  # Increased from -100.0
        self.SUCCESS_REWARD = 300.0  # Increased from 200.0
        self.DISTANCE_PROGRESS_SCALE = 1.5
        self.PLATE_ACTIVATION_REWARD = 50.0  # Increased from 25.0
        self.PLATE_STAY_REWARD = 5.0  # INCREASED: Much stronger reward for staying on plate
        self.BOTH_PLATES_ACTIVE_BONUS = 5.0  # NEW: Bonus per step when both plates active
        self.GOAL_PROXIMITY_BONUS = 2.0
        self.PLATE_PROXIMITY_BONUS = 15.0  # MASSIVELY INCREASED: Ultra-strong bonus for moving toward plate
        self.TIME_PENALTY = -0.05
        self.STUCK_PENALTY = -0.1
        self.COOPERATION_BONUS = 15.0  # Increased from 10.0
        self.JUMP_PENALTY = -0.3
        self.AIRBORNE_PENALTY = -0.15
        self.JUMP_ON_PLATE_PENALTY = -2.0  # NEW: Strong penalty for jumping when on plate
        self.HORIZONTAL_PROGRESS_BONUS = 1.0
        self.COLLABORATIVE_PHASE_COMPLETE_REWARD = 100.0  # NEW: Huge reward when both plates active
        
        # NEW: Safety-related rewards/penalties
        self.HAZARD_PROXIMITY_PENALTY_SCALE = 2.0  # Penalty for being near hazards
        self.SAFE_PROGRESS_BONUS = 0.5  # Bonus for making progress while staying safe
        self.UNSAFE_ACTION_PENALTY = -5.0  # Penalty for taking actions predicted to be unsafe
        
        # REDUCED: Open space bonus (was too high)
        self.OPEN_SPACE_BONUS = 0.0001  # NEARLY ELIMINATED - plates are priority
        # self.WRONG_DIRECTION_PENALTY = -5.0  # REMOVED: No longer using arbitrary directional bias

    def calculate_reward(self, old_state, new_state, agent_specific_data, agent_name='unknown'):
        reward = self.TIME_PENALTY
        
        if agent_specific_data['died']: return self.DEATH_PENALTY
        if agent_specific_data['success']: return self.SUCCESS_REWARD
        
        # Determine if collaborative phase is complete (both plates active)
        both_plates_active = agent_specific_data.get('both_plates_active', False)
        collaborative_phase_just_completed = (not agent_specific_data.get('both_plates_were_active', False) 
                                             and both_plates_active)
        
        # PHASE 1: Collaborative Task - Activate Both Plates (HIGHEST PRIORITY)
        if not both_plates_active:
            # Before both plates are active, ONLY reward plate-related progress
            
            # DEBUG: Print plate progress for troubleshooting
            plate_dist_delta = old_state.get('plate_dist', 1000) - new_state.get('plate_dist', 1000)
            if random.random() < 0.01:  # Print 1% of the time
                plate_dist = new_state.get('plate_dist', 0)
                target_plate = 'plateB (RIGHT/RED x=840)' if agent_name == 'fire' else 'plateA (LEFT/BLUE x=220)'
                print(f"[{agent_name}] -> {target_plate} | dist={plate_dist:.1f} | delta={plate_dist_delta:.1f} | reward={plate_dist_delta * self.PLATE_PROXIMITY_BONUS:.2f}")
            
            # Strong reward for moving toward assigned plate
            if plate_dist_delta > 0:  # Moving closer to plate
                reward += plate_dist_delta * self.PLATE_PROXIMITY_BONUS
            
            # Direction is now entirely objective-based (plate proximity) rather than arbitrary directional bias
            
            # Massive reward for staying on plate during collaborative phase
            if agent_specific_data.get('on_plate', False):
                reward += self.PLATE_STAY_REWARD * 2  # Double reward during collaborative phase
            
            # Plate activation mega-reward
            if new_state['plates_active'] > old_state['plates_active']:
                reward += self.PLATE_ACTIVATION_REWARD
            
            # STRONGER penalty for moving away from plate
            if plate_dist_delta < -5:  # Moving away from plate
                reward -= 5.0  # Increased from -1.0
            
            # Medium penalty for being far from plate (urgency signal)
            if new_state.get('plate_dist', 0) > 300:
                reward -= 1.0  # Increased from -0.2
                
        else:
            # PHASE 2: Both Plates Active - Now prioritize reaching goals
            
            # Continuous bonus for maintaining both plates active
            reward += self.BOTH_PLATES_ACTIVE_BONUS
            
            # If we JUST completed collaborative phase, huge one-time reward
            if collaborative_phase_just_completed:
                reward += self.COLLABORATIVE_PHASE_COMPLETE_REWARD
            
            # NOW we reward distance progress toward goal
            dist_delta = old_state['dist'] - new_state['dist']
            reward += dist_delta * self.DISTANCE_PROGRESS_SCALE
            
            # Horizontal progress bonus (ground movement toward goal)
            if agent_specific_data.get('on_ground', False):
                horizontal_delta = old_state.get('horizontal_dist', 0) - new_state.get('horizontal_dist', 0)
                if horizontal_delta > 0:
                    reward += horizontal_delta * self.HORIZONTAL_PROGRESS_BONUS
            
            # Goal proximity bonus (exponential - gets stronger near goal)
            if new_state['dist'] < 200:
                proximity_bonus = self.GOAL_PROXIMITY_BONUS * (1.0 - new_state['dist'] / 200.0)
                reward += proximity_bonus
            
            # Being in goal area reward
            if agent_specific_data.get('in_goal', False):
                reward += 3.0  # Strong incentive to stay in goal
        
        # Universal rewards/penalties (apply in both phases)
        
        # Stuck penalty
        if new_state['pos'] == old_state['pos']:
            reward += self.STUCK_PENALTY
        
        # Jump penalty (discourages spam jumping)
        if agent_specific_data.get('jumped', False):
            reward += self.JUMP_PENALTY
            
            # EXTRA penalty for jumping when on plate (should stay still!)
            if agent_specific_data.get('on_plate', False):
                reward += self.JUMP_ON_PLATE_PENALTY
        
        # Airborne penalty (discourages staying in air)
        if not agent_specific_data.get('on_ground', False):
            reward += self.AIRBORNE_PENALTY
        
        # Cooperation bonus (helping partner with their plate)
        if agent_specific_data.get('helped_partner', False):
            reward += self.COOPERATION_BONUS
        
        # Small open-space bonus (reduced emphasis compared to plate objectives)
        clearances = agent_specific_data.get('clearances', [])
        if clearances:
            # Calculate average clearance across all 18 directions
            avg_clearance = sum([dist for angle, dist, endpoint in clearances]) / len(clearances)
            # Normalize to 0-1 range (max_distance is 500)
            normalized_clearance = avg_clearance / 500.0
            # Apply small bonus for being in open space
            reward += normalized_clearance * self.OPEN_SPACE_BONUS
        
        return reward

def line_segments_intersect(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return False
    
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
    
    return 0 <= t <= 1 and 0 <= u <= 1

def line_intersects_rect(p1, p2, rect):
    if rect.collidepoint(p1) or rect.collidepoint(p2):
        return True
    rect_lines = [
        ((rect.left, rect.top), (rect.right, rect.top)),
        ((rect.right, rect.top), (rect.right, rect.bottom)),
        ((rect.right, rect.bottom), (rect.left, rect.bottom)),
        ((rect.left, rect.bottom), (rect.left, rect.top))
    ]
    for line_start, line_end in rect_lines:
        if line_segments_intersect(p1, p2, line_start, line_end):
            return True
    return False

def segment_intersection_t(p1, p2, q1, q2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return t
    return None

def first_intersection_t_with_rect(p1, p2, rect):
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

def get_radial_clearance(agent_rect, solids, hazards=None, max_distance=500, num_rays=18):
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

def move(p, keys, left, right, jump, solids):
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
        s = list(self.geometry['base_solids'])
        if bridge_up: s.append(self.geometry['bridge'])
        if not gate_open: s.append(self.geometry['gate'])
        return s

class GameState:
    def __init__(self):
        self.level = Level()
        self.reward_calculator = RewardCalculator()
        self.reset()
        
    def reset(self):
        self.fire = {"r": pg.Rect(*self.level.fire_start, 28, 36), "v": [0.0, 0.0], "g": False, "c": RED}
        self.water = {"r": pg.Rect(*self.level.water_start, 28, 36), "v": [0.0, 0.0], "g": False, "c": BLUE}
        self.bridge_up = False
        self.gate_open = False
    
    def get_reward_state_for_agent(self, agent_name):
        player = self.fire if agent_name == 'fire' else self.water
        exit_rect = self.level.geometry['exitF'] if agent_name == 'fire' else self.level.geometry['exitW']
        
        # Get assigned plate for this agent
        plate_key = 'plateB' if agent_name == 'fire' else 'plateA'
        plate_rect = self.level.geometry[plate_key]
        
        # Calculate total distance to goal
        dx = player["r"].centerx - exit_rect.centerx
        dy = player["r"].centery - exit_rect.centery
        total_dist = np.sqrt(dx**2 + dy**2)
        
        # Calculate horizontal distance to goal (for directional movement reward)
        horizontal_dist = abs(dx)
        
        # Calculate distance to assigned pressure plate
        plate_dx = player["r"].centerx - plate_rect.centerx
        plate_dy = player["r"].centery - plate_rect.centery
        plate_dist = np.sqrt(plate_dx**2 + plate_dy**2)
        
        return {
            'dist': total_dist,
            'horizontal_dist': horizontal_dist,
            'plate_dist': plate_dist,
            'plates_active': int(self.bridge_up) + int(self.gate_open),
            'pos': player["r"].center
        }

# ---- 8. Rendering Function ----
def draw_game(state, fire_agent, water_agent, episode, total_episodes, metrics):
    if S is None: return
    
    G.fill(BLACK)
    geo = state.level.geometry
    
    for s in geo['base_solids']: pg.draw.rect(G, GRAY, s)
    if 'black_barrier' in geo: pg.draw.rect(G, BLACK, geo['black_barrier'])
    if state.bridge_up: pg.draw.rect(G, GREEN, geo['bridge'])
    if not state.gate_open: pg.draw.rect(G, GREEN, geo['gate'])
    
    plate_a_color = (100, 150, 255) if state.bridge_up else (50, 75, 150)
    plate_b_color = (255, 100, 100) if state.gate_open else (150, 50, 50)
    pg.draw.rect(G, plate_a_color, geo['plateA'])
    pg.draw.rect(G, plate_b_color, geo['plateB'])
    
    pg.draw.rect(G, BLUE, geo['water_pool'])
    pg.draw.rect(G, (255, 100, 100), geo['exitF'])
    pg.draw.rect(G, (100, 100, 255), geo['exitW'])
    
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
    
    pg.draw.rect(G, state.fire["c"], state.fire["r"])
    pg.draw.rect(G, state.water["c"], state.water["r"])
    
    S.fill((30, 30, 35))
    scaled_game = pg.transform.scale(G, (int(W*SCALE), int(H*SCALE)))
    S.blit(scaled_game, (0, 0))
    
    # Enhanced metrics panel
    panel_x = int(W*SCALE) + 12
    y = 12
    
    S.blit(FONT_MAIN.render("Cooperative RL Training", True, YELLOW), (panel_x, y)); y += 30
    S.blit(FONT_SMALL.render(f"Episode: {episode}/{total_episodes}", True, WHITE), (panel_x, y)); y += 20
    S.blit(FONT_SMALL.render(f"Success Rate: {metrics.get('success_rate', 0):.1%}", True, WHITE), (panel_x, y)); y += 20
    S.blit(FONT_SMALL.render(f"Cooperation Events: {metrics.get('coop_events', 0)}", True, GREEN), (panel_x, y)); y += 25
    
    # Training losses
    if 'total_loss' in metrics:
        S.blit(FONT_SMALL.render(f"Total Loss: {metrics['total_loss']:.3f}", True, WHITE), (panel_x, y)); y += 18
        S.blit(FONT_SMALL.render(f"Fire Loss: {metrics['fire_loss']:.3f}", True, RED), (panel_x, y)); y += 18
        S.blit(FONT_SMALL.render(f"Water Loss: {metrics['water_loss']:.3f}", True, BLUE), (panel_x, y)); y += 18
        S.blit(FONT_SMALL.render(f"Coop Penalty: {metrics['coop_penalty']:.3f}", True, YELLOW), (panel_x, y)); y += 25
    
    # Fire agent data
    fire_dist = ((state.fire["r"].centerx - geo['exitF'].centerx)**2 + 
                 (state.fire["r"].centery - geo['exitF'].centery)**2)**0.5
    S.blit(FONT_SMALL.render(f"Fire Agent", True, RED), (panel_x, y)); y += 18
    S.blit(FONT_SMALL.render(f"Exit Dist: {fire_dist:.1f}px", True, RED), (panel_x, y)); y += 18
    S.blit(FONT_SMALL.render(f"Epsilon: {fire_agent.epsilon:.3f}", True, RED), (panel_x, y)); y += 18
    S.blit(FONT_SMALL.render(f"Plate Activations: {fire_agent.plate_activations}", True, RED), (panel_x, y)); y += 25
    
    # Water agent data
    water_dist = ((state.water["r"].centerx - geo['exitW'].centerx)**2 + 
                  (state.water["r"].centery - geo['exitW'].centery)**2)**0.5
    S.blit(FONT_SMALL.render(f"Water Agent", True, BLUE), (panel_x, y)); y += 18
    S.blit(FONT_SMALL.render(f"Exit Dist: {water_dist:.1f}px", True, BLUE), (panel_x, y)); y += 18
    S.blit(FONT_SMALL.render(f"Epsilon: {water_agent.epsilon:.3f}", True, BLUE), (panel_x, y)); y += 18
    S.blit(FONT_SMALL.render(f"Plate Activations: {water_agent.plate_activations}", True, BLUE), (panel_x, y)); y += 25
    
    S.blit(FONT_SMALL.render("Press H: Human Play", True, GREEN), (panel_x, y)); y += 20
    S.blit(FONT_SMALL.render("Press R: Reset Episode", True, YELLOW), (panel_x, y))
    
    pg.display.flip()

# ---- 9. Human Play Mode ----
def human_play_mode():
    init_display(True)
    game = GameState()
    
    print("\n" + "=" * 50)
    print("HUMAN PLAY MODE")
    print("=" * 50)
    print("Controls:")
    print("  Fire (Red):   A/D = move, W = jump")
    print("  Water (Blue): Arrow Keys = move/jump")
    print("  Press 'R' to reset level")
    print("  Press 'T' to switch to RL Training Mode")
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
                elif event.key == pg.K_t:
                    print("\nSwitching to RL Training Mode...")
                    pg.quit()
                    return "train"
                elif event.key == pg.K_r:
                    print("\n[R pressed] Resetting level...")
                    game.reset()
        
        keys = pg.key.get_pressed()
        fire_keys = {pg.K_a: keys[pg.K_a], pg.K_d: keys[pg.K_d], pg.K_w: keys[pg.K_w]}
        water_keys = {pg.K_LEFT: keys[pg.K_LEFT], pg.K_RIGHT: keys[pg.K_RIGHT], pg.K_UP: keys[pg.K_UP]}
        
        solids = game.level.get_solids(game.bridge_up, game.gate_open)
        move(game.fire, fire_keys, pg.K_a, pg.K_d, pg.K_w, solids)
        move(game.water, water_keys, pg.K_LEFT, pg.K_RIGHT, pg.K_UP, solids)
        
        geo = game.level.geometry
        game.bridge_up = game.water["r"].colliderect(geo['plateA'])
        game.gate_open = game.fire["r"].colliderect(geo['plateB'])
        
        fire_died = game.fire["r"].colliderect(geo['water_pool'])
        fire_won = geo['exitF'].collidepoint(game.fire["r"].center)
        water_won = geo['exitW'].collidepoint(game.water["r"].center)
        
        if fire_died:
            print("Fire died in water! Resetting...")
            game.reset()
        
        if fire_won and water_won:
            print("\nYOU WIN! Both agents reached their exits!\n")
            game.reset()
        
        # Simple rendering for human play
        G.fill(BLACK)
        for s in geo['base_solids']: pg.draw.rect(G, GRAY, s)
        if 'black_barrier' in geo: pg.draw.rect(G, BLACK, geo['black_barrier'])
        if game.bridge_up: pg.draw.rect(G, GREEN, geo['bridge'])
        if not game.gate_open: pg.draw.rect(G, GREEN, geo['gate'])
        
        plate_a_color = (100, 150, 255) if game.bridge_up else (50, 75, 150)
        plate_b_color = (255, 100, 100) if game.gate_open else (150, 50, 50)
        pg.draw.rect(G, plate_a_color, geo['plateA'])
        pg.draw.rect(G, plate_b_color, geo['plateB'])
        pg.draw.rect(G, BLUE, geo['water_pool'])
        pg.draw.rect(G, (255, 100, 100), geo['exitF'])
        pg.draw.rect(G, (100, 100, 255), geo['exitW'])
        
        pg.draw.rect(G, game.fire["c"], game.fire["r"])
        pg.draw.rect(G, game.water["c"], game.water["r"])
        
        S.fill((30, 30, 35))
        scaled_game = pg.transform.scale(G, (int(W*SCALE), int(H*SCALE)))
        S.blit(scaled_game, (0, 0))
        
        panel_x = int(W*SCALE) + 12
        y = 12
        S.blit(FONT_MAIN.render("HUMAN PLAY MODE", True, YELLOW), (panel_x, y)); y += 30
        S.blit(FONT_SMALL.render("Fire: A/D/W", True, RED), (panel_x, y)); y += 20
        S.blit(FONT_SMALL.render("Water: Arrows", True, BLUE), (panel_x, y)); y += 25
        S.blit(FONT_SMALL.render("Press T: RL Training", True, WHITE), (panel_x, y)); y += 20
        S.blit(FONT_SMALL.render("Press R: Reset", True, YELLOW), (panel_x, y)); y += 20
        S.blit(FONT_SMALL.render("Press ESC: Quit", True, WHITE), (panel_x, y))
        
        pg.display.flip()
        clock.tick(60)
    
    pg.quit()
    return "quit"

# ---- 10. Main Training Loop with Cooperative System ----
def train_cooperative_agents(num_episodes=5000, render=True, use_wandb=True):
    """Main cooperative training loop with synchronized learning"""
    if use_wandb:
        wandb.init(project="firewater-cooperative-rl", config={
            "episodes": num_episodes, 
            "learning_rate": 3e-4,
            "architecture": "shared_backbone_hybrid"
        })
    
    init_display(render)
    
    # Create shared backbone with correct state dimension
    shared_backbone = SharedBackboneNetwork(state_dim=52, hidden_dim=256)  # Updated to 52
    
    # Create cooperative agents with shared backbone
    fire_agent = CooperativeRLAgent("fire", "water", shared_backbone, state_dim=52)  # Updated to 52
    water_agent = CooperativeRLAgent("water", "fire", shared_backbone, state_dim=52)  # Updated to 52
    
    # Create cooperative trainer
    trainer = CooperativeTrainer(fire_agent, water_agent)
    
    # Try to load existing models (will skip if dimensions don't match)
    print("\nChecking for existing models...")
    loaded = trainer.load_models("models/cooperative_agent")
    if not loaded:
        print("Starting fresh training with new 52-dimensional state architecture")
        print("Old 32-dimensional models (if any) will be overwritten when saving")
    
    game = GameState()
    
    episode_rewards = deque(maxlen=100)
    success_tracker = deque(maxlen=100)
    fire_success_tracker = deque(maxlen=100)
    water_success_tracker = deque(maxlen=100)
    cooperation_events_tracker = deque(maxlen=100)
    best_avg_reward = -float('inf')
    
    print("Starting cooperative training with hybrid parameter sharing...")
    for episode in range(num_episodes):
        game.reset()
        episode_reward_fire, episode_reward_water = 0, 0
        cooperation_events = 0
        plate_a_activations = 0
        plate_b_activations = 0
        fire_clearances = []  # Track latest clearances for logging
        water_clearances = []  # Track latest clearances for logging
        
        for step in range(1500):
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
            
            solids = game.level.get_solids(game.bridge_up, game.gate_open)
            switches = {
                'plateA': game.bridge_up, 
                'plateB': game.gate_open, 
                'bridge_up': game.bridge_up, 
                'gate_open': game.gate_open
            }
            
            # Collect hazards for safety system
            geo = game.level.geometry
            hazards = {
                'water_pool': geo.get('water_pool'),
                'lava_pool': geo.get('lava_pool')
            }
            
            # Get enhanced state vectors with partner awareness and safety
            fire_state_vec = fire_agent.get_enhanced_state_vector(
                game.fire, game.water, game.level, solids, switches)
            water_state_vec = water_agent.get_enhanced_state_vector(
                game.water, game.fire, game.level, solids, switches)
            
            # Get partner information
            fire_partner_info = fire_agent.get_partner_info(
                game.water, game.level, solids, switches)
            water_partner_info = water_agent.get_partner_info(
                game.fire, game.level, solids, switches)
            
            # Prepare state data for safety-aware action selection
            fire_state_data = {
                'player': game.fire,
                'solids': solids,
                'hazards': hazards
            }
            water_state_data = {
                'player': game.water,
                'solids': solids,
                'hazards': hazards
            }
            
            # Select actions with safety masking
            fire_action_idx = fire_agent.select_safe_action(
                fire_state_vec, fire_partner_info, fire_state_data)
            water_action_idx = water_agent.select_safe_action(
                water_state_vec, water_partner_info, water_state_data)
            
            # Check if selected actions are safe (for reward calculation)
            fire_safety_mask = fire_agent.get_safe_action_mask(game.fire, solids, hazards)
            water_safety_mask = water_agent.get_safe_action_mask(game.water, solids, hazards)
            fire_action_is_unsafe = (fire_safety_mask[fire_action_idx] == 0.0)
            water_action_is_unsafe = (water_safety_mask[water_action_idx] == 0.0)
            
            fire_keys_list = fire_agent.action_to_keys(fire_action_idx)
            water_keys_list = water_agent.action_to_keys(water_action_idx)
            
            fire_keys = {pg.K_a: 'left' in fire_keys_list, pg.K_d: 'right' in fire_keys_list, pg.K_w: 'jump' in fire_keys_list}
            water_keys = {pg.K_LEFT: 'left' in water_keys_list, pg.K_RIGHT: 'right' in water_keys_list, pg.K_UP: 'jump' in water_keys_list}
            
            old_reward_state_fire = game.get_reward_state_for_agent('fire')
            old_reward_state_water = game.get_reward_state_for_agent('water')
            
            # Ensure hazard_proximity exists in old states (for backward compatibility)
            if 'hazard_proximity' not in old_reward_state_fire:
                old_reward_state_fire['hazard_proximity'] = 1.0
            if 'hazard_proximity' not in old_reward_state_water:
                old_reward_state_water['hazard_proximity'] = 1.0
            
            # Track pre-action state for progress detection
            old_fire_pos = game.fire["r"].center
            old_water_pos = game.water["r"].center
            
            # Track pre-action plate states
            old_bridge_up = game.bridge_up
            old_gate_open = game.gate_open
            old_both_plates_active = old_bridge_up and old_gate_open
            
            # Track if agents were on ground before action
            fire_was_grounded = game.fire["g"]
            water_was_grounded = game.water["g"]
            
            # Execute actions
            move(game.fire, fire_keys, pg.K_a, pg.K_d, pg.K_w, solids)
            move(game.water, water_keys, pg.K_LEFT, pg.K_RIGHT, pg.K_UP, solids)
            
            geo = game.level.geometry
            game.bridge_up = game.water["r"].colliderect(geo['plateA'])
            game.gate_open = game.fire["r"].colliderect(geo['plateB'])
            
            # NEW: Check if both plates are now active (collaborative task complete)
            both_plates_active = game.bridge_up and game.gate_open
            
            # Check if agents are on their plates
            fire_on_plate = game.fire["r"].colliderect(geo['plateB'])
            water_on_plate = game.water["r"].colliderect(geo['plateA'])
            
            # Check if agents are in goal area
            fire_in_goal = geo['exitF'].colliderect(game.fire["r"])
            water_in_goal = geo['exitW'].colliderect(game.water["r"])
            
            # Detect if agent jumped (was grounded and pressed jump)
            fire_jumped = fire_was_grounded and 'jump' in fire_keys_list
            water_jumped = water_was_grounded and 'jump' in water_keys_list
            
            # Detect if agents made progress (moved)
            fire_made_progress = (game.fire["r"].center != old_fire_pos)
            water_made_progress = (game.water["r"].center != old_water_pos)
            
            # Get 18-directional clearances for open-space bonus
            fire_clearances = get_radial_clearance(game.fire["r"], solids, list(hazards.values()), num_rays=18)
            water_clearances = get_radial_clearance(game.water["r"], solids, list(hazards.values()), num_rays=18)
            
            # Track cooperation events
            if not old_bridge_up and game.bridge_up:
                cooperation_events += 1
                plate_a_activations += 1
                water_agent.plate_activations += 1
            
            if not old_gate_open and game.gate_open:
                cooperation_events += 1
                plate_b_activations += 1
                fire_agent.plate_activations += 1
            
            fire_died = game.fire["r"].colliderect(geo['water_pool'])
            water_died = False
            fire_success = geo['exitF'].collidepoint(game.fire["r"].center)
            water_success = geo['exitW'].collidepoint(game.water["r"].center)
            success = fire_success and water_success
            done = fire_died or water_died or success
            
            new_reward_state_fire = game.get_reward_state_for_agent('fire')
            new_reward_state_water = game.get_reward_state_for_agent('water')
            
            # Ensure hazard_proximity exists in new states (for backward compatibility)
            if 'hazard_proximity' not in new_reward_state_fire:
                new_reward_state_fire['hazard_proximity'] = 1.0
            if 'hazard_proximity' not in new_reward_state_water:
                new_reward_state_water['hazard_proximity'] = 1.0
            
            # Calculate rewards with collaboration-focused context and safety
            # FIXED: Remove collaboration incentives - agents focus on their own plates only
            fire_helped_partner = False  # Fire doesn't get bonus for helping water
            water_helped_partner = False  # Water doesn't get bonus for helping fire
            
            reward_fire = game.reward_calculator.calculate_reward(
                old_reward_state_fire, new_reward_state_fire, 
                {
                    'died': fire_died, 
                    'success': success, 
                    'helped_partner': fire_helped_partner,
                    'on_ground': game.fire["g"],
                    'jumped': fire_jumped,
                    'on_plate': fire_on_plate,
                    'in_goal': fire_in_goal,
                    'both_plates_active': both_plates_active,
                    'both_plates_were_active': old_both_plates_active,
                    'hazard_proximity': new_reward_state_fire['hazard_proximity'],
                    'unsafe_action': fire_action_is_unsafe,
                    'made_progress': fire_made_progress,
                    'clearances': fire_clearances  # NEW: Pass clearance data for open-space bonus
                },
                'fire'  # Agent name for agent-specific rewards
            )
            reward_water = game.reward_calculator.calculate_reward(
                old_reward_state_water, new_reward_state_water, 
                {
                    'died': water_died, 
                    'success': success, 
                    'helped_partner': water_helped_partner,
                    'on_ground': game.water["g"],
                    'jumped': water_jumped,
                    'on_plate': water_on_plate,
                    'in_goal': water_in_goal,
                    'both_plates_active': both_plates_active,
                    'both_plates_were_active': old_both_plates_active,
                    'hazard_proximity': new_reward_state_water['hazard_proximity'],
                    'unsafe_action': water_action_is_unsafe,
                    'made_progress': water_made_progress,
                    'clearances': water_clearances  # NEW: Pass clearance data for open-space bonus
                },
                'water'  # Agent name for agent-specific rewards
            )
            
            episode_reward_fire += reward_fire
            episode_reward_water += reward_water
            
            # Get next states
            solids_next = game.level.get_solids(game.bridge_up, game.gate_open)
            switches_next = {
                'plateA': game.bridge_up, 
                'plateB': game.gate_open, 
                'bridge_up': game.bridge_up, 
                'gate_open': game.gate_open
            }
            
            fire_state_next = fire_agent.get_enhanced_state_vector(
                game.fire, game.water, game.level, solids_next, switches_next)
            water_state_next = water_agent.get_enhanced_state_vector(
                game.water, game.fire, game.level, solids_next, switches_next)
            
            fire_partner_info_next = fire_agent.get_partner_info(
                game.water, game.level, solids_next, switches_next)
            water_partner_info_next = water_agent.get_partner_info(
                game.fire, game.level, solids_next, switches_next)
            
            # Store experiences
            fire_agent.replay_buffer.push(
                fire_state_vec, fire_partner_info, fire_action_idx, 
                reward_fire, fire_state_next, fire_partner_info_next, done)
            water_agent.replay_buffer.push(
                water_state_vec, water_partner_info, water_action_idx, 
                reward_water, water_state_next, water_partner_info_next, done)
            
            # Cooperative training step with enhanced collaboration metrics
            cooperation_metrics = {
                'bridge_needed': not game.bridge_up and fire_died,
                'gate_needed': not game.gate_open,
                'plate_a_activated': game.bridge_up,
                'plate_b_activated': game.gate_open,
                'both_plates_active': both_plates_active,  # NEW: Critical collaboration metric
                'both_failed': fire_died or (not fire_success and not water_success),
                'success_disparity': abs(int(fire_success) - int(water_success))
            }
            
            loss_dict = trainer.synchronized_training_step(
                batch_size=64, cooperation_metrics=cooperation_metrics)
            
            if render and loss_dict:
                metrics = {
                    'success_rate': np.mean(success_tracker) if success_tracker else 0,
                    'coop_events': cooperation_events,
                    **loss_dict
                }
                draw_game(game, fire_agent, water_agent, episode, num_episodes, metrics)
                C.tick(60)
            
            if done:
                break
        
        fire_agent.decay_epsilon()
        water_agent.decay_epsilon()
        
        total_reward = episode_reward_fire + episode_reward_water
        episode_rewards.append(total_reward)
        success_tracker.append(1 if success else 0)
        fire_success_tracker.append(1 if fire_success else 0)
        water_success_tracker.append(1 if water_success else 0)
        cooperation_events_tracker.append(cooperation_events)
        
        avg_reward = np.mean(episode_rewards)
        success_rate = np.mean(success_tracker)
        fire_success_rate = np.mean(fire_success_tracker)
        water_success_rate = np.mean(water_success_tracker)
        avg_coop_events = np.mean(cooperation_events_tracker)
        
        if use_wandb and loss_dict:
            # Calculate open-space scores for logging
            fire_open_space = sum([d for _, d, _ in fire_clearances]) / (500.0 * 18) if fire_clearances else 0
            water_open_space = sum([d for _, d, _ in water_clearances]) / (500.0 * 18) if water_clearances else 0
            
            wandb.log({
                "episode": episode,
                "total_reward": total_reward,
                "avg_reward_100": avg_reward,
                "success_rate_100": success_rate,
                "fire_success_rate_100": fire_success_rate,
                "water_success_rate_100": water_success_rate,
                "cooperation_events": cooperation_events,
                "avg_cooperation_events": avg_coop_events,
                "plate_a_activations": plate_a_activations,
                "plate_b_activations": plate_b_activations,
                "steps": step + 1,
                "fire_epsilon": fire_agent.epsilon,
                "total_loss": loss_dict['total_loss'],
                "fire_loss": loss_dict['fire_loss'],
                "water_loss": loss_dict['water_loss'],
                "cooperation_penalty": loss_dict['coop_penalty'],
                "fire_open_space_score": fire_open_space,  # NEW: Track open-space utilization
                "water_open_space_score": water_open_space  # NEW: Track open-space utilization
            })
        
        if episode % 10 == 0:
            safety_rate_fire = fire_agent.safe_actions_taken / max(fire_agent.safe_actions_taken + fire_agent.safety_violations, 1)
            safety_rate_water = water_agent.safe_actions_taken / max(water_agent.safe_actions_taken + water_agent.safety_violations, 1)
            print(f"Ep {episode} | Avg Reward: {avg_reward:.2f} | "
                  f"Success: {success_rate:.1%} (F:{fire_success_rate:.1%} W:{water_success_rate:.1%}) | "
                  f"Coop Events: {avg_coop_events:.1f} | "
                  f"Safety: F:{safety_rate_fire:.1%} W:{safety_rate_water:.1%} | "
                  f"ε: {fire_agent.epsilon:.3f}")
        
        if len(episode_rewards) == 100 and avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            trainer.save_models("models/cooperative_agent")
            print(f"New best average reward of {avg_reward:.2f}. Models saved.")
    
    if use_wandb:
        wandb.finish()
    return "quit"

# ---- 11. Main Execution Block ----
if __name__ == "__main__":
    print("=" * 50)
    print("Fire & Water - Cooperative RL Training")
    print("Hybrid Parameter Sharing with Mutual Penalties")
    print("=" * 50)
    print("\nStarting in HUMAN PLAY MODE")
    print("Press 'T' during play to switch to RL Training")
    print("Press 'H' during training to switch to Human Play")
    print("=" * 50)
    
    USE_RENDERING = True
    mode = "human"
    
    try:
        while True:
            if mode == "human":
                result = human_play_mode()
                if result == "train":
                    mode = "train"
                elif result == "quit":
                    break
            elif mode == "train":
                result = train_cooperative_agents(
                    num_episodes=5000, render=USE_RENDERING, use_wandb=True)
                if result == "human":
                    mode = "human"
                elif result == "quit":
                    break
            else:
                break
    except KeyboardInterrupt:
        print("\nSession interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down gracefully...")
        if wandb.run:
            wandb.finish()
        if USE_RENDERING and pg.get_init():
            pg.quit()
        print("Session ended.")
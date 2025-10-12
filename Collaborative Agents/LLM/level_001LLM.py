import pygame as pg
import json
import numpy as np
from datetime import datetime
from collections import deque
import anthropic
import os

pg.init()

# ---- Display settings ----
W, H = 960, 540
SCALE = 0.8
PANEL_W = 300
SW, SH = int(W*SCALE) + PANEL_W, int(H*SCALE)

S = pg.display.set_mode((SW, SH))
G = pg.Surface((W, H))
C = pg.time.Clock()
F = pg.font.SysFont(None, 22)
F_SMALL = pg.font.SysFont(None, 18)

RED, BLUE, GREEN, GRAY, WHITE, BLACK = (220,60,60), (60,120,255), (60,200,120), (170,170,170), (240,240,240), (15,15,20)
YELLOW, PURPLE, ORANGE = (255,220,60), (180,60,200), (255,140,0)

# ---- LLM Agent System ----
class LLMAgent:
    """Agent that uses Claude API to play the game"""
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
        self.decision_history = []
        self.reward_history = []
        self.enabled = False
        self.last_decision_time = 0
        self.decision_interval = 500  # ms between decisions
        self.current_action = {'fire': set(), 'water': set()}
        self.episode_data = []
        
    def get_game_state_summary(self, state, level):
        """Create a concise state representation for the LLM"""
        geo = level.geometry
        
        fire_pos = state.fire["r"].center
        water_pos = state.water["r"].center
        fire_exit = geo['exitF'].center
        water_exit = geo['exitW'].center
        
        fire_dist = np.sqrt((fire_pos[0] - fire_exit[0])**2 + (fire_pos[1] - fire_exit[1])**2)
        water_dist = np.sqrt((water_pos[0] - water_exit[0])**2 + (water_pos[1] - water_exit[1])**2)
        
        solids = level.get_solids(state.bridge_up, state.gate_open)
        fire_walls = count_walls_between(state.fire["r"], geo['exitF'], solids)
        water_walls = count_walls_between(state.water["r"], geo['exitW'], solids)
        
        # Check plate states
        plateA_on = False
        plateB_on = False
        if 'plateA' in geo:
            plateA_on = (state.fire["r"].colliderect(geo['plateA']) or 
                        state.water["r"].colliderect(geo['plateA']))
        if 'plateB' in geo:
            plateB_on = (state.fire["r"].colliderect(geo['plateB']) or 
                        state.water["r"].colliderect(geo['plateB']))
        
        return {
            'level': level.name,
            'fire': {
                'pos': fire_pos,
                'dist_to_exit': round(fire_dist, 1),
                'walls_blocking': fire_walls,
                'on_ground': state.fire["g"]
            },
            'water': {
                'pos': water_pos,
                'dist_to_exit': round(water_dist, 1),
                'walls_blocking': water_walls,
                'on_ground': state.water["g"]
            },
            'switches': {
                'plateA': plateA_on,
                'plateB': plateB_on,
                'bridge_up': state.bridge_up,
                'gate_open': state.gate_open
            },
            'combined_distance': round(fire_dist + water_dist, 1)
        }
    
    def decide_action(self, state, level, use_simple=True):
        """Decide what actions to take"""
        if use_simple:
            return self._simple_heuristic(state, level)
        else:
            return self._llm_decision(state, level)
    
    def _simple_heuristic(self, state, level):
        """Simple rule-based agent for testing"""
        actions = {'fire': set(), 'water': set()}
        geo = level.geometry
        
        # Fire agent: move towards exit
        fire_x = state.fire["r"].centerx
        fire_exit_x = geo['exitF'].centerx
        
        if fire_x < fire_exit_x - 10:
            actions['fire'].add('right')
        elif fire_x > fire_exit_x + 10:
            actions['fire'].add('left')
        
        # Jump if blocked
        solids = level.get_solids(state.bridge_up, state.gate_open)
        fire_walls = count_walls_between(state.fire["r"], geo['exitF'], solids)
        if fire_walls > 0 and state.fire["g"]:
            actions['fire'].add('jump')
        
        # Water agent: move towards exit
        water_x = state.water["r"].centerx
        water_exit_x = geo['exitW'].centerx
        
        if water_x < water_exit_x - 10:
            actions['water'].add('right')
        elif water_x > water_exit_x + 10:
            actions['water'].add('left')
        
        # Jump if blocked
        water_walls = count_walls_between(state.water["r"], geo['exitW'], solids)
        if water_walls > 0 and state.water["g"]:
            actions['water'].add('jump')
        
        return actions
    
    def _llm_decision(self, state, level):
        """Use Claude API to make decisions"""
        if not self.client:
            print("No API key - using heuristic")
            return self._simple_heuristic(state, level)
        
        game_state = self.get_game_state_summary(state, level)
        
        prompt = f"""You are playing a cooperative puzzle platformer game. You control both Fire (red) and Water (blue) characters.

Current State:
- Level: {game_state['level']}
- Fire: pos={game_state['fire']['pos']}, distance to exit={game_state['fire']['dist_to_exit']}px, walls blocking={game_state['fire']['walls_blocking']}, on_ground={game_state['fire']['on_ground']}
- Water: pos={game_state['water']['pos']}, distance to exit={game_state['water']['dist_to_exit']}px, walls blocking={game_state['water']['walls_blocking']}, on_ground={game_state['water']['on_ground']}
- Switches: Plate A={game_state['switches']['plateA']}, Plate B={game_state['switches']['plateB']}, Bridge={game_state['switches']['bridge_up']}, Gate={game_state['switches']['gate_open']}

Controls:
- Fire: left, right, jump
- Water: left, right, jump

Respond with ONLY a JSON object with your actions:
{{"fire": ["action1", "action2"], "water": ["action1"]}}

Actions can be: "left", "right", "jump"
Keep response minimal - just the JSON."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text.strip()
            # Extract JSON from response
            if '{' in response_text:
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                response_text = response_text[json_start:json_end]
            
            decision = json.loads(response_text)
            
            actions = {
                'fire': set(decision.get('fire', [])),
                'water': set(decision.get('water', []))
            }
            
            self.decision_history.append({
                'state': game_state,
                'action': actions,
                'timestamp': pg.time.get_ticks()
            })
            
            return actions
            
        except Exception as e:
            print(f"LLM decision error: {e}")
            return self._simple_heuristic(state, level)
    
    def calculate_reward(self, old_state, new_state, level):
        """Calculate reward for RL training"""
        reward = 0.0
        
        # Distance improvement reward
        old_dist = old_state.get('combined_distance', 1000)
        new_dist = new_state.get('combined_distance', 1000)
        reward += (old_dist - new_dist) * 0.1  # Reward for getting closer
        
        # Wall clearance reward
        old_walls = old_state.get('fire', {}).get('walls_blocking', 0) + old_state.get('water', {}).get('walls_blocking', 0)
        new_walls = new_state.get('fire', {}).get('walls_blocking', 0) + new_state.get('water', {}).get('walls_blocking', 0)
        if new_walls < old_walls:
            reward += 5.0  # Bonus for clearing obstacles
        
        # Switch activation reward
        if new_state['switches']['bridge_up'] and not old_state['switches']['bridge_up']:
            reward += 10.0
        if new_state['switches']['gate_open'] and not old_state['switches']['gate_open']:
            reward += 10.0
        
        return reward
    
    def record_episode_step(self, state_summary, action, reward):
        """Record step for training"""
        self.episode_data.append({
            'state': state_summary,
            'action': action,
            'reward': reward,
            'timestamp': datetime.now().isoformat()
        })
        self.reward_history.append(reward)
    
    def export_training_data(self, filename_prefix="training_data"):
        """Export collected training data"""
        filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        data = {
            'episodes': self.episode_data,
            'total_reward': sum(self.reward_history),
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0,
            'decisions_made': len(self.decision_history)
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Training data exported to {filename}")
        return filename

# ---- Data Logging System ----
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
        
    def log_keystroke(self, agent, keys_pressed, is_ai=False):
        """Log which keys are pressed for each agent"""
        timestamp = pg.time.get_ticks() - self.start_time
        self.keystrokes.append({
            'time_ms': timestamp,
            'agent': agent,
            'keys': list(keys_pressed),
            'is_ai': is_ai
        })
    
    def log_positions(self, fire_pos, water_pos, fire_exit, water_exit, fire_rect, water_rect, solids):
        """Log current positions and distances to goals"""
        timestamp = pg.time.get_ticks() - self.start_time
        
        fire_dist = ((fire_pos[0] - fire_exit.centerx)**2 + (fire_pos[1] - fire_exit.centery)**2)**0.5
        water_dist = ((water_pos[0] - water_exit.centerx)**2 + (water_pos[1] - water_exit.centery)**2)**0.5
        
        fire_walls = count_walls_between(fire_rect, fire_exit, solids)
        water_walls = count_walls_between(water_rect, water_exit, solids)
        
        self.position_history.append({
            'time_ms': timestamp,
            'fire': {
                'x': fire_pos[0], 
                'y': fire_pos[1], 
                'dist_to_exit': fire_dist,
                'walls_blocking': fire_walls,
                'path_clear': fire_walls == 0
            },
            'water': {
                'x': water_pos[0], 
                'y': water_pos[1], 
                'dist_to_exit': water_dist,
                'walls_blocking': water_walls,
                'path_clear': water_walls == 0
            },
            'combined_dist': fire_dist + water_dist,
            'total_walls_blocking': fire_walls + water_walls
        })
    
    def log_switch_event(self, switch_name, activated_by, state):
        """Log when switches/plates are activated or deactivated"""
        timestamp = pg.time.get_ticks() - self.start_time
        self.switch_events.append({
            'time_ms': timestamp,
            'switch': switch_name,
            'agent': activated_by,
            'state': state
        })
    
    def export_session(self, level_name, success, retries, agent_mode="human"):
        """Export complete session data to JSON"""
        total_time = (pg.time.get_ticks() - self.start_time) / 1000
        data = {
            'timestamp': datetime.now().isoformat(),
            'level': level_name,
            'success': success,
            'time_seconds': total_time,
            'retries': retries,
            'agent_mode': agent_mode,
            'keystrokes': self.keystrokes,
            'position_history': self.position_history[-100:],
            'switch_events': self.switch_events,
            'summary': {
                'total_keystrokes': len(self.keystrokes),
                'ai_keystrokes': sum(1 for k in self.keystrokes if k.get('is_ai', False)),
                'human_keystrokes': sum(1 for k in self.keystrokes if not k.get('is_ai', False)),
                'final_distance': self.position_history[-1]['combined_dist'] if self.position_history else None
            }
        }
        
        filename = f"game_log_{level_name}_{agent_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
            pg.Rect(0, 500, W, 40),
            pg.Rect(40, 420, 200, 20),
            pg.Rect(400, 420, 120, 20),
            pg.Rect(700, 420, 120, 20),
            pg.Rect(40, 340, 240, 20)
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

LEVEL_2 = Level(
    "Coordination",
    (100, 464),
    (850, 464),
    {
        'base_solids': [
            pg.Rect(0, 500, W, 40),
            pg.Rect(40, 300, 200, 20),
            pg.Rect(720, 300, 200, 20),
            pg.Rect(300, 400, 80, 20),
            pg.Rect(580, 400, 80, 20),
            pg.Rect(420, 350, 120, 20)
        ],
        'bridge': pg.Rect(420, 280, 120, 20),
        'plateA': pg.Rect(320, 384, 40, 16),
        'plateB': pg.Rect(600, 384, 40, 16),
        'water_pool': pg.Rect(670, 420, 90, 80),
        'lava_pool': pg.Rect(200, 500, 90, 40),
        'exitW': pg.Rect(80, 268, 36, 36),
        'exitF': pg.Rect(844, 268, 36, 36)
    }
)

LEVEL_3 = Level(
    "Elevation",
    (100, 464),
    (100, 420),
    {
        'base_solids': [
            pg.Rect(0, 500, W, 40),
            pg.Rect(40, 460, 180, 20),
            pg.Rect(40, 200, 120, 20),
            pg.Rect(200, 300, 100, 20),
            pg.Rect(350, 400, 100, 20),
            pg.Rect(500, 320, 100, 20),
            pg.Rect(650, 240, 120, 20),
            pg.Rect(800, 400, 140, 20)
        ],
        'bridge': pg.Rect(650, 160, 120, 20),
        'gate': pg.Rect(500, 240, 20, 80),
        'plateA': pg.Rect(360, 384, 40, 16),
        'plateB': pg.Rect(820, 384, 40, 16),
        'water_pool': pg.Rect(470, 420, 100, 80),
        'lava_pool': pg.Rect(720, 500, 100, 40),
        'exitW': pg.Rect(680, 128, 36, 36),
        'exitF': pg.Rect(740, 128, 36, 36)
    }
)

LEVELS = [LEVEL_1, LEVEL_2, LEVEL_3]

# ---- Game State ----
class GameState:
    def __init__(self):
        self.current_level_idx = 0
        self.reset_level()
        self.logger = GameLogger()
        self.agent = LLMAgent()
        self.retries = 0
        self.done = False
        self.last_bridge_state = False
        self.last_gate_state = False
        self.agent_mode = "human"  # "human", "agent", "hybrid"
        self.last_state_summary = None
        
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
        if hasattr(self, 'agent'):
            self.agent.episode_data = []
            self.agent.reward_history = []
        
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
    
    def toggle_agent_mode(self):
        """Cycle through agent modes"""
        modes = ["human", "agent", "hybrid"]
        idx = modes.index(self.agent_mode)
        self.agent_mode = modes[(idx + 1) % len(modes)]
        print(f"Agent mode: {self.agent_mode}")

def move(p, keys, left, right, jump, solids, agent_name, logger, is_ai=False):
    VMAX, ACC, GRAV = 3.4, 0.9, 0.5
    
    keys_pressed = []
    if keys[left]: keys_pressed.append('left')
    if keys[right]: keys_pressed.append('right')
    if keys[jump] and p["g"]: keys_pressed.append('jump')
    
    if keys_pressed:
        logger.log_keystroke(agent_name, keys_pressed, is_ai)
    
    p["v"][0] = (-VMAX if keys[left] else VMAX if keys[right] else 0)
    if p["g"] and keys[jump]: p["v"][1] = -13
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
            if p["v"][1] > 0:
                p["r"].bottom = s.top
                p["g"] = True
            else:
                p["r"].top = s.bottom
            p["v"][1] = 0

def count_walls_between(start_rect, end_rect, solids):
    start_center = start_rect.center
    end_center = end_rect.center
    
    wall_count = 0
    for solid in solids:
        if line_intersects_rect(start_center, end_center, solid):
            wall_count += 1
    
    return wall_count

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

def draw(state):
    G.fill(BLACK)
    level = state.get_level()
    geo = level.geometry
    
    if 'water_pool' in geo:
        pg.draw.rect(G, (30, 90, 200), geo['water_pool'])
    if 'lava_pool' in geo:
        pg.draw.rect(G, (200, 60, 30), geo['lava_pool'])
    
    for s in level.get_solids(state.bridge_up, state.gate_open):
        pg.draw.rect(G, GRAY, s)
    
    if 'plateA' in geo:
        pg.draw.rect(G, BLUE, geo['plateA'])
    if 'plateB' in geo:
        pg.draw.rect(G, RED, geo['plateB'])
    
    pg.draw.rect(G, BLUE, geo['exitW'], 2)
    pg.draw.rect(G, RED, geo['exitF'], 2)
    
    # Draw players with indicator if AI-controlled
    fire_color = state.fire["c"] if state.agent_mode == "human" else ORANGE
    water_color = state.water["c"] if state.agent_mode == "human" else PURPLE
    
    pg.draw.rect(G, fire_color, state.fire["r"])
    pg.draw.rect(G, water_color, state.water["r"])
    
    # AI indicator
    if state.agent_mode != "human":
        pg.draw.circle(G, YELLOW, (state.fire["r"].centerx, state.fire["r"].top - 10), 4)
        pg.draw.circle(G, YELLOW, (state.water["r"].centerx, state.water["r"].top - 10), 4)
    
    solids = level.get_solids(state.bridge_up, state.gate_open)
    fire_walls = count_walls_between(state.fire["r"], geo['exitF'], solids)
    water_walls = count_walls_between(state.water["r"], geo['exitW'], solids)
    
    line_color_f = (255, 100, 100, 128) if fire_walls > 0 else (100, 255, 100, 128)
    line_color_w = (100, 100, 255, 128) if water_walls > 0 else (100, 255, 100, 128)
    pg.draw.line(G, line_color_f[:3], state.fire["r"].center, geo['exitF'].center, 1)
    pg.draw.line(G, line_color_w[:3], state.water["r"].center, geo['exitW'].center, 1)
    
    tt = (pg.time.get_ticks() - state.t0) / 1000 if not state.done else state.final_time
    hud = f"Level {state.current_level_idx+1}: {level.name} | Time {tt:.2f}s | Retries {state.retries} | Mode: {state.agent_mode.upper()}"
    G.blit(F.render(hud, True, WHITE), (12, 10))
    G.blit(F.render("A/D/W Fire | ←/→/↑ Water | R reset | N/P level | E export | T toggle AI | X export training", True, WHITE), (12, 34))
    
    if state.done:
        box = pg.Rect(0, 0, 400, 80)
        box.center = (W//2, 80)
        pg.draw.rect(G, (35, 35, 50), box, 0, 8)
        pg.draw.rect(G, GREEN, box, 2, 8)
        mode_text = f" ({state.agent_mode})" if state.agent_mode != "human" else ""
        G.blit(F.render(f"SUCCESS: {state.final_time:.2f}s{mode_text}", True, WHITE), (box.x+14, box.y+24))
    
    S.fill(BLACK)
    scaled = pg.transform.smoothscale(G, (int(W*SCALE), int(H*SCALE)))
    S.blit(scaled, (0, 0))
    
    # Right panel
    panel_x = int(W*SCALE)
    panel = pg.Rect(panel_x, 0, PANEL_W, SH)
    pg.draw.rect(S, (28, 28, 42), panel)
    
    mode_colors = {"human": WHITE, "agent": ORANGE, "hybrid": PURPLE}
    S.blit(F.render(f"{state.agent_mode.upper()} MODE", True, mode_colors[state.agent_mode]), (panel.x+12, 12))
    
    y = 40
    fire_dist = ((state.fire["r"].centerx - geo['exitF'].centerx)**2 + 
                 (state.fire["r"].centery - geo['exitF'].centery)**2)**0.5
    water_dist = ((state.water["r"].centerx - geo['exitW'].centerx)**2 + 
                  (state.water["r"].centery - geo['exitW'].centery)**2)**0.5
    
    S.blit(F_SMALL.render(f"Fire Pos: ({state.fire['r'].x}, {state.fire['r'].y})", True, RED), (panel.x+12, y))
    y += 20
    S.blit(F_SMALL.render(f"Exit Dist: {fire_dist:.1f}px", True, RED), (panel.x+12, y))
    y += 18
    wall_color = RED if fire_walls > 0 else GREEN
    S.blit(F_SMALL.render(f"Walls: {fire_walls} {'(BLOCKED)' if fire_walls > 0 else '(CLEAR)'}", True, wall_color), (panel.x+12, y))
    y += 25
    
    S.blit(F_SMALL.render(f"Water Pos: ({state.water['r'].x}, {state.water['r'].y})", True, BLUE), (panel.x+12, y))
    y += 20
    S.blit(F_SMALL.render(f"Exit Dist: {water_dist:.1f}px", True, BLUE), (panel.x+12, y))
    y += 18
    wall_color = RED if water_walls > 0 else GREEN
    S.blit(F_SMALL.render(f"Walls: {water_walls} {'(BLOCKED)' if water_walls > 0 else '(CLEAR)'}", True, wall_color), (panel.x+12, y))
    y += 25
    
    S.blit(F_SMALL.render("SWITCHES:", True, YELLOW), (panel.x+12, y))
    y += 20
    if 'plateA' in geo:
        plateA_active = (state.fire["r"].colliderect(geo['plateA']) or 
                        state.water["r"].colliderect(geo['plateA']))
        color = GREEN if plateA_active else GRAY
        S.blit(F_SMALL.render(f"Plate A: {'ON' if plateA_active else 'OFF'}", True, color), (panel.x+12, y))
        y += 18
    
    if 'plateB' in geo:
        plateB_active = (state.fire["r"].colliderect(geo['plateB']) or 
                        state.water["r"].colliderect(geo['plateB']))
        color = GREEN if plateB_active else GRAY
        S.blit(F_SMALL.render(f"Plate B: {'ON' if plateB_active else 'OFF'}", True, color), (panel.x+12, y))
        y += 18
    
    y += 10
    S.blit(F_SMALL.render(f"Bridge: {'UP' if state.bridge_up else 'DOWN'}", True, GREEN if state.bridge_up else GRAY), (panel.x+12, y))
    y += 18
    S.blit(F_SMALL.render(f"Gate: {'OPEN' if state.gate_open else 'CLOSED'}", True, GREEN if state.gate_open else GRAY), (panel.x+12, y))
    
    y += 25
    S.blit(F_SMALL.render("STATISTICS:", True, YELLOW), (panel.x+12, y))
    y += 20
    S.blit(F_SMALL.render(f"Keystrokes: {len(state.logger.keystrokes)}", True, WHITE), (panel.x+12, y))
    y += 18
    ai_keys = sum(1 for k in state.logger.keystrokes if k.get('is_ai', False))
    S.blit(F_SMALL.render(f"AI Actions: {ai_keys}", True, ORANGE), (panel.x+12, y))
    y += 18
    S.blit(F_SMALL.render(f"Switch Events: {len(state.logger.switch_events)}", True, WHITE), (panel.x+12, y))
    y += 18
    S.blit(F_SMALL.render(f"Positions Logged: {len(state.logger.position_history)}", True, WHITE), (panel.x+12, y))
    
    # RL Training info
    if state.agent_mode != "human":
        y += 25
        S.blit(F_SMALL.render("RL TRAINING:", True, YELLOW), (panel.x+12, y))
        y += 20
        S.blit(F_SMALL.render(f"Decisions: {len(state.agent.decision_history)}", True, WHITE), (panel.x+12, y))
        y += 18
        S.blit(F_SMALL.render(f"Episode Steps: {len(state.agent.episode_data)}", True, WHITE), (panel.x+12, y))
        y += 18
        if state.agent.reward_history:
            avg_reward = np.mean(state.agent.reward_history[-10:])  # Last 10
            S.blit(F_SMALL.render(f"Avg Reward: {avg_reward:.2f}", True, GREEN if avg_reward > 0 else RED), (panel.x+12, y))
    
    pg.display.flip()

# ---- Main Loop ----
state = GameState()
running = True

print("=" * 60)
print("FIRE AND WATER GAME - AI AGENT EDITION")
print("=" * 60)
print("Controls:")
print("  T - Toggle AI mode (human/agent/hybrid)")
print("  E - Export game session data")
print("  X - Export RL training data")
print("  R - Reset level")
print("  N/P - Next/Previous level")
print()
print("Agent Modes:")
print("  human - You control both characters")
print("  agent - AI controls both (simple heuristic by default)")
print("  hybrid - You control Fire, AI controls Water")
print()
print("To use LLM agent: Set ANTHROPIC_API_KEY environment variable")
print("=" * 60)

while running:
    for e in pg.event.get():
        if e.type == pg.QUIT:
            running = False
        if e.type == pg.KEYDOWN:
            if e.key == pg.K_r:
                state.retries += 1
                state.reset_level()
            if e.key == pg.K_n:
                state.next_level()
            if e.key == pg.K_p:
                state.prev_level()
            if e.key == pg.K_e:
                state.logger.export_session(
                    state.get_level().name,
                    state.done,
                    state.retries,
                    state.agent_mode
                )
            if e.key == pg.K_t:
                state.toggle_agent_mode()
            if e.key == pg.K_x:
                if state.agent_mode != "human":
                    state.agent.export_training_data()
                else:
                    print("Switch to agent mode first (press T)")
    
    if not state.done:
        level = state.get_level()
        geo = level.geometry
        solids = level.get_solids(state.bridge_up, state.gate_open)
        
        # Get current state for RL
        current_state = state.agent.get_game_state_summary(state, level)
        
        # Agent decision making
        agent_actions = {'fire': set(), 'water': set()}
        if state.agent_mode in ["agent", "hybrid"]:
            current_time = pg.time.get_ticks()
            if current_time - state.agent.last_decision_time > state.agent.decision_interval:
                agent_actions = state.agent.decide_action(state, level, use_simple=True)
                state.agent.last_decision_time = current_time
                
                # Calculate reward if we have previous state
                if state.last_state_summary:
                    reward = state.agent.calculate_reward(state.last_state_summary, current_state, level)
                    state.agent.record_episode_step(state.last_state_summary, agent_actions, reward)
                
                state.last_state_summary = current_state
        
        # Check plate activation
        if level.name == "Coordination":
            plateA_on = (state.fire["r"].colliderect(geo['plateA']) or 
                        state.water["r"].colliderect(geo['plateA']))
            plateB_on = (state.fire["r"].colliderect(geo['plateB']) or 
                        state.water["r"].colliderect(geo['plateB']))
            state.bridge_up = plateA_on and plateB_on
            
            if plateA_on != state.last_bridge_state:
                agent = 'fire' if state.fire["r"].colliderect(geo['plateA']) else 'water'
                state.logger.log_switch_event('plateA', agent, plateA_on)
            if plateB_on != state.last_gate_state:
                agent = 'fire' if state.fire["r"].colliderect(geo['plateB']) else 'water'
                state.logger.log_switch_event('plateB', agent, plateB_on)
            state.last_bridge_state = plateA_on
            state.last_gate_state = plateB_on
        else:
            if 'plateA' in geo:
                plateA_on = (state.fire["r"].colliderect(geo['plateA']) or 
                            state.water["r"].colliderect(geo['plateA']))
                if plateA_on != state.bridge_up:
                    agent = 'fire' if state.fire["r"].colliderect(geo['plateA']) else 'water'
                    state.logger.log_switch_event('plateA', agent, plateA_on)
                state.bridge_up = plateA_on
            
            if 'plateB' in geo:
                plateB_on = (state.fire["r"].colliderect(geo['plateB']) or 
                            state.water["r"].colliderect(geo['plateB']))
                if plateB_on != state.gate_open:
                    agent = 'fire' if state.fire["r"].colliderect(geo['plateB']) else 'water'
                    state.logger.log_switch_event('plateB', agent, plateB_on)
                state.gate_open = plateB_on
        
        # Player movement
        keys = pg.key.get_pressed()
        
        # Create key dictionary for agent control
        agent_key_dict = {
            pg.K_a: 'left' in agent_actions['fire'],
            pg.K_d: 'right' in agent_actions['fire'],
            pg.K_w: 'jump' in agent_actions['fire'],
            pg.K_LEFT: 'left' in agent_actions['water'],
            pg.K_RIGHT: 'right' in agent_actions['water'],
            pg.K_UP: 'jump' in agent_actions['water']
        }
        
        # Fire movement
        if state.agent_mode == "human":
            move(state.fire, keys, pg.K_a, pg.K_d, pg.K_w, solids, 'fire', state.logger, False)
        else:
            move(state.fire, agent_key_dict, pg.K_a, pg.K_d, pg.K_w, solids, 'fire', state.logger, True)
        
        # Water movement
        if state.agent_mode in ["human", "hybrid"]:
            if state.agent_mode == "human":
                move(state.water, keys, pg.K_LEFT, pg.K_RIGHT, pg.K_UP, solids, 'water', state.logger, False)
            else:  # hybrid - human controls fire, AI controls water
                move(state.water, agent_key_dict, pg.K_LEFT, pg.K_RIGHT, pg.K_UP, solids, 'water', state.logger, True)
        else:  # agent mode
            move(state.water, agent_key_dict, pg.K_LEFT, pg.K_RIGHT, pg.K_UP, solids, 'water', state.logger, True)
        
        # Log positions
        state.logger.log_positions(
            state.fire["r"].topleft,
            state.water["r"].topleft,
            geo['exitF'],
            geo['exitW'],
            state.fire["r"],
            state.water["r"],
            solids
        )
        
        # Death check
        if 'water_pool' in geo and state.fire["r"].colliderect(geo['water_pool']):
            # Negative reward for death
            if state.agent_mode != "human" and state.last_state_summary:
                state.agent.record_episode_step(current_state, agent_actions, -50.0)
            state.retries += 1
            state.reset_level()
        if 'lava_pool' in geo and state.water["r"].colliderect(geo['lava_pool']):
            if state.agent_mode != "human" and state.last_state_summary:
                state.agent.record_episode_step(current_state, agent_actions, -50.0)
            state.retries += 1
            state.reset_level()
        
        # Win check
        fire_in = geo['exitF'].collidepoint(state.fire["r"].center)
        water_in = geo['exitW'].collidepoint(state.water["r"].center)
        if fire_in and water_in:
            state.done = True
            state.final_time = (pg.time.get_ticks() - state.t0) / 1000
            
            # Big reward for winning
            if state.agent_mode != "human" and state.last_state_summary:
                state.agent.record_episode_step(current_state, agent_actions, 100.0)
            
            state.logger.export_session(level.name, True, state.retries, state.agent_mode)
            if state.agent_mode != "human":
                state.agent.export_training_data()
    
    draw(state)
    C.tick(60)

pg.quit()
print("\nGame closed. Training data saved if AI mode was used.")
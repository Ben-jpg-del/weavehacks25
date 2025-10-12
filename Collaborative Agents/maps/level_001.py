import pygame as pg
import json
from datetime import datetime
from collections import deque
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
YELLOW, PURPLE = (255,220,60), (180,60,200)

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
        # Failure-run tracking
        self._last_level_name = None
        self._death_logged = False
        # per-frame action labels for BC
        self.fire_actions = []
        self.water_actions = []
        
    def log_keystroke(self, agent, keys_pressed):
        """Log which keys are pressed for each agent"""
        timestamp = pg.time.get_ticks() - self.start_time
        self.keystrokes.append({
            'time_ms': timestamp,
            'agent': agent,
            'keys': list(keys_pressed)
        })
    
    def log_positions(self, fire_pos, water_pos, fire_exit, water_exit,
                      fire_rect, water_rect, solids,
                      water_pool=None, lava_pool=None, plateA=None, plateB=None):
        """Log current positions, distances, and hazard/bridge exposure (center + bottom LOS)"""
        timestamp = pg.time.get_ticks() - self.start_time
        
        # Calculate distances to exits
        fire_dist = ((fire_pos[0] - fire_exit.centerx)**2 + (fire_pos[1] - fire_exit.centery)**2)**0.5
        water_dist = ((water_pos[0] - water_exit.centerx)**2 + (water_pos[1] - water_exit.centery)**2)**0.5
        
        # Calculate distances to pressure plates (STAGED OBJECTIVES)
        fire_plate_dist = None
        water_plate_dist = None
        if plateB:  # Fire usually goes to plateB
            fire_plate_dist = ((fire_pos[0] - plateB.centerx)**2 + (fire_pos[1] - plateB.centery)**2)**0.5
        if plateA:  # Water usually goes to plateA
            water_plate_dist = ((water_pos[0] - plateA.centerx)**2 + (water_pos[1] - plateA.centery)**2)**0.5
        
        # Count walls between sprite and goal (CENTER LOS)
        fire_walls = count_walls_between(fire_rect, fire_exit, solids)
        water_walls = count_walls_between(water_rect, water_exit, solids)

        # ---- CENTER LOS hazard crosses (geometry only)
        fire_LOS_haz  = count_threats_between(fire_rect,  fire_exit,  [water_pool] if water_pool else [])
        water_LOS_haz = count_threats_between(water_rect, water_exit, [lava_pool]  if lava_pool  else [])

        # ---- Exposure-aware hazard along CENTER LOS (solid before hazard => not exposed)
        _, fire_exposed_center = exposed_hazard_along_path(fire_rect,  fire_exit,  solids, water_pool) if water_pool else (0, False)
        _, water_exposed_center = exposed_hazard_along_path(water_rect, water_exit, solids, lava_pool)  if lava_pool  else (0, False)

        # ---- BOTTOM-EDGE parallel LOS (used to detect water/lava directly underneath path)
        fire_bottom_hits, fire_bottom_exposed = exposed_hazard_along_bottom_line(
            fire_rect, fire_exit, solids, water_pool
        ) if water_pool else (0, False)
        water_bottom_hits, water_bottom_exposed = exposed_hazard_along_bottom_line(
            water_rect, water_exit, solids, lava_pool
        ) if lava_pool else (0, False)

        # Interpretation per spec:
        # - "unsafe water ahead" for Fire if bottom line sees water before any solid AND the center LOS sees no solids
        fire_unsafe_bottom = bool(fire_bottom_exposed and fire_walls == 0)
        # - If bottom sees water but there *is* a solid on center LOS, it’s a bridge (hazard masked) and you still must jump
        fire_bridge_ahead  = bool(fire_bottom_hits and fire_walls > 0)

        water_unsafe_bottom = bool(water_bottom_exposed and water_walls == 0)
        water_bridge_ahead  = bool(water_bottom_hits and water_walls > 0)
        
        self.position_history.append({
            'time_ms': timestamp,
            'fire': {
                'x': fire_pos[0], 
                'y': fire_pos[1], 
                'dist_to_exit': fire_dist,
                'dist_to_plate': fire_plate_dist if fire_plate_dist is not None else fire_dist,
                'walls_blocking': fire_walls,
                'path_clear': fire_walls == 0,
                'los_hazard': int(fire_LOS_haz),              # center LOS hits hazard (geom)
                'hazard_exposed': bool(fire_exposed_center),  # center LOS exposed
                'bottom_los_hazard': int(fire_bottom_hits),   # bottom LOS hits hazard (geom)
                'bottom_exposed': bool(fire_bottom_exposed),  # bottom LOS exposed (no solid before)
                'unsafe_bottom': fire_unsafe_bottom,          # water ahead with no bridge
                'bridge_ahead': fire_bridge_ahead             # water present but bridge blocks -> need jump
            },
            'water': {
                'x': water_pos[0], 
                'y': water_pos[1], 
                'dist_to_exit': water_dist,
                'dist_to_plate': water_plate_dist if water_plate_dist is not None else water_dist,
                'walls_blocking': water_walls,
                'path_clear': water_walls == 0,
                'los_hazard': int(water_LOS_haz),
                'hazard_exposed': bool(water_exposed_center),
                'bottom_los_hazard': int(water_bottom_hits),
                'bottom_exposed': bool(water_bottom_exposed),
                'unsafe_bottom': water_unsafe_bottom,
                'bridge_ahead': water_bridge_ahead
            },
            'combined_dist': fire_dist + water_dist,
            'total_walls_blocking': fire_walls + water_walls
        })

        # --- Auto-export failed runs on death (before reset) ---
        try:
            for _lvl in LEVELS:
                g = _lvl.geometry
                if 'exitF' in g and 'exitW' in g and g['exitF'] == fire_exit and g['exitW'] == water_exit:
                    self._last_level_name = _lvl.name
                    fire_dead = ('water_pool' in g) and fire_rect.colliderect(g['water_pool'])
                    water_dead = ('lava_pool' in g) and water_rect.colliderect(g['lava_pool'])
                    if (fire_dead or water_dead) and not self._death_logged:
                        self.export_session(self._last_level_name, success=False, retries=0)
                        self._death_logged = True
                    break
        except Exception:
            pass
    
    def log_switch_event(self, switch_name, activated_by, state):
        """Log when switches/plates are activated or deactivated"""
        timestamp = pg.time.get_ticks() - self.start_time
        self.switch_events.append({
            'time_ms': timestamp,
            'switch': switch_name,
            'agent': activated_by,
            'state': state
        })
    
    def log_frame(self, game_state):
        """Log complete frame state for replay/analysis"""
        timestamp = pg.time.get_ticks() - self.start_time
        self.frame_data.append({
            'time_ms': timestamp,
            'state': game_state.copy()
        })
    
    def export_session(self, level_name, success, retries):
        """Export complete session data to JSON"""
        total_time = (pg.time.get_ticks() - self.start_time) / 1000
        data = {
            'timestamp': datetime.now().isoformat(),
            'level': level_name,
            'success': success,
            'time_seconds': total_time,
            'retries': retries,
            'keystrokes': self.keystrokes,
            'position_history': self.position_history[-100:],  # Last 100 samples
            'switch_events': self.switch_events,
            # include per-frame BC labels
            'fire_actions': self.fire_actions,
            'water_actions': self.water_actions,
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

# Level 1: Original tutorial level
LEVEL_1 = Level(
    "Tutorial",
    (420, 380),
    (60, 384),
    {
        'base_solids': [
            pg.Rect(0, 500, W, 40),      # floor
            pg.Rect(40, 420, 200, 20),   # left platform
            pg.Rect(400, 420, 120, 20),  # midL
            pg.Rect(700, 420, 120, 20),  # midR
            pg.Rect(40, 340, 240, 20)    # upper platform
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

def move(p, keys, left, right, jump, solids, agent_name, logger):
    VMAX, ACC, GRAV = 3.4, 0.9, 0.5
    
    # Log keys pressed
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

# ---------------- Geometry helpers ----------------
def line_segments_intersect(p1, p2, p3, p4):
    """Check if line segment p1-p2 intersects with line segment p3-p4"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return False  # Parallel or coincident
    
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
    
    return 0 <= t <= 1 and 0 <= u <= 1

def segment_intersection_t(p1, p2, q1, q2):
    """Return t along p1->p2 where it intersects q1->q2, else None."""
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
    """Smallest t where segment p1->p2 hits rect edges (or 0 if starting inside)."""
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
    """Check if line segment from p1 to p2 intersects rectangle"""
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

def count_walls_between(start_rect, end_rect, solids):
    """Count how many solids intersect the direct line between start and end (CENTER LOS)."""
    start_center = start_rect.center
    end_center = end_rect.center
    wall_count = 0
    for solid in solids:
        if line_intersects_rect(start_center, end_center, solid):
            wall_count += 1
    return wall_count

def count_threats_between(start_rect, end_rect, hazard_rects):
    """Count how many hazard rects the CENTER LOS crosses (pure geometry)."""
    if not hazard_rects:
        return 0
    p1 = start_rect.center
    p2 = end_rect.center
    hits = 0
    for hz in hazard_rects:
        if hz and line_intersects_rect(p1, p2, hz):
            hits += 1
    return hits

def exposed_hazard_between_points(p1, p2, solids, hazard_rect):
    """
    Generic: determine if segment p1->p2 hits hazard, and whether a solid
    lies BEFORE the hazard along the same segment.
    Returns (hazard_hits: int(0/1), exposed: bool)
    """
    if not hazard_rect:
        return (0, False)
    if not line_intersects_rect(p1, p2, hazard_rect):
        return (0, False)

    t_hz = first_intersection_t_with_rect(p1, p2, hazard_rect)
    if t_hz is None:
        return (1, False)

    t_solid = None
    for s in solids:
        t = first_intersection_t_with_rect(p1, p2, s)
        if t is not None:
            t_solid = t if t_solid is None else min(t_solid, t)

    exposed = (t_solid is None) or (t_solid > t_hz + 1e-6)
    return (1, exposed)

def exposed_hazard_along_path(start_rect, end_rect, solids, hazard_rect):
    """CENTER LOS convenience wrapper."""
    p1 = start_rect.center
    p2 = end_rect.center
    return exposed_hazard_between_points(p1, p2, solids, hazard_rect)

def exposed_hazard_along_bottom_line(start_rect, end_rect, solids, hazard_rect):
    """
    BOTTOM-EDGE LOS: start from block bottom center, extend a line PARALLEL to the
    center LOS (same vector as center->exit), used to detect water/lava directly
    underneath the path.
    """
    c1 = pg.math.Vector2(start_rect.center)
    c2 = pg.math.Vector2(end_rect.center)
    v = c2 - c1
    if v.length_squared() == 0:
        v = pg.math.Vector2(1, 0)
    p1 = (start_rect.centerx, start_rect.bottom - 1)  # slight -1 to stay within platform top
    p2 = (p1[0] + v.x, p1[1] + v.y)
    return exposed_hazard_between_points(p1, p2, solids, hazard_rect)

# ---------------- Drawing ----------------
def draw(state):
    G.fill(BLACK)
    level = state.get_level()
    geo = level.geometry
    
    # Draw hazards
    if 'water_pool' in geo:
        pg.draw.rect(G, (30, 90, 200), geo['water_pool'])
    if 'lava_pool' in geo:
        pg.draw.rect(G, (200, 60, 30), geo['lava_pool'])
    
    # Draw solids
    for s in level.get_solids(state.bridge_up, state.gate_open):
        pg.draw.rect(G, GRAY, s)
    
    # Draw plates
    if 'plateA' in geo:
        pg.draw.rect(G, BLUE, geo['plateA'])
    if 'plateB' in geo:
        pg.draw.rect(G, RED, geo['plateB'])
    
    # Draw exits
    pg.draw.rect(G, BLUE, geo['exitW'], 2)
    pg.draw.rect(G, RED, geo['exitF'], 2)
    
    # Draw players
    pg.draw.rect(G, state.fire["c"], state.fire["r"])
    pg.draw.rect(G, state.water["c"], state.water["r"])
    
    # LOS + hazards
    solids = level.get_solids(state.bridge_up, state.gate_open)

    # CENTER LOS walls
    fire_walls = count_walls_between(state.fire["r"], geo['exitF'], solids)
    water_walls = count_walls_between(state.water["r"], geo['exitW'], solids)

    # CENTER LOS hazard exposure
    _, fire_exposed_center = exposed_hazard_along_path(state.fire["r"], geo['exitF'], solids, geo.get('water_pool'))
    _, water_exposed_center = exposed_hazard_along_path(state.water["r"], geo['exitW'], solids, geo.get('lava_pool'))

    # BOTTOM-EDGE LOS hazard exposure
    fire_bottom_hits, fire_bottom_exposed = exposed_hazard_along_bottom_line(state.fire["r"], geo['exitF'], solids, geo.get('water_pool'))
    water_bottom_hits, water_bottom_exposed = exposed_hazard_along_bottom_line(state.water["r"], geo['exitW'], solids, geo.get('lava_pool'))

    fire_unsafe_bottom = bool(fire_bottom_exposed and fire_walls == 0)
    fire_bridge_ahead  = bool(fire_bottom_hits and fire_walls > 0)

    water_unsafe_bottom = bool(water_bottom_exposed and water_walls == 0)
    water_bridge_ahead  = bool(water_bottom_hits and water_walls > 0)
    
    # Base CENTER LOS lines (green if clear of walls, tinted if walls block)
    line_color_f = (255, 100, 100) if fire_walls > 0 else (100, 255, 100)
    line_color_w = (100, 100, 255) if water_walls > 0 else (100, 255, 100)
    pg.draw.line(G, line_color_f, state.fire["r"].center, geo['exitF'].center, 1)
    pg.draw.line(G, line_color_w, state.water["r"].center, geo['exitW'].center, 1)

    # Draw BOTTOM-EDGE LOS for debug (thin purple)
    fire_bottom_start = (state.fire["r"].centerx, state.fire["r"].bottom - 1)
    fire_bottom_end   = (fire_bottom_start[0] + (geo['exitF'].centerx - state.fire["r"].centerx),
                         fire_bottom_start[1] + (geo['exitF'].centery - state.fire["r"].centery))
    water_bottom_start = (state.water["r"].centerx, state.water["r"].bottom - 1)
    water_bottom_end   = (water_bottom_start[0] + (geo['exitW'].centerx - state.water["r"].centerx),
                          water_bottom_start[1] + (geo['exitW'].centery - state.water["r"].centery))
    pg.draw.line(G, PURPLE, fire_bottom_start, fire_bottom_end, 1)
    pg.draw.line(G, PURPLE, water_bottom_start, water_bottom_end, 1)

    # Overlay hazard warnings based on new logic
    # Exposed center (rare) OR unsafe bottom (typical case when water under path and no bridge)
    if fire_exposed_center or fire_unsafe_bottom:
        pg.draw.line(G, YELLOW, state.fire["r"].center, geo['exitF'].center, 3)
        if 'water_pool' in geo:
            pg.draw.rect(G, YELLOW, geo['water_pool'], 2)
    # Bridge ahead case: show center LOS in red already (wall), keep pool outline subtle
    if fire_bridge_ahead and 'water_pool' in geo:
        pg.draw.rect(G, (255, 255, 120), geo['water_pool'], 1)

    if water_exposed_center or water_unsafe_bottom:
        pg.draw.line(G, YELLOW, state.water["r"].center, geo['exitW'].center, 3)
        if 'lava_pool' in geo:
            pg.draw.rect(G, YELLOW, geo['lava_pool'], 2)
    if water_bridge_ahead and 'lava_pool' in geo:
        pg.draw.rect(G, (255, 255, 120), geo['lava_pool'], 1)
    
    # HUD
    tt = (pg.time.get_ticks() - state.t0) / 1000 if not state.done else state.final_time
    hud = f"Level {state.current_level_idx+1}: {level.name} | Time {tt:.2f}s | Retries {state.retries}"
    G.blit(F.render(hud, True, WHITE), (12, 10))
    G.blit(F.render("A/D/W Fire | ←/→/↑ Water | R reset | N/P level | E export", True, WHITE), (12, 34))
    
    # Win message
    if state.done:
        box = pg.Rect(0, 0, 400, 80)
        box.center = (W//2, 80)
        pg.draw.rect(G, (35, 35, 50), box, 0, 8)
        pg.draw.rect(G, GREEN, box, 2, 8)
        G.blit(F.render(f"SUCCESS: {state.final_time:.2f}s, Retries {state.retries}", True, WHITE), (box.x+14, box.y+24))
    
    # Composite to window
    S.fill(BLACK)
    scaled = pg.transform.smoothscale(G, (int(W*SCALE), int(H*SCALE)))
    S.blit(scaled, (0, 0))
    
    # Right panel - Live stats
    panel_x = int(W*SCALE)
    panel = pg.Rect(panel_x, 0, PANEL_W, SH)
    pg.draw.rect(S, (28, 28, 42), panel)
    
    S.blit(F.render("LIVE DATA LOG", True, WHITE), (panel.x+12, 12))
    
    y = 40
    # Positions and distances
    geo = level.geometry
    fire_dist = ((state.fire["r"].centerx - geo['exitF'].centerx)**2 + 
                 (state.fire["r"].centery - geo['exitF'].centery)**2)**0.5
    water_dist = ((state.water["r"].centerx - geo['exitW'].centerx)**2 + 
                  (state.water["r"].centery - geo['exitW'].centery)**2)**0.5
    
    S.blit(F_SMALL.render(f"Fire Pos: ({state.fire['r'].x}, {state.fire['r'].y})", True, RED), (panel.x+12, y)); y += 20
    S.blit(F_SMALL.render(f"Exit Dist: {fire_dist:.1f}px", True, RED), (panel.x+12, y)); y += 18
    wall_color = RED if fire_walls > 0 else GREEN
    S.blit(F_SMALL.render(f"Walls (center LOS): {fire_walls}", True, wall_color), (panel.x+12, y)); y += 18
    # Fire hazard legend using bottom LOS logic
    fire_status = "EXPOSED" if fire_unsafe_bottom or fire_exposed_center else ("BRIDGED" if fire_bridge_ahead else "NONE")
    hz_color = YELLOW if fire_status == "EXPOSED" else (GREEN if fire_status == "NONE" else WHITE)
    S.blit(F_SMALL.render(f"Ground Hazard (water): {fire_status}", True, hz_color), (panel.x+12, y)); y += 25
    
    S.blit(F_SMALL.render(f"Water Pos: ({state.water['r'].x}, {state.water['r'].y})", True, BLUE), (panel.x+12, y)); y += 20
    S.blit(F_SMALL.render(f"Exit Dist: {water_dist:.1f}px", True, BLUE), (panel.x+12, y)); y += 18
    wall_color = RED if water_walls > 0 else GREEN
    S.blit(F_SMALL.render(f"Walls (center LOS): {water_walls}", True, wall_color), (panel.x+12, y)); y += 18
    water_status = "EXPOSED" if water_unsafe_bottom or water_exposed_center else ("BRIDGED" if water_bridge_ahead else "NONE")
    hz_color = YELLOW if water_status == "EXPOSED" else (GREEN if water_status == "NONE" else WHITE)
    S.blit(F_SMALL.render(f"Ground Hazard (lava): {water_status}", True, hz_color), (panel.x+12, y)); y += 25
    
    # Switch states
    S.blit(F_SMALL.render("SWITCHES:", True, YELLOW), (panel.x+12, y)); y += 20
    if 'plateA' in geo:
        plateA_active = (state.fire["r"].colliderect(geo['plateA']) or 
                        state.water["r"].colliderect(geo['plateA']))
        color = GREEN if plateA_active else GRAY
        S.blit(F_SMALL.render(f"Plate A: {'ON' if plateA_active else 'OFF'}", True, color), (panel.x+12, y)); y += 18
    
    if 'plateB' in geo:
        plateB_active = (state.fire["r"].colliderect(geo['plateB']) or 
                        state.water["r"].colliderect(geo['plateB']))
        color = GREEN if plateB_active else GRAY
        S.blit(F_SMALL.render(f"Plate B: {'ON' if plateB_active else 'OFF'}", True, color), (panel.x+12, y)); y += 18
    
    y += 10
    S.blit(F_SMALL.render(f"Bridge: {'UP' if state.bridge_up else 'DOWN'}", True, GREEN if state.bridge_up else GRAY), (panel.x+12, y)); y += 18
    S.blit(F_SMALL.render(f"Gate: {'OPEN' if state.gate_open else 'CLOSED'}", True, GREEN if state.gate_open else GRAY), (panel.x+12, y)); y += 25
    
    S.blit(F_SMALL.render("STATISTICS:", True, YELLOW), (panel.x+12, y)); y += 20
    S.blit(F_SMALL.render(f"Keystrokes: {len(state.logger.keystrokes)}", True, WHITE), (panel.x+12, y)); y += 18
    S.blit(F_SMALL.render(f"Switch Events: {len(state.logger.switch_events)}", True, WHITE), (panel.x+12, y)); y += 18
    S.blit(F_SMALL.render(f"Positions Logged: {len(state.logger.position_history)}", True, WHITE), (panel.x+12, y))
    
    pg.display.flip()

# ---- Mapping held keys → action id (for BC labels) ----
# 0 none, 1 L, 2 R, 3 J, 4 L+J, 5 R+J
def map_keys_to_action(keys, left, right, jump):
    L = keys[left]; R = keys[right]; J = keys[jump]
    if L and J: return 4
    if R and J: return 5
    if L:       return 1
    if R:       return 2
    if J:       return 3
    return 0

# ---- Main Loop ----
state = GameState()
running = True

while running:
    for e in pg.event.get():
        if e.type == pg.QUIT:
            running = False
        if e.type == pg.KEYDOWN:
            if e.key == pg.K_r:
                state.retries += 1
                state.reset_level()
            if e.key == pg.K_n:  # Next level
                state.next_level()
            if e.key == pg.K_p:  # Previous level
                state.prev_level()
            if e.key == pg.K_e:  # Export data
                state.logger.export_session(
                    state.get_level().name,
                    state.done,
                    state.retries
                )
    
    if not state.done:
        level = state.get_level()
        geo = level.geometry
        solids = level.get_solids(state.bridge_up, state.gate_open)
        
        # Standard single-plate logic
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

        # per-frame held-action labels (use RAW human keys)
        a_fire  = map_keys_to_action(keys, pg.K_a,   pg.K_d,   pg.K_w)
        a_water = map_keys_to_action(keys, pg.K_LEFT,pg.K_RIGHT,pg.K_UP)
        state.logger.fire_actions.append(a_fire)
        state.logger.water_actions.append(a_water)

        move(state.fire, keys, pg.K_a, pg.K_d, pg.K_w, solids, 'fire', state.logger)
        move(state.water, keys, pg.K_LEFT, pg.K_RIGHT, pg.K_UP, solids, 'water', state.logger)
        
        # Log positions every frame (now includes bottom-LOS signals + staged objectives)
        state.logger.log_positions(
            state.fire["r"].topleft,
            state.water["r"].topleft,
            geo['exitF'], geo['exitW'],
            state.fire["r"], state.water["r"],
            solids,
            water_pool=geo.get('water_pool'),
            lava_pool=geo.get('lava_pool'),
            plateA=geo.get('plateA'),
            plateB=geo.get('plateB')
        )
        
        # Death check
        if 'water_pool' in geo and state.fire["r"].colliderect(geo['water_pool']):
            state.retries += 1
            state.reset_level()
        if 'lava_pool' in geo and state.water["r"].colliderect(geo['lava_pool']):
            state.retries += 1
            state.reset_level()
        
        # Win check
        fire_in = geo['exitF'].collidepoint(state.fire["r"].center)
        water_in = geo['exitW'].collidepoint(state.water["r"].center)
        if fire_in and water_in:
            state.done = True
            state.final_time = (pg.time.get_ticks() - state.t0) / 1000
            state.logger.export_session(level.name, True, state.retries)
    
    draw(state)
    C.tick(60)

pg.quit()

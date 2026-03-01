"""
8 PUZZLE SOLVER - A* Algorithm con Pygame
Configuración inicial/final personalizable, A* con heurística Manhattan
"""
import pygame, sys, heapq, numpy as np, time
from itertools import count

WIDTH, HEIGHT, TILE_SIZE, FPS = 700, 520, 120, 60
GOAL_STATE = np.array([[1,2,3],[4,5,6],[7,8,0]])
COLORS = {
    'bg': (20,20,30), 'tile': (52,152,219), 'empty': (100,100,120),
    'text': (255,255,255), 'button': (46,204,113), 'config': (241,196,15),
    'border': (200,200,200), 'info': (40,50,70)
}

def manhattan(state):
    """Heurística Manhattan para A*"""
    dist = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                gx, gy = divmod(val-1, 3)
                dist += abs(gx-i) + abs(gy-j)
    return dist

def find_zero(state):
    """Posición del espacio vacío"""
    return tuple(np.argwhere(state == 0)[0])

def get_neighbors(state):
    """Estados vecinos válidos"""
    x, y = find_zero(state)
    neighbors = []
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new = state.copy()
            new[x,y], new[nx,ny] = new[nx,ny], new[x,y]
            neighbors.append(new)
    return neighbors

def a_star(start, goal):
    """A* para resolver el puzzle"""
    open_list, visited, counter = [], set(), count()
    heapq.heappush(open_list, (0, next(counter), start, []))
    
    while open_list:
        _, _, state, path = heapq.heappop(open_list)
        key = tuple(state.flatten())
        if key in visited: continue
        visited.add(key)
        path = path + [state]
        if np.array_equal(state, goal): return path
        for neighbor in get_neighbors(state):
            k = tuple(neighbor.flatten())
            if k not in visited:
                g, f = len(path), len(path) + manhattan(neighbor)
                heapq.heappush(open_list, (f, next(counter), neighbor, path))
    return None

class PuzzleGUI:
    """GUI del 8 puzzle. Modos: config, play, solving"""
    
    BUTTONS = {
        'config': [('COMENZAR JUEGO', 30, 'start_game'), ('USAR COMO GOAL', 250, 'set_goal'), ('RESETEAR', 470, 'reset')],
        'play': [('RESOLVER', 30, 'solve'), ('VOLVER', 250, 'back')],
        'solving': [('SALTAR', 30, 'skip'), ('VOLVER', 250, 'back'), ('DE NUEVO', 470, 'solve')]
    }

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("8 Puzzle Solver - A* Algorithm")
        self.font = pygame.font.SysFont("Arial", 32, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 20)
        self.clock = pygame.time.Clock()
        
        self.mode = 'config'
        self.state = GOAL_STATE.copy()
        self.moves = self.solve_time = self.total_steps = 0
        self.solution = []
        self.step = 0

    def draw(self):
        self.screen.fill(COLORS['bg'])
        
        titles = {'config': ("MODO CONFIGURACIÓN - Haz clic en fichas", COLORS['config']),
                  'play': ("MODO JUEGO - ¡Resuelve el puzzle!", (100,200,255)),
                  'solving': ("RESOLVIENDO CON A*...", COLORS['button'])}
        title, color = titles[self.mode]
        self.screen.blit(self.small_font.render(title, True, color), (10, 5))

        for i in range(3):
            for j in range(3):
                val = self.state[i][j]
                rect = pygame.Rect(j*TILE_SIZE+10, i*TILE_SIZE+40, TILE_SIZE-2, TILE_SIZE-2)
                c = COLORS['tile'] if val else COLORS['empty']
                pygame.draw.rect(self.screen, c, rect, border_radius=10)
                if val: self.screen.blit(self.font.render(str(val), True, COLORS['text']), rect.center)
                pygame.draw.rect(self.screen, COLORS['border'], rect, 3, border_radius=10)

        info_rect = pygame.Rect(10, 400, 680, 110)
        pygame.draw.rect(self.screen, COLORS['info'], info_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLORS['border'], info_rect, 2, border_radius=8)
        
        infos = {'config': ("Configura la posición inicial. No se cuentan los movimientos.", COLORS['config']),
                 'play': (f"Movimientos realizados: {self.moves}", COLORS['text']),
                 'solving': (f"Solución: {self.step}/{self.total_steps} | Tiempo: {self.solve_time:.3f}s", COLORS['button'])}
        text, color = infos[self.mode]
        self.screen.blit(self.small_font.render(text, True, color), (20, 410))

        self._draw_buttons()
        pygame.display.flip()

    def _draw_buttons(self):
        for label, x, action in self.BUTTONS[self.mode]:
            rect = pygame.Rect(x, 450, 200, 40)
            if 'RESETEAR' in label or 'VOLVER' in label:
                color = (220, 50, 50)
            elif 'COMENZAR' in label:
                color = COLORS['config']
            else:
                color = COLORS['button']
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            pygame.draw.rect(self.screen, COLORS['border'], rect, 2, border_radius=6)
            self.screen.blit(self.small_font.render(label, True, COLORS['text']), 
                           (x+20, 456) if 'SALTAR' in label else (x+30 if x<50 else x+25, 456))

    def click_tile(self, x, y):
        if self.mode not in ['config', 'play']: return
        x, y = y // TILE_SIZE, x // TILE_SIZE
        if not (0 <= x < 3 and 0 <= y < 3): return
        zx, zy = find_zero(self.state)
        if abs(x-zx) + abs(y-zy) == 1:
            self.state[zx,zy], self.state[x,y] = self.state[x,y], self.state[zx,zy]
            if self.mode == 'play': self.moves += 1

    def solve(self):
        print(f"\n{'='*60}\nBÚSQUEDA A*\n{'='*60}")
        start = time.time()
        self.solution = a_star(self.state, GOAL_STATE)
        self.solve_time = time.time() - start
        self.total_steps = len(self.solution)-1 if self.solution else 0
        
        if self.solution:
            print(f"✓ SOLUCIÓN: {self.total_steps} pasos en {self.solve_time:.4f}s\n")
            for i, s in enumerate(self.solution):
                print(f"Paso {i}:\n{s}\n")
            self.mode = 'solving'
            self.step = 0
        else:
            print("✗ Sin solución")

    def handle_click(self, mx, my):
        if 10 <= mx <= 370 and 40 <= my <= 400:
            self.click_tile(mx, my)
        elif my > 440:
            for label, x, action in self.BUTTONS[self.mode]:
                if x < mx < x+200:
                    {'start_game': self.start_game, 'set_goal': self.set_goal, 'reset': self.reset,
                     'solve': self.solve, 'back': self.back, 'skip': self.skip}[action]()
                    break

    def start_game(self):
        self.mode, self.moves = 'play', 0
        print("\n✓ MODO JUEGO: Los movimientos serán contados.\n")

    def set_goal(self):
        global GOAL_STATE
        GOAL_STATE = self.state.copy()
        print(f"\n✓ GOAL establecido:\n{GOAL_STATE}\n")

    def reset(self):
        self.state, self.moves, self.mode = GOAL_STATE.copy(), 0, 'config'
        print("\n✓ Resetear a CONFIGURACIÓN.\n")

    def back(self):
        self.mode, self.moves, self.solution, self.step = 'config', 0, [], 0
        print("\n✓ Volviendo a CONFIGURACIÓN.\n")

    def skip(self):
        if self.solution:
            self.state, self.step, self.mode = self.solution[-1], len(self.solution), 'play'
            print("✓ Animación saltada.\n")

    def animate(self):
        if self.step < len(self.solution):
            self.state = self.solution[self.step]
            self.step += 1
            time.sleep(0.5)
        else:
            self.mode = 'play'
            print("✓ Animación completada.\n")

    def run(self):
        print("="*60+"\nBIENVENIDO AL 8 PUZZLE SOLVER\nA* con heurística Manhattan\n"+"="*60+"\n")
        running = True
        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(*pygame.mouse.get_pos())
            if self.mode == 'solving': self.animate()
            self.draw()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    print("="*60+"\nBIENVENIDO AL 8 PUZZLE SOLVER\nA* + Manhattan\n"+"="*60+"\n")
    gui = PuzzleGUI()
    gui.run()

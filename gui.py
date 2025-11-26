"""
Alpha2048 - Футуристичный GUI
==============================

Современный графический интерфейс с:
- Неоновым дизайном
- Анимациями
- Главным меню
- Режимами игры
- Визуализацией AI
- Системой бонусов

Управление:
- Стрелки / WASD - движение
- B - использовать бонус удаления
- T - использовать супер-бонус сортировки
- ESC - меню/пауза
- SPACE - AI сделает ход
"""

import pygame
import pygame.freetype  # Используем freetype вместо font для совместимости с Python 3.14
import numpy as np
import math
import time
from typing import Optional, Tuple, Dict, List
from enum import Enum, auto

from game_2048 import Game2048, Direction


# ============================================================================
# CONSTANTS & COLORS
# ============================================================================

# Window
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 900
FPS = 60

# Board
BOARD_SIZE = 500
BOARD_PADDING = 15
TILE_GAP = 10
BOARD_X = (WINDOW_WIDTH - BOARD_SIZE) // 2
BOARD_Y = 250

# Colors - Neon Futuristic Theme
COLORS = {
    'bg': (15, 15, 25),
    'bg_gradient': (25, 25, 45),
    'board_bg': (30, 30, 50),
    'tile_empty': (45, 45, 70),
    'text': (255, 255, 255),
    'text_dim': (150, 150, 170),
    'neon_cyan': (0, 255, 255),
    'neon_magenta': (255, 0, 255),
    'neon_yellow': (255, 255, 0),
    'neon_green': (0, 255, 128),
    'neon_orange': (255, 165, 0),
    'neon_red': (255, 50, 50),
    'neon_blue': (50, 150, 255),
    'gold': (255, 215, 0),
    'combo': (255, 100, 255),
}

# Tile colors based on value (gradient from cool to warm)
TILE_COLORS = {
    0: (45, 45, 70),
    2: (70, 130, 180),
    4: (100, 149, 237),
    8: (65, 105, 225),
    16: (138, 43, 226),
    32: (186, 85, 211),
    64: (255, 105, 180),
    128: (255, 165, 0),
    256: (255, 140, 0),
    512: (255, 99, 71),
    1024: (255, 69, 0),
    2048: (255, 215, 0),
    4096: (50, 255, 50),
    8192: (0, 255, 255),
    16384: (255, 0, 255),
    32768: (255, 255, 255),
    65536: (255, 255, 255),
}


class GameState(Enum):
    MENU = auto()
    PLAYING = auto()
    PAUSED = auto()
    GAME_OVER = auto()
    AI_WATCHING = auto()


# ============================================================================
# GUI CLASS
# ============================================================================

class Alpha2048GUI:
    """Футуристичный GUI для Alpha2048"""
    
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Alpha2048 - Infinite")
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts using freetype with None (default font) - compatible with Python 3.14
        # Using None loads the default pygame font without needing system font lookup
        self.font_title = pygame.freetype.Font(None, 56)
        self.font_large = pygame.freetype.Font(None, 38)
        self.font_medium = pygame.freetype.Font(None, 28)
        self.font_small = pygame.freetype.Font(None, 18)
        self.font_tile = pygame.freetype.Font(None, 32)
        self.font_tile_small = pygame.freetype.Font(None, 24)
        
        # Game
        self.game: Optional[Game2048] = None
        self.state = GameState.MENU
        self.selected_mode = 'infinite'
        
        # AI
        self.ai_agent = None
        self.ai_enabled = False
        self.ai_delay = 0.2
        self.last_ai_move = 0
        
        # Animation
        self.animations = []
        self.glow_phase = 0
        
        # Menu
        self.menu_selection = 0
        self.menu_items = [
            ('PLAY INFINITE', 'infinite'),
            ('CLASSIC MODE', 'classic'),
            ('WATCH AI', 'ai'),
            ('ABOUT', 'about'),
            ('EXIT', 'exit')
        ]
        
        # Bonus selection
        self.selecting_bonus_target = False
        self.bonus_type = None  # 'remove' or 'sort'
    
    def render_text(self, font, text: str, color, pos, center: bool = False):
        """Рендер текста с freetype"""
        surface, rect = font.render(text, color)
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        self.screen.blit(surface, rect)
        return rect
    
    def run(self):
        """Главный цикл"""
        running = True
        
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            self.glow_phase += dt * 2
            
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self.handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
            
            # AI move
            if self.state == GameState.AI_WATCHING and self.game:
                if time.time() - self.last_ai_move > self.ai_delay:
                    self.ai_move()
                    self.last_ai_move = time.time()
            
            # Render
            self.render()
            pygame.display.flip()
        
        pygame.quit()
    
    def handle_key(self, key) -> bool:
        """Обработка клавиш"""
        if self.state == GameState.MENU:
            if key == pygame.K_UP:
                self.menu_selection = (self.menu_selection - 1) % len(self.menu_items)
            elif key == pygame.K_DOWN:
                self.menu_selection = (self.menu_selection + 1) % len(self.menu_items)
            elif key in (pygame.K_RETURN, pygame.K_SPACE):
                return self.select_menu_item()
            elif key == pygame.K_ESCAPE:
                return False
        
        elif self.state in (GameState.PLAYING, GameState.AI_WATCHING):
            if self.selecting_bonus_target:
                # Выбор цели для бонуса удаления
                if key == pygame.K_ESCAPE:
                    self.selecting_bonus_target = False
                    self.bonus_type = None
                return True
            
            if key == pygame.K_ESCAPE:
                self.state = GameState.PAUSED
            elif key in (pygame.K_UP, pygame.K_w):
                self.make_move(Direction.UP)
            elif key in (pygame.K_DOWN, pygame.K_s):
                self.make_move(Direction.DOWN)
            elif key in (pygame.K_LEFT, pygame.K_a):
                self.make_move(Direction.LEFT)
            elif key in (pygame.K_RIGHT, pygame.K_d):
                self.make_move(Direction.RIGHT)
            elif key == pygame.K_b:
                # Бонус удаления
                if self.game and self.game.can_use_bonus():
                    self.selecting_bonus_target = True
                    self.bonus_type = 'remove'
            elif key == pygame.K_t:
                # Супер-бонус сортировки (T)
                if self.game and self.game.can_use_sort_bonus():
                    self.game.use_sort_bonus()
            elif key == pygame.K_SPACE:
                # AI делает один ход
                self.ai_move()
        
        elif self.state == GameState.PAUSED:
            if key == pygame.K_ESCAPE:
                self.state = GameState.PLAYING
            elif key == pygame.K_q:
                self.state = GameState.MENU
        
        elif self.state == GameState.GAME_OVER:
            if key in (pygame.K_RETURN, pygame.K_SPACE):
                self.state = GameState.MENU
        
        return True
    
    def handle_click(self, pos):
        """Обработка кликов мыши"""
        x, y = pos
        
        if self.selecting_bonus_target and self.game:
            # Проверяем клик по тайлу
            tile_size = (BOARD_SIZE - 2 * BOARD_PADDING - 3 * TILE_GAP) // 4
            
            for row in range(4):
                for col in range(4):
                    tx = BOARD_X + BOARD_PADDING + col * (tile_size + TILE_GAP)
                    ty = BOARD_Y + BOARD_PADDING + row * (tile_size + TILE_GAP)
                    
                    if tx <= x <= tx + tile_size and ty <= y <= ty + tile_size:
                        if self.bonus_type == 'remove':
                            if self.game.use_bonus_remove_tile(row, col):
                                self.selecting_bonus_target = False
                                self.bonus_type = None
                        return
        
        if self.state == GameState.MENU:
            # Проверяем клик по пунктам меню
            menu_y = 350
            for i, (text, action) in enumerate(self.menu_items):
                item_rect = pygame.Rect(
                    WINDOW_WIDTH // 2 - 150,
                    menu_y + i * 60,
                    300, 50
                )
                if item_rect.collidepoint(pos):
                    self.menu_selection = i
                    self.select_menu_item()
                    break
    
    def select_menu_item(self) -> bool:
        """Выбор пункта меню"""
        _, action = self.menu_items[self.menu_selection]
        
        if action == 'exit':
            return False
        elif action == 'about':
            pass  # TODO: показать about
        elif action == 'ai':
            self.start_game('infinite')
            self.state = GameState.AI_WATCHING
            self.ai_enabled = True
        else:
            self.start_game(action)
            self.state = GameState.PLAYING
        
        return True
    
    def start_game(self, mode: str):
        """Начать новую игру"""
        self.game = Game2048(mode=mode)
        self.ai_enabled = False
        self.selecting_bonus_target = False
        self.bonus_type = None
    
    def make_move(self, direction: Direction):
        """Сделать ход"""
        if not self.game or self.game.is_game_over():
            return
        
        reward, done, info = self.game.move(direction)
        
        if info.get('combo_triggered'):
            # Показать эффект combo
            pass
        
        if done:
            self.state = GameState.GAME_OVER
    
    def ai_move(self):
        """AI делает ход"""
        if not self.game or self.game.is_game_over():
            if self.game and self.game.is_game_over():
                self.state = GameState.GAME_OVER
            return
        
        # Простая эвристика (можно заменить на Alpha2048)
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            self.state = GameState.GAME_OVER
            return
        
        # Приоритеты: DOWN > LEFT > RIGHT > UP
        priority = [Direction.DOWN, Direction.LEFT, Direction.RIGHT, Direction.UP]
        for d in priority:
            if d in valid_moves:
                self.make_move(d)
                return
        
        self.make_move(valid_moves[0])
    
    # ========================================================================
    # RENDERING
    # ========================================================================
    
    def render(self):
        """Отрисовка"""
        # Background gradient
        self.draw_gradient_bg()
        
        if self.state == GameState.MENU:
            self.render_menu()
        elif self.state in (GameState.PLAYING, GameState.AI_WATCHING):
            self.render_game()
        elif self.state == GameState.PAUSED:
            self.render_game()
            self.render_pause_overlay()
        elif self.state == GameState.GAME_OVER:
            self.render_game()
            self.render_game_over_overlay()
    
    def draw_gradient_bg(self):
        """Градиентный фон"""
        for y in range(WINDOW_HEIGHT):
            progress = y / WINDOW_HEIGHT
            r = int(COLORS['bg'][0] + (COLORS['bg_gradient'][0] - COLORS['bg'][0]) * progress)
            g = int(COLORS['bg'][1] + (COLORS['bg_gradient'][1] - COLORS['bg'][1]) * progress)
            b = int(COLORS['bg'][2] + (COLORS['bg_gradient'][2] - COLORS['bg'][2]) * progress)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (WINDOW_WIDTH, y))
    
    def render_menu(self):
        """Отрисовка меню"""
        # Title with glow
        glow = abs(math.sin(self.glow_phase)) * 0.5 + 0.5
        title_color = (
            int(COLORS['neon_cyan'][0] * glow + 255 * (1 - glow)),
            int(COLORS['neon_cyan'][1] * glow + 255 * (1 - glow)),
            int(COLORS['neon_cyan'][2] * glow + 255 * (1 - glow))
        )
        
        self.render_text(self.font_title, "ALPHA 2048", title_color, 
                        (WINDOW_WIDTH // 2, 120), center=True)
        
        # Subtitle
        self.render_text(self.font_medium, "INFINITE MODE", COLORS['neon_magenta'],
                        (WINDOW_WIDTH // 2, 180), center=True)
        
        # Decorative line
        pygame.draw.line(
            self.screen, COLORS['neon_cyan'],
            (WINDOW_WIDTH // 2 - 200, 220),
            (WINDOW_WIDTH // 2 + 200, 220), 2
        )
        
        # Menu items
        menu_y = 350
        for i, (text, action) in enumerate(self.menu_items):
            is_selected = i == self.menu_selection
            
            # Background
            if is_selected:
                glow_alpha = int(abs(math.sin(self.glow_phase * 2)) * 100 + 50)
                s = pygame.Surface((300, 50), pygame.SRCALPHA)
                s.fill((*COLORS['neon_cyan'][:3], glow_alpha))
                self.screen.blit(s, (WINDOW_WIDTH // 2 - 150, menu_y + i * 60))
                
                # Border
                pygame.draw.rect(
                    self.screen, COLORS['neon_cyan'],
                    (WINDOW_WIDTH // 2 - 150, menu_y + i * 60, 300, 50), 2
                )
            
            color = COLORS['text'] if is_selected else COLORS['text_dim']
            self.render_text(self.font_medium, text, color,
                           (WINDOW_WIDTH // 2, menu_y + i * 60 + 25), center=True)
        
        # Instructions
        self.render_text(self.font_small, "UP/DOWN Select  -  ENTER Confirm  -  ESC Exit", 
                        COLORS['text_dim'], (WINDOW_WIDTH // 2, WINDOW_HEIGHT - 50), center=True)
    
    def render_game(self):
        """Отрисовка игры"""
        if not self.game:
            return
        
        # Header
        self.render_header()
        
        # Board
        self.render_board()
        
        # Bonuses
        self.render_bonuses()
        
        # Controls hint
        self.render_controls()
        
        # Bonus selection overlay
        if self.selecting_bonus_target:
            self.render_bonus_selection()
    
    def render_header(self):
        """Заголовок с очками и статистикой"""
        # Score
        self.render_text(self.font_large, f"SCORE: {self.game.score:,}", 
                        COLORS['text'], (30, 30))
        
        # Max tile
        max_color = COLORS['gold'] if self.game.max_tile >= 2048 else COLORS['neon_cyan']
        self.render_text(self.font_medium, f"MAX: {self.game.max_tile:,}", 
                        max_color, (30, 80))
        
        # Moves
        self.render_text(self.font_small, f"MOVES: {self.game.moves}", 
                        COLORS['text_dim'], (30, 120))
        
        # Mode
        mode_color = COLORS['neon_magenta'] if self.game.mode == 'infinite' else COLORS['text_dim']
        self.render_text(self.font_small, f"MODE: {self.game.mode.upper()}", 
                        mode_color, (WINDOW_WIDTH - 180, 30))
        
        # Spawn info (dynamic mode)
        if self.game.mode != 'classic':
            spawn = self.game.get_spawn_tiles()
            self.render_text(self.font_small, f"SPAWN: {spawn[0]}/{spawn[1]}", 
                           COLORS['neon_green'], (WINDOW_WIDTH - 180, 60))
        
        # Combo counter
        if self.game.total_combos > 0:
            self.render_text(self.font_medium, f"COMBO x{self.game.total_combos}", 
                           COLORS['combo'], (WINDOW_WIDTH - 200, 100))
        
        # AI indicator
        if self.state == GameState.AI_WATCHING:
            self.render_text(self.font_medium, "AI PLAYING", 
                           COLORS['neon_green'], (WINDOW_WIDTH // 2, 30), center=True)
    
    def render_board(self):
        """Отрисовка доски"""
        # Board background with glow
        pygame.draw.rect(
            self.screen, COLORS['board_bg'],
            (BOARD_X - 5, BOARD_Y - 5, BOARD_SIZE + 10, BOARD_SIZE + 10),
            border_radius=15
        )
        
        # Board border
        glow = abs(math.sin(self.glow_phase * 0.5)) * 0.3 + 0.7
        border_color = tuple(int(c * glow) for c in COLORS['neon_cyan'])
        pygame.draw.rect(
            self.screen, border_color,
            (BOARD_X - 5, BOARD_Y - 5, BOARD_SIZE + 10, BOARD_SIZE + 10),
            3, border_radius=15
        )
        
        # Tiles
        tile_size = (BOARD_SIZE - 2 * BOARD_PADDING - 3 * TILE_GAP) // 4
        
        for row in range(4):
            for col in range(4):
                value = self.game.board[row, col]
                x = BOARD_X + BOARD_PADDING + col * (tile_size + TILE_GAP)
                y = BOARD_Y + BOARD_PADDING + row * (tile_size + TILE_GAP)
                
                self.render_tile(x, y, tile_size, value)
    
    def render_tile(self, x: int, y: int, size: int, value: int):
        """Отрисовка одного тайла"""
        # Get color
        if value in TILE_COLORS:
            color = TILE_COLORS[value]
        else:
            # For very large values
            color = TILE_COLORS[65536]
        
        # Tile background
        pygame.draw.rect(self.screen, color, (x, y, size, size), border_radius=8)
        
        # Glow for high values
        if value >= 2048:
            glow = abs(math.sin(self.glow_phase + value * 0.001)) * 0.5 + 0.5
            glow_surf = pygame.Surface((size + 10, size + 10), pygame.SRCALPHA)
            glow_color = (*COLORS['gold'][:3], int(100 * glow))
            pygame.draw.rect(glow_surf, glow_color, (0, 0, size + 10, size + 10), border_radius=10)
            self.screen.blit(glow_surf, (x - 5, y - 5))
            # Redraw tile on top
            pygame.draw.rect(self.screen, color, (x, y, size, size), border_radius=8)
        
        # Value text
        if value > 0:
            text_color = (255, 255, 255) if value > 4 else (50, 50, 50)
            
            if value >= 10000:
                font = self.font_tile_small
            else:
                font = self.font_tile
            
            self.render_text(font, str(value), text_color, 
                           (x + size // 2, y + size // 2), center=True)
    
    def render_bonuses(self):
        """Отрисовка бонусов"""
        if self.game.mode != 'infinite':
            return
        
        bonus_y = BOARD_Y + BOARD_SIZE + 30
        
        # Remove bonus
        if self.game.bonus_count > 0:
            self.render_text(self.font_medium, f"Remove: {self.game.bonus_count}  [B]",
                           COLORS['neon_orange'], (BOARD_X, bonus_y))
        
        # Sort bonus
        if self.game.sort_bonuses > 0:
            self.render_text(self.font_medium, f"Sort: {self.game.sort_bonuses}  [T]",
                           COLORS['neon_magenta'], (BOARD_X, bonus_y + 35))
    
    def render_controls(self):
        """Подсказки управления"""
        controls = "Arrows Move  -  SPACE AI Move  -  ESC Pause"
        self.render_text(self.font_small, controls, COLORS['text_dim'],
                        (WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30), center=True)
    
    def render_bonus_selection(self):
        """Оверлей выбора цели для бонуса"""
        # Darken
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        # Instruction
        self.render_text(self.font_large, "SELECT TILE TO REMOVE", 
                        COLORS['neon_orange'], (WINDOW_WIDTH // 2, 180), center=True)
        
        self.render_text(self.font_medium, "Click on a tile  -  ESC to cancel", 
                        COLORS['text'], (WINDOW_WIDTH // 2, 220), center=True)
        
        # Highlight board
        pygame.draw.rect(
            self.screen, COLORS['neon_orange'],
            (BOARD_X - 10, BOARD_Y - 10, BOARD_SIZE + 20, BOARD_SIZE + 20),
            4, border_radius=15
        )
        
        # Re-render board
        self.render_board()
    
    def render_pause_overlay(self):
        """Оверлей паузы"""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        self.render_text(self.font_title, "PAUSED", COLORS['neon_cyan'],
                        (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50), center=True)
        
        self.render_text(self.font_medium, "ESC - Resume  -  Q - Quit to Menu", 
                        COLORS['text'], (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 30), center=True)
    
    def render_game_over_overlay(self):
        """Оверлей окончания игры"""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        # Title
        self.render_text(self.font_title, "GAME OVER", COLORS['neon_red'],
                        (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100), center=True)
        
        # Final score
        self.render_text(self.font_large, f"FINAL SCORE: {self.game.score:,}", 
                        COLORS['gold'], (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 20), center=True)
        
        # Max tile
        self.render_text(self.font_medium, f"MAX TILE: {self.game.max_tile:,}", 
                        COLORS['text'], (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 30), center=True)
        
        # Combos
        if self.game.total_combos > 0:
            self.render_text(self.font_medium, f"COMBOS: {self.game.total_combos}", 
                           COLORS['combo'], (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 70), center=True)
        
        # Hint
        self.render_text(self.font_medium, "Press ENTER to continue", 
                        COLORS['text_dim'], (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 130), center=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Запуск GUI"""
    gui = Alpha2048GUI()
    gui.run()


if __name__ == "__main__":
    main()

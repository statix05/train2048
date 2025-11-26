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

import os
# Отключаем приветствие pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
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
# SIMPLE TEXT RENDERER (без pygame.font)
# ============================================================================

class SimpleFont:
    """Простой рендерер текста без использования pygame.font/freetype"""
    
    # 5x7 bitmap font для цифр и букв
    CHARS = {
        '0': ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
        '1': ['00100', '01100', '00100', '00100', '00100', '00100', '01110'],
        '2': ['01110', '10001', '00001', '00110', '01000', '10000', '11111'],
        '3': ['01110', '10001', '00001', '00110', '00001', '10001', '01110'],
        '4': ['00010', '00110', '01010', '10010', '11111', '00010', '00010'],
        '5': ['11111', '10000', '11110', '00001', '00001', '10001', '01110'],
        '6': ['00110', '01000', '10000', '11110', '10001', '10001', '01110'],
        '7': ['11111', '00001', '00010', '00100', '01000', '01000', '01000'],
        '8': ['01110', '10001', '10001', '01110', '10001', '10001', '01110'],
        '9': ['01110', '10001', '10001', '01111', '00001', '00010', '01100'],
        'A': ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
        'B': ['11110', '10001', '10001', '11110', '10001', '10001', '11110'],
        'C': ['01110', '10001', '10000', '10000', '10000', '10001', '01110'],
        'D': ['11110', '10001', '10001', '10001', '10001', '10001', '11110'],
        'E': ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
        'F': ['11111', '10000', '10000', '11110', '10000', '10000', '10000'],
        'G': ['01110', '10001', '10000', '10111', '10001', '10001', '01110'],
        'H': ['10001', '10001', '10001', '11111', '10001', '10001', '10001'],
        'I': ['01110', '00100', '00100', '00100', '00100', '00100', '01110'],
        'J': ['00111', '00010', '00010', '00010', '00010', '10010', '01100'],
        'K': ['10001', '10010', '10100', '11000', '10100', '10010', '10001'],
        'L': ['10000', '10000', '10000', '10000', '10000', '10000', '11111'],
        'M': ['10001', '11011', '10101', '10101', '10001', '10001', '10001'],
        'N': ['10001', '11001', '10101', '10011', '10001', '10001', '10001'],
        'O': ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
        'P': ['11110', '10001', '10001', '11110', '10000', '10000', '10000'],
        'Q': ['01110', '10001', '10001', '10001', '10101', '10010', '01101'],
        'R': ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
        'S': ['01110', '10001', '10000', '01110', '00001', '10001', '01110'],
        'T': ['11111', '00100', '00100', '00100', '00100', '00100', '00100'],
        'U': ['10001', '10001', '10001', '10001', '10001', '10001', '01110'],
        'V': ['10001', '10001', '10001', '10001', '10001', '01010', '00100'],
        'W': ['10001', '10001', '10001', '10101', '10101', '10101', '01010'],
        'X': ['10001', '10001', '01010', '00100', '01010', '10001', '10001'],
        'Y': ['10001', '10001', '01010', '00100', '00100', '00100', '00100'],
        'Z': ['11111', '00001', '00010', '00100', '01000', '10000', '11111'],
        ' ': ['00000', '00000', '00000', '00000', '00000', '00000', '00000'],
        ':': ['00000', '00100', '00100', '00000', '00100', '00100', '00000'],
        '-': ['00000', '00000', '00000', '11111', '00000', '00000', '00000'],
        '/': ['00001', '00010', '00010', '00100', '01000', '01000', '10000'],
        '[': ['01110', '01000', '01000', '01000', '01000', '01000', '01110'],
        ']': ['01110', '00010', '00010', '00010', '00010', '00010', '01110'],
        ',': ['00000', '00000', '00000', '00000', '00000', '00100', '01000'],
        '.': ['00000', '00000', '00000', '00000', '00000', '00000', '00100'],
        'x': ['00000', '00000', '10001', '01010', '00100', '01010', '10001'],
    }
    
    def __init__(self, scale: int = 3):
        self.scale = scale
        self.char_width = 5 * scale + scale  # 5 pixels + spacing
        self.char_height = 7 * scale
    
    def render(self, screen, text: str, color: Tuple[int, int, int], 
               pos: Tuple[int, int], center: bool = False):
        """Рендер текста"""
        text = text.upper()
        total_width = len(text) * self.char_width
        
        if center:
            start_x = pos[0] - total_width // 2
            start_y = pos[1] - self.char_height // 2
        else:
            start_x, start_y = pos
        
        for i, char in enumerate(text):
            if char in self.CHARS:
                self._draw_char(screen, char, color, 
                              start_x + i * self.char_width, start_y)
    
    def _draw_char(self, screen, char: str, color: Tuple[int, int, int], 
                   x: int, y: int):
        """Отрисовка одного символа"""
        bitmap = self.CHARS[char]
        for row_idx, row in enumerate(bitmap):
            for col_idx, pixel in enumerate(row):
                if pixel == '1':
                    pygame.draw.rect(
                        screen, color,
                        (x + col_idx * self.scale, 
                         y + row_idx * self.scale,
                         self.scale, self.scale)
                    )


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
        
        # Simple bitmap fonts (no pygame.font needed)
        self.font_title = SimpleFont(scale=5)
        self.font_large = SimpleFont(scale=4)
        self.font_medium = SimpleFont(scale=3)
        self.font_small = SimpleFont(scale=2)
        self.font_tile = SimpleFont(scale=4)
        self.font_tile_small = SimpleFont(scale=3)
        
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
        self.bonus_type = None
    
    def run(self):
        """Главный цикл"""
        running = True
        
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            self.glow_phase += dt * 2
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self.handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
            
            if self.state == GameState.AI_WATCHING and self.game:
                if time.time() - self.last_ai_move > self.ai_delay:
                    self.ai_move()
                    self.last_ai_move = time.time()
            
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
                if self.game and self.game.can_use_bonus():
                    self.selecting_bonus_target = True
                    self.bonus_type = 'remove'
            elif key == pygame.K_t:
                if self.game and self.game.can_use_sort_bonus():
                    self.game.use_sort_bonus()
            elif key == pygame.K_SPACE:
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
            pass
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
        
        if done:
            self.state = GameState.GAME_OVER
    
    def ai_move(self):
        """AI делает ход"""
        if not self.game or self.game.is_game_over():
            if self.game and self.game.is_game_over():
                self.state = GameState.GAME_OVER
            return
        
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            self.state = GameState.GAME_OVER
            return
        
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
        glow = abs(math.sin(self.glow_phase)) * 0.5 + 0.5
        title_color = (
            int(COLORS['neon_cyan'][0] * glow + 255 * (1 - glow)),
            int(COLORS['neon_cyan'][1] * glow + 255 * (1 - glow)),
            int(COLORS['neon_cyan'][2] * glow + 255 * (1 - glow))
        )
        
        self.font_title.render(self.screen, "ALPHA 2048", title_color, 
                               (WINDOW_WIDTH // 2, 100), center=True)
        
        self.font_medium.render(self.screen, "INFINITE MODE", COLORS['neon_magenta'],
                                (WINDOW_WIDTH // 2, 180), center=True)
        
        pygame.draw.line(
            self.screen, COLORS['neon_cyan'],
            (WINDOW_WIDTH // 2 - 200, 220),
            (WINDOW_WIDTH // 2 + 200, 220), 2
        )
        
        menu_y = 350
        for i, (text, action) in enumerate(self.menu_items):
            is_selected = i == self.menu_selection
            
            if is_selected:
                glow_alpha = int(abs(math.sin(self.glow_phase * 2)) * 100 + 50)
                s = pygame.Surface((300, 50), pygame.SRCALPHA)
                s.fill((*COLORS['neon_cyan'][:3], glow_alpha))
                self.screen.blit(s, (WINDOW_WIDTH // 2 - 150, menu_y + i * 60))
                
                pygame.draw.rect(
                    self.screen, COLORS['neon_cyan'],
                    (WINDOW_WIDTH // 2 - 150, menu_y + i * 60, 300, 50), 2
                )
            
            color = COLORS['text'] if is_selected else COLORS['text_dim']
            self.font_medium.render(self.screen, text, color,
                                    (WINDOW_WIDTH // 2, menu_y + i * 60 + 25), center=True)
        
        self.font_small.render(self.screen, "UP/DOWN SELECT - ENTER CONFIRM - ESC EXIT", 
                               COLORS['text_dim'], (WINDOW_WIDTH // 2, WINDOW_HEIGHT - 50), center=True)
    
    def render_game(self):
        """Отрисовка игры"""
        if not self.game:
            return
        
        self.render_header()
        self.render_board()
        self.render_bonuses()
        self.render_controls()
        
        if self.selecting_bonus_target:
            self.render_bonus_selection()
    
    def render_header(self):
        """Заголовок с очками и статистикой"""
        self.font_large.render(self.screen, f"SCORE: {self.game.score}", 
                               COLORS['text'], (30, 30))
        
        max_color = COLORS['gold'] if self.game.max_tile >= 2048 else COLORS['neon_cyan']
        self.font_medium.render(self.screen, f"MAX: {self.game.max_tile}", 
                                max_color, (30, 90))
        
        self.font_small.render(self.screen, f"MOVES: {self.game.moves}", 
                               COLORS['text_dim'], (30, 140))
        
        mode_color = COLORS['neon_magenta'] if self.game.mode == 'infinite' else COLORS['text_dim']
        self.font_small.render(self.screen, f"MODE: {self.game.mode.upper()}", 
                               mode_color, (WINDOW_WIDTH - 200, 30))
        
        if self.game.mode != 'classic':
            spawn = self.game.get_spawn_tiles()
            self.font_small.render(self.screen, f"SPAWN: {spawn[0]}/{spawn[1]}", 
                                   COLORS['neon_green'], (WINDOW_WIDTH - 200, 70))
        
        if self.game.total_combos > 0:
            self.font_medium.render(self.screen, f"COMBO x{self.game.total_combos}", 
                                    COLORS['combo'], (WINDOW_WIDTH - 220, 110))
        
        if self.state == GameState.AI_WATCHING:
            self.font_medium.render(self.screen, "AI PLAYING", 
                                    COLORS['neon_green'], (WINDOW_WIDTH // 2, 30), center=True)
    
    def render_board(self):
        """Отрисовка доски"""
        pygame.draw.rect(
            self.screen, COLORS['board_bg'],
            (BOARD_X - 5, BOARD_Y - 5, BOARD_SIZE + 10, BOARD_SIZE + 10),
            border_radius=15
        )
        
        glow = abs(math.sin(self.glow_phase * 0.5)) * 0.3 + 0.7
        border_color = tuple(int(c * glow) for c in COLORS['neon_cyan'])
        pygame.draw.rect(
            self.screen, border_color,
            (BOARD_X - 5, BOARD_Y - 5, BOARD_SIZE + 10, BOARD_SIZE + 10),
            3, border_radius=15
        )
        
        tile_size = (BOARD_SIZE - 2 * BOARD_PADDING - 3 * TILE_GAP) // 4
        
        for row in range(4):
            for col in range(4):
                value = self.game.board[row, col]
                x = BOARD_X + BOARD_PADDING + col * (tile_size + TILE_GAP)
                y = BOARD_Y + BOARD_PADDING + row * (tile_size + TILE_GAP)
                
                self.render_tile(x, y, tile_size, value)
    
    def render_tile(self, x: int, y: int, size: int, value: int):
        """Отрисовка одного тайла"""
        if value in TILE_COLORS:
            color = TILE_COLORS[value]
        else:
            color = TILE_COLORS[65536]
        
        pygame.draw.rect(self.screen, color, (x, y, size, size), border_radius=8)
        
        if value >= 2048:
            glow = abs(math.sin(self.glow_phase + value * 0.001)) * 0.5 + 0.5
            glow_surf = pygame.Surface((size + 10, size + 10), pygame.SRCALPHA)
            glow_color = (*COLORS['gold'][:3], int(100 * glow))
            pygame.draw.rect(glow_surf, glow_color, (0, 0, size + 10, size + 10), border_radius=10)
            self.screen.blit(glow_surf, (x - 5, y - 5))
            pygame.draw.rect(self.screen, color, (x, y, size, size), border_radius=8)
        
        if value > 0:
            text_color = (255, 255, 255) if value > 4 else (50, 50, 50)
            
            if value >= 10000:
                font = self.font_tile_small
            else:
                font = self.font_tile
            
            font.render(self.screen, str(value), text_color, 
                       (x + size // 2, y + size // 2), center=True)
    
    def render_bonuses(self):
        """Отрисовка бонусов"""
        if self.game.mode != 'infinite':
            return
        
        bonus_y = BOARD_Y + BOARD_SIZE + 30
        
        if self.game.bonus_count > 0:
            self.font_medium.render(self.screen, f"REMOVE: {self.game.bonus_count} [B]",
                                    COLORS['neon_orange'], (BOARD_X, bonus_y))
        
        if self.game.sort_bonuses > 0:
            self.font_medium.render(self.screen, f"SORT: {self.game.sort_bonuses} [T]",
                                    COLORS['neon_magenta'], (BOARD_X, bonus_y + 40))
    
    def render_controls(self):
        """Подсказки управления"""
        self.font_small.render(self.screen, "ARROWS MOVE - SPACE AI - ESC PAUSE", 
                               COLORS['text_dim'], (WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30), center=True)
    
    def render_bonus_selection(self):
        """Оверлей выбора цели для бонуса"""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        self.font_large.render(self.screen, "SELECT TILE TO REMOVE", 
                               COLORS['neon_orange'], (WINDOW_WIDTH // 2, 180), center=True)
        
        self.font_medium.render(self.screen, "CLICK ON A TILE - ESC TO CANCEL", 
                                COLORS['text'], (WINDOW_WIDTH // 2, 230), center=True)
        
        pygame.draw.rect(
            self.screen, COLORS['neon_orange'],
            (BOARD_X - 10, BOARD_Y - 10, BOARD_SIZE + 20, BOARD_SIZE + 20),
            4, border_radius=15
        )
        
        self.render_board()
    
    def render_pause_overlay(self):
        """Оверлей паузы"""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        self.font_title.render(self.screen, "PAUSED", COLORS['neon_cyan'],
                               (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50), center=True)
        
        self.font_medium.render(self.screen, "ESC RESUME - Q QUIT TO MENU", 
                                COLORS['text'], (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50), center=True)
    
    def render_game_over_overlay(self):
        """Оверлей окончания игры"""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        self.font_title.render(self.screen, "GAME OVER", COLORS['neon_red'],
                               (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100), center=True)
        
        self.font_large.render(self.screen, f"FINAL SCORE: {self.game.score}", 
                               COLORS['gold'], (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 20), center=True)
        
        self.font_medium.render(self.screen, f"MAX TILE: {self.game.max_tile}", 
                                COLORS['text'], (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 40), center=True)
        
        if self.game.total_combos > 0:
            self.font_medium.render(self.screen, f"COMBOS: {self.game.total_combos}", 
                                    COLORS['combo'], (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 90), center=True)
        
        self.font_medium.render(self.screen, "PRESS ENTER TO CONTINUE", 
                                COLORS['text_dim'], (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 150), center=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Запуск GUI"""
    gui = Alpha2048GUI()
    gui.run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Терминальная версия игры 2048 с интерактивным управлением
Использует только встроенные модули Python
"""
import sys
import os
import tty
import termios
from typing import Optional, Callable
import time

from game_2048 import Game2048, Direction


# ANSI цвета для терминала
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Цвета фона для плиток
    BG_EMPTY = '\033[48;5;250m'
    BG_2 = '\033[48;5;230m'
    BG_4 = '\033[48;5;223m'
    BG_8 = '\033[48;5;216m'
    BG_16 = '\033[48;5;209m'
    BG_32 = '\033[48;5;208m'
    BG_64 = '\033[48;5;202m'
    BG_128 = '\033[48;5;226m'
    BG_256 = '\033[48;5;220m'
    BG_512 = '\033[48;5;214m'
    BG_1024 = '\033[48;5;208m'
    BG_2048 = '\033[48;5;202m'
    BG_4096 = '\033[48;5;94m'
    
    # Цвета текста
    FG_DARK = '\033[38;5;239m'
    FG_LIGHT = '\033[38;5;255m'
    FG_GREEN = '\033[38;5;46m'
    FG_YELLOW = '\033[38;5;226m'
    FG_BLUE = '\033[38;5;39m'


TILE_COLORS = {
    0: (Colors.BG_EMPTY, Colors.FG_DARK),
    2: (Colors.BG_2, Colors.FG_DARK),
    4: (Colors.BG_4, Colors.FG_DARK),
    8: (Colors.BG_8, Colors.FG_LIGHT),
    16: (Colors.BG_16, Colors.FG_LIGHT),
    32: (Colors.BG_32, Colors.FG_LIGHT),
    64: (Colors.BG_64, Colors.FG_LIGHT),
    128: (Colors.BG_128, Colors.FG_LIGHT),
    256: (Colors.BG_256, Colors.FG_LIGHT),
    512: (Colors.BG_512, Colors.FG_LIGHT),
    1024: (Colors.BG_1024, Colors.FG_LIGHT),
    2048: (Colors.BG_2048, Colors.FG_LIGHT),
    4096: (Colors.BG_4096, Colors.FG_LIGHT),
}


class TerminalGUI:
    """Терминальный интерфейс для игры 2048"""
    
    def __init__(self):
        self.game = Game2048()
        self.running = True
        self.ai_mode = False
        self.ai_callback: Optional[Callable] = None
        
    def clear_screen(self):
        """Очистка экрана"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def get_tile_color(self, value: int):
        """Получить цвета для плитки"""
        if value > 4096:
            return TILE_COLORS[4096]
        return TILE_COLORS.get(value, TILE_COLORS[0])
    
    def draw_tile(self, value: int, width: int = 8) -> str:
        """Нарисовать плитку"""
        bg_color, fg_color = self.get_tile_color(value)
        
        if value == 0:
            text = "·"
        else:
            text = str(value)
        
        # Центрируем текст
        padding = width - len(text)
        left_pad = padding // 2
        right_pad = padding - left_pad
        
        tile = f"{bg_color}{fg_color}{' ' * left_pad}{text}{' ' * right_pad}{Colors.RESET}"
        return tile
    
    def draw_board(self):
        """Отрисовка игрового поля"""
        lines = []
        
        # Заголовок
        lines.append("")
        lines.append(f"{Colors.BOLD}{Colors.FG_BLUE}╔════════════════════════════════════════╗{Colors.RESET}")
        lines.append(f"{Colors.BOLD}{Colors.FG_BLUE}║              2048 GAME                 ║{Colors.RESET}")
        lines.append(f"{Colors.BOLD}{Colors.FG_BLUE}╚════════════════════════════════════════╝{Colors.RESET}")
        lines.append("")
        
        # Счёт
        score_text = f"{Colors.BOLD}Score: {Colors.FG_YELLOW}{self.game.score}{Colors.RESET}"
        best_text = f"{Colors.BOLD}Best: {Colors.FG_GREEN}{self.game.max_tile}{Colors.RESET}"
        moves_text = f"{Colors.BOLD}Moves: {Colors.FG_BLUE}{self.game.moves}{Colors.RESET}"
        
        lines.append(f"  {score_text}  |  {best_text}  |  {moves_text}")
        
        # AI индикатор
        if self.ai_mode:
            lines.append(f"  {Colors.FG_GREEN}🤖 AI Playing...{Colors.RESET}")
        
        lines.append("")
        
        # Игровое поле
        lines.append("  ┌────────┬────────┬────────┬────────┐")
        
        for row in range(4):
            # Строка с плитками
            row_tiles = "  │"
            for col in range(4):
                value = self.game.board[row, col]
                tile = self.draw_tile(value)
                row_tiles += tile + "│"
            lines.append(row_tiles)
            
            # Разделитель
            if row < 3:
                lines.append("  ├────────┼────────┼────────┼────────┤")
        
        lines.append("  └────────┴────────┴────────┴────────┘")
        lines.append("")
        
        # Инструкции
        lines.append(f"  {Colors.BOLD}Controls:{Colors.RESET}")
        lines.append(f"    {Colors.FG_BLUE}↑↓←→{Colors.RESET} Move tiles")
        lines.append(f"    {Colors.FG_YELLOW}r{Colors.RESET}    Restart game")
        lines.append(f"    {Colors.FG_GREEN}a{Colors.RESET}    Toggle AI mode")
        lines.append(f"    {Colors.FG_BLUE}q{Colors.RESET}    Quit")
        lines.append("")
        
        return "\n".join(lines)
    
    def draw(self):
        """Полная перерисовка экрана"""
        self.clear_screen()
        print(self.draw_board())
        
        if self.game.is_game_over():
            print(f"\n  {Colors.BOLD}{Colors.FG_YELLOW}╔════════════════════════════════════════╗{Colors.RESET}")
            print(f"  {Colors.BOLD}{Colors.FG_YELLOW}║            GAME OVER!                  ║{Colors.RESET}")
            print(f"  {Colors.BOLD}{Colors.FG_YELLOW}╚════════════════════════════════════════╝{Colors.RESET}")
            print(f"\n  Final Score: {Colors.FG_GREEN}{self.game.score}{Colors.RESET}")
            print(f"  Best Tile: {Colors.FG_YELLOW}{self.game.max_tile}{Colors.RESET}")
            print(f"\n  Press {Colors.FG_YELLOW}r{Colors.RESET} to restart or {Colors.FG_BLUE}q{Colors.RESET} to quit\n")
    
    def get_key(self):
        """Получить нажатую клавишу (Unix/Mac)"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            
            # Обработка escape-последовательностей для стрелок
            if ch == '\x1b':
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A':
                        return 'up'
                    elif ch3 == 'B':
                        return 'down'
                    elif ch3 == 'C':
                        return 'right'
                    elif ch3 == 'D':
                        return 'left'
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        return ch
    
    def handle_key(self, key: str):
        """Обработка нажатия клавиши"""
        if key in ['q', 'Q']:
            self.running = False
            return
        
        if key in ['r', 'R']:
            self.game.reset()
            self.draw()
            return
        
        if key in ['a', 'A']:
            self.ai_mode = not self.ai_mode
            self.draw()
            return
        
        if self.ai_mode or self.game.is_game_over():
            return
        
        # Обработка движений
        direction_map = {
            'up': Direction.UP,
            'down': Direction.DOWN,
            'left': Direction.LEFT,
            'right': Direction.RIGHT,
            'w': Direction.UP,
            'W': Direction.UP,
            's': Direction.DOWN,
            'S': Direction.DOWN,
            'a': Direction.LEFT,
            'd': Direction.RIGHT,
            'D': Direction.RIGHT,
        }
        
        if key in direction_map:
            direction = direction_map[key]
            self.game.move(direction)
            self.draw()
    
    def ai_step(self):
        """Один шаг AI"""
        if not self.ai_callback or self.game.is_game_over():
            return
        
        valid_moves = self.game.get_valid_moves()
        if valid_moves:
            state = self.game.get_state()
            features = self.game.get_features()
            valid_moves_int = [int(m) for m in valid_moves]
            
            action = self.ai_callback(state, features, valid_moves_int)
            self.game.move(Direction(action))
            self.draw()
            time.sleep(0.1)  # Небольшая задержка для наблюдения
    
    def run(self, ai_callback: Optional[Callable] = None):
        """Основной игровой цикл"""
        self.ai_callback = ai_callback
        
        print("\n" + "="*50)
        print("  Starting 2048 Terminal Edition...")
        print("="*50)
        time.sleep(1)
        
        self.draw()
        
        try:
            while self.running:
                if self.ai_mode and self.ai_callback:
                    self.ai_step()
                    # Проверяем ввод с таймаутом
                    import select
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = self.get_key()
                        self.handle_key(key)
                else:
                    key = self.get_key()
                    self.handle_key(key)
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self.clear_screen()
            print("\n  Thanks for playing 2048!\n")
            print(f"  Final Score: {self.game.score}")
            print(f"  Best Tile: {self.game.max_tile}\n")


def play_terminal():
    """Запуск терминальной игры"""
    gui = TerminalGUI()
    gui.run()


def play_terminal_with_ai(model_path: str = "models/model_best.pt"):
    """Игра с AI в терминале"""
    import torch
    from neural_network import DQNAgent, device
    
    print("Loading AI model...")
    agent = DQNAgent()
    if not agent.load(model_path):
        print(f"Model not found at {model_path}, using untrained model")
    
    agent.policy_net.eval()
    
    def ai_callback(state, features, valid_moves):
        return agent.policy_net.get_action(state, features, valid_moves, epsilon=0.0)
    
    gui = TerminalGUI()
    gui.ai_mode = True
    gui.run(ai_callback=ai_callback)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--ai":
        model_path = sys.argv[2] if len(sys.argv) > 2 else "models/model_best.pt"
        play_terminal_with_ai(model_path)
    else:
        play_terminal()

"""
Графический интерфейс 2048 с оригинальным дизайном
Реализован на Tkinter (встроен в Python, не требует установки)
- Оригинальные цвета 2048
- Отображение статистики
- Режим наблюдения за AI
"""
import tkinter as tk
from tkinter import messagebox
import numpy as np
from typing import Optional, Callable, List
import time

from game_2048 import Game2048, Direction


# Цветовая схема - оригинальный стиль 2048
COLORS = {
    'background': '#faf8ef',
    'board_bg': '#bbada0',
    'empty_cell': '#cdc1b4',
    'text_dark': '#776e65',
    'text_light': '#f9f6f2',
    'score_bg': '#8f7a66',
}

# Цвета плиток
TILE_COLORS = {
    0: '#cdc1b4',
    2: '#eee4da',
    4: '#ede0c8',
    8: '#f2b179',
    16: '#f59563',
    32: '#f67c5f',
    64: '#f65e3b',
    128: '#edcf72',
    256: '#edcc61',
    512: '#edc850',
    1024: '#edc53f',
    2048: '#edc22e',
    4096: '#3c3a32',
    8192: '#3c3a32',
    16384: '#3c3a32',
    32768: '#3c3a32',
    65536: '#3c3a32',
}

# Размер шрифта для разных чисел
FONT_SIZES = {
    2: 48, 4: 48, 8: 48, 16: 44, 32: 44, 64: 44,
    128: 38, 256: 38, 512: 38, 1024: 32, 2048: 32,
    4096: 28, 8192: 28, 16384: 24, 32768: 24, 65536: 22
}


class Game2048GUI:
    """Графический интерфейс игры 2048 на Tkinter"""
    
    def __init__(
        self,
        cell_size: int = 100,
        cell_padding: int = 10,
    ):
        print("Initializing GUI...")
        self.cell_size = cell_size
        self.cell_padding = cell_padding
        self.board_size = 4
        
        # Размеры окна
        self.board_pixels = self.board_size * cell_size + (self.board_size + 1) * cell_padding
        self.window_width = self.board_pixels + 40
        self.window_height = self.board_pixels + 200
        
        # Создаём окно
        print("Creating Tkinter window...")
        self.root = tk.Tk()
        self.root.title("2048 - AI Edition")
        self.root.configure(bg=COLORS['background'])
        self.root.resizable(False, False)
        print(f"Window size: {self.window_width}x{self.window_height}")
        
        # Центрируем окно
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - self.window_width) // 2
        y = (screen_height - self.window_height) // 2
        self.root.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")
        
        # Игра
        self.game: Optional[Game2048] = None
        self.ai_mode = False
        self.ai_callback: Optional[Callable] = None
        self.ai_speed = 100  # мс между ходами AI
        self.running = True
        
        # Создаём интерфейс
        self._create_widgets()
        self._bind_keys()
    
    def _create_widgets(self):
        """Создание виджетов"""
        print("Creating widgets...")
        # Заголовок
        header_frame = tk.Frame(self.root, bg=COLORS['background'])
        header_frame.pack(pady=10)
        
        # Название
        title_label = tk.Label(
            header_frame, 
            text="2048", 
            font=("Helvetica", 48, "bold"),
            fg=COLORS['text_dark'],
            bg=COLORS['background']
        )
        title_label.pack(side=tk.LEFT, padx=20)
        
        # Счёт
        score_frame = tk.Frame(header_frame, bg=COLORS['score_bg'], padx=15, pady=5)
        score_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            score_frame, text="SCORE", 
            font=("Helvetica", 12, "bold"),
            fg="#eee4da", bg=COLORS['score_bg']
        ).pack()
        
        self.score_label = tk.Label(
            score_frame, text="0",
            font=("Helvetica", 24, "bold"),
            fg=COLORS['text_light'], bg=COLORS['score_bg']
        )
        self.score_label.pack()
        
        # Лучший тайл
        best_frame = tk.Frame(header_frame, bg=COLORS['score_bg'], padx=15, pady=5)
        best_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            best_frame, text="BEST TILE",
            font=("Helvetica", 12, "bold"),
            fg="#eee4da", bg=COLORS['score_bg']
        ).pack()
        
        self.best_label = tk.Label(
            best_frame, text="0",
            font=("Helvetica", 24, "bold"),
            fg=COLORS['text_light'], bg=COLORS['score_bg']
        )
        self.best_label.pack()
        
        # AI индикатор
        self.ai_label = tk.Label(
            self.root, text="",
            font=("Helvetica", 14, "bold"),
            fg="#2ecc71", bg=COLORS['background']
        )
        self.ai_label.pack()
        
        # Игровое поле (Canvas)
        canvas_size = self.board_pixels
        self.canvas = tk.Canvas(
            self.root,
            width=canvas_size,
            height=canvas_size,
            bg=COLORS['board_bg'],
            highlightthickness=0
        )
        self.canvas.pack(pady=10)
        
        # Статистика
        self.stats_label = tk.Label(
            self.root, text="Moves: 0 | Empty: 14",
            font=("Helvetica", 14),
            fg=COLORS['text_dark'], bg=COLORS['background']
        )
        self.stats_label.pack()
        
        # Инструкции
        instructions = tk.Label(
            self.root,
            text="Arrow Keys: Move | R: Restart | A: Toggle AI | Q: Quit",
            font=("Helvetica", 12),
            fg=COLORS['text_dark'], bg=COLORS['background']
        )
        instructions.pack(pady=10)
        
        # Ячейки на canvas
        self.cells = []
        self.cell_texts = []
        
        for row in range(self.board_size):
            row_cells = []
            row_texts = []
            for col in range(self.board_size):
                x1 = self.cell_padding + col * (self.cell_size + self.cell_padding)
                y1 = self.cell_padding + row * (self.cell_size + self.cell_padding)
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                # Ячейка (скруглённый прямоугольник)
                cell = self._create_rounded_rect(x1, y1, x2, y2, 6, COLORS['empty_cell'])
                row_cells.append(cell)
                
                # Текст
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                text = self.canvas.create_text(cx, cy, text="", font=("Helvetica", 48, "bold"))
                row_texts.append(text)
            
            self.cells.append(row_cells)
            self.cell_texts.append(row_texts)
    
    def _create_rounded_rect(self, x1, y1, x2, y2, radius, color):
        """Создание скруглённого прямоугольника"""
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        return self.canvas.create_polygon(points, fill=color, smooth=True, outline="")
    
    def _bind_keys(self):
        """Привязка клавиш"""
        self.root.bind("<Up>", lambda e: self._handle_key(Direction.UP))
        self.root.bind("<Down>", lambda e: self._handle_key(Direction.DOWN))
        self.root.bind("<Left>", lambda e: self._handle_key(Direction.LEFT))
        self.root.bind("<Right>", lambda e: self._handle_key(Direction.RIGHT))
        self.root.bind("<r>", lambda e: self._restart())
        self.root.bind("<R>", lambda e: self._restart())
        self.root.bind("<a>", lambda e: self._toggle_ai())
        self.root.bind("<A>", lambda e: self._toggle_ai())
        self.root.bind("<q>", lambda e: self._quit())
        self.root.bind("<Q>", lambda e: self._quit())
        self.root.bind("<Escape>", lambda e: self._quit())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
    
    def _handle_key(self, direction: Direction):
        """Обработка нажатия клавиши"""
        if self.ai_mode or self.game is None:
            return
        
        if self.game.is_game_over():
            return
        
        self._make_move(direction)
    
    def _make_move(self, direction: Direction):
        """Выполнение хода"""
        reward, done, info = self.game.move(direction)
        self._update_display()
        
        if done:
            self._show_game_over()
    
    def _restart(self):
        """Перезапуск игры"""
        if self.game:
            self.game.reset()
            self._update_display()
    
    def _toggle_ai(self):
        """Переключение режима AI"""
        self.ai_mode = not self.ai_mode
        self.ai_label.config(text="🤖 AI Playing" if self.ai_mode else "")
        
        if self.ai_mode:
            self._ai_step()
    
    def _quit(self):
        """Выход из игры"""
        self.running = False
        self.root.quit()
        self.root.destroy()
    
    def _ai_step(self):
        """Один шаг AI"""
        if not self.ai_mode or not self.running:
            return
        
        if self.game is None or self.game.is_game_over():
            return
        
        if self.ai_callback is None:
            return
        
        valid_moves = self.game.get_valid_moves()
        if valid_moves:
            state = self.game.get_state()
            features = self.game.get_features()
            valid_moves_int = [int(m) for m in valid_moves]
            
            action = self.ai_callback(state, features, valid_moves_int)
            self._make_move(Direction(action))
        
        # Следующий шаг
        if self.ai_mode and not self.game.is_game_over():
            self.root.after(self.ai_speed, self._ai_step)
    
    def _update_display(self):
        """Обновление отображения"""
        if self.game is None:
            return
        
        # Обновляем ячейки
        for row in range(self.board_size):
            for col in range(self.board_size):
                value = self.game.board[row, col]
                
                # Цвет ячейки
                color = TILE_COLORS.get(value, TILE_COLORS[65536])
                self.canvas.itemconfig(self.cells[row][col], fill=color)
                
                # Текст
                if value > 0:
                    text_color = COLORS['text_light'] if value >= 8 else COLORS['text_dark']
                    font_size = FONT_SIZES.get(value, 22)
                    self.canvas.itemconfig(
                        self.cell_texts[row][col],
                        text=str(value),
                        fill=text_color,
                        font=("Helvetica", font_size, "bold")
                    )
                else:
                    self.canvas.itemconfig(self.cell_texts[row][col], text="")
        
        # Счёт
        self.score_label.config(text=str(self.game.score))
        self.best_label.config(text=str(self.game.max_tile))
        
        # Статистика
        empty = np.sum(self.game.board == 0)
        self.stats_label.config(text=f"Moves: {self.game.moves} | Empty: {empty}")
    
    def _show_game_over(self):
        """Показать сообщение о конце игры"""
        if not self.running:
            return
        messagebox.showinfo(
            "Game Over!",
            f"Final Score: {self.game.score}\nBest Tile: {self.game.max_tile}\n\nPress R to restart"
        )
    
    def run(self, game: Optional[Game2048] = None, ai_callback: Optional[Callable] = None):
        """Запуск игры"""
        print("Starting game loop...")
        self.game = game if game else Game2048()
        self.ai_callback = ai_callback
        
        self._update_display()
        print("Display updated, entering mainloop...")
        print("Window should be visible now!")
        
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Error in mainloop: {e}")
            import traceback
            traceback.print_exc()


def play_manual():
    """Ручная игра"""
    gui = Game2048GUI()
    gui.run()


def play_with_ai(model_path: str = "models/model_best.pt"):
    """Игра с AI"""
    import torch
    from neural_network import DQNAgent, device
    
    # Загружаем модель
    agent = DQNAgent()
    if not agent.load(model_path):
        print(f"Model not found at {model_path}, using untrained model")
    
    agent.policy_net.eval()
    
    def ai_callback(state, features, valid_moves):
        return agent.policy_net.get_action(state, features, valid_moves, epsilon=0.0)
    
    gui = Game2048GUI()
    gui.ai_mode = True
    gui.ai_label.config(text="🤖 AI Playing")
    gui.run(ai_callback=ai_callback)
    gui.root.after(100, gui._ai_step)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--ai":
        model_path = sys.argv[2] if len(sys.argv) > 2 else "models/model_best.pt"
        play_with_ai(model_path)
    else:
        play_manual()

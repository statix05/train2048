#!/usr/bin/env python3
"""
GUI для визуализации процесса обучения нейросети
Показывает:
- Игровое поле в реальном времени
- Графики прогресса (score, max tile, loss)
- Статистику обучения
- Управление процессом обучения
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
import numpy as np
from typing import Optional, List, Tuple
import os

from game_2048 import Game2048, Direction
from neural_network import DQNAgent, device
from trainer import Trainer


class TrainingStatsPanel(tk.Frame):
    """Панель со статистикой обучения"""
    
    def __init__(self, parent):
        super().__init__(parent, bg='#faf8ef')
        
        # Текущий эпизод
        self.episode_label = tk.Label(self, text="Episode: 0 / 0", 
                                      font=("Arial", 14, "bold"),
                                      bg='#faf8ef', fg='#776e65')
        self.episode_label.pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self, length=300, mode='determinate')
        self.progress.pack(pady=5)
        
        # Статистика
        stats_frame = tk.Frame(self, bg='#faf8ef')
        stats_frame.pack(pady=10)
        
        # Левая колонка
        left_frame = tk.Frame(stats_frame, bg='#faf8ef')
        left_frame.pack(side=tk.LEFT, padx=20)
        
        self.score_label = self._create_stat_label(left_frame, "Avg Score", "0")
        self.tile_label = self._create_stat_label(left_frame, "Avg Max Tile", "0")
        self.moves_label = self._create_stat_label(left_frame, "Avg Moves", "0")
        
        # Правая колонка
        right_frame = tk.Frame(stats_frame, bg='#faf8ef')
        right_frame.pack(side=tk.LEFT, padx=20)
        
        self.best_score_label = self._create_stat_label(right_frame, "Best Score", "0")
        self.best_tile_label = self._create_stat_label(right_frame, "Best Tile", "0")
        self.loss_label = self._create_stat_label(right_frame, "Loss", "0.000")
        
        # Epsilon
        self.epsilon_label = tk.Label(self, text="Epsilon: 1.000", 
                                      font=("Arial", 12),
                                      bg='#faf8ef', fg='#776e65')
        self.epsilon_label.pack(pady=5)
        
        # Скорость обучения
        speed_frame = tk.Frame(self, bg='#faf8ef')
        speed_frame.pack(pady=5)
        
        tk.Label(speed_frame, text="Speed:", font=("Arial", 10),
                bg='#faf8ef', fg='#776e65').pack(side=tk.LEFT)
        
        self.speed_var = tk.StringVar(value="normal")
        speeds = [("Fast", "fast"), ("Normal", "normal"), ("Slow", "slow")]
        for text, value in speeds:
            tk.Radiobutton(speed_frame, text=text, variable=self.speed_var,
                          value=value, bg='#faf8ef', font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
    
    def _create_stat_label(self, parent, title, value):
        frame = tk.Frame(parent, bg='#8f7a66', padx=10, pady=5)
        frame.pack(pady=3, fill=tk.X)
        
        tk.Label(frame, text=title, font=("Arial", 10),
                bg='#8f7a66', fg='#f9f6f2').pack()
        label = tk.Label(frame, text=value, font=("Arial", 14, "bold"),
                        bg='#8f7a66', fg='#f9f6f2')
        label.pack()
        return label
    
    def update_stats(self, stats: dict):
        """Обновление статистики"""
        self.episode_label.config(text=f"Episode: {stats['episode']} / {stats['total_episodes']}")
        self.progress['value'] = (stats['episode'] / stats['total_episodes']) * 100
        
        self.score_label.config(text=f"{stats['avg_score']:.0f}")
        self.tile_label.config(text=f"{stats['avg_max_tile']:.0f}")
        self.moves_label.config(text=f"{stats['avg_moves']:.0f}")
        
        self.best_score_label.config(text=f"{stats['best_score']}")
        self.best_tile_label.config(text=f"{stats['best_max_tile']}")
        self.loss_label.config(text=f"{stats['loss']:.3f}")
        
        self.epsilon_label.config(text=f"Epsilon: {stats['epsilon']:.3f}")


class MiniGameBoard(tk.Canvas):
    """Мини игровое поле для отображения текущей игры"""
    
    TILE_COLORS = {
        0: '#cdc1b4', 2: '#eee4da', 4: '#ede0c8', 8: '#f2b179',
        16: '#f59563', 32: '#f67c5f', 64: '#f65e3b', 128: '#edcf72',
        256: '#edcc61', 512: '#edc850', 1024: '#edc53f', 2048: '#edc22e',
        4096: '#3c3a32', 8192: '#3c3a32',
    }
    
    def __init__(self, parent, cell_size=50):
        self.cell_size = cell_size
        self.padding = 5
        board_size = 4 * cell_size + 5 * self.padding
        
        super().__init__(parent, width=board_size, height=board_size,
                        bg='#bbada0', highlightthickness=0)
        
        self.cells = []
        self.texts = []
        
        for i in range(4):
            row_cells = []
            row_texts = []
            for j in range(4):
                x = self.padding + j * (cell_size + self.padding)
                y = self.padding + i * (cell_size + self.padding)
                
                cell = self.create_rectangle(x, y, x + cell_size, y + cell_size,
                                            fill='#cdc1b4', outline='')
                row_cells.append(cell)
                
                text = self.create_text(x + cell_size // 2, y + cell_size // 2,
                                       text='', font=('Arial', 20, 'bold'))
                row_texts.append(text)
            
            self.cells.append(row_cells)
            self.texts.append(row_texts)
    
    def update_board(self, board: np.ndarray):
        """Обновление отображения доски"""
        for i in range(4):
            for j in range(4):
                value = board[i, j]
                color = self.TILE_COLORS.get(value, '#3c3a32')
                text_color = '#776e65' if value <= 4 else '#f9f6f2'
                
                self.itemconfig(self.cells[i][j], fill=color)
                self.itemconfig(self.texts[i][j], 
                              text=str(value) if value > 0 else '',
                              fill=text_color)


class SimpleGraph(tk.Canvas):
    """Простой график для отображения метрик"""
    
    def __init__(self, parent, width=300, height=100, title=""):
        super().__init__(parent, width=width, height=height,
                        bg='white', highlightthickness=1, highlightbackground='#bbada0')
        
        self.width = width
        self.height = height
        self.title = title
        self.data = []
        self.max_points = 100
        
        # Заголовок
        self.create_text(width // 2, 15, text=title, font=('Arial', 10, 'bold'))
    
    def add_point(self, value):
        """Добавить точку данных"""
        self.data.append(value)
        if len(self.data) > self.max_points:
            self.data.pop(0)
        self.redraw()
    
    def redraw(self):
        """Перерисовать график"""
        self.delete('line')
        
        if len(self.data) < 2:
            return
        
        # Нормализация данных
        min_val = min(self.data)
        max_val = max(self.data)
        range_val = max_val - min_val if max_val > min_val else 1
        
        # Отступы
        margin = 30
        graph_height = self.height - 2 * margin
        graph_width = self.width - 2 * margin
        
        # Рисуем линию
        points = []
        for i, val in enumerate(self.data):
            x = margin + (i / (len(self.data) - 1)) * graph_width
            normalized = (val - min_val) / range_val
            y = self.height - margin - normalized * graph_height
            points.extend([x, y])
        
        if len(points) >= 4:
            self.create_line(points, fill='#3498db', width=2, tags='line', smooth=True)
        
        # Показываем последнее значение
        last_val = self.data[-1]
        self.create_text(self.width - 10, margin,
                        text=f"{last_val:.0f}" if abs(last_val) > 1 else f"{last_val:.3f}",
                        font=('Arial', 9), anchor='ne')


class TrainingGUI:
    """Главное окно GUI для обучения"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("2048 AI Training")
        self.root.configure(bg='#faf8ef')
        
        # Центрирование окна
        window_width = 900
        window_height = 700
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Состояние
        self.training = False
        self.paused = False
        self.agent: Optional[DQNAgent] = None
        self.trainer: Optional[Trainer] = None
        self.update_queue = queue.Queue()
        
        # Создание интерфейса
        self._create_widgets()
        
        # Таймер обновления
        self.root.after(100, self._check_updates)
    
    def _create_widgets(self):
        """Создание всех виджетов"""
        # Заголовок
        header = tk.Label(self.root, text="🧠 2048 Neural Network Training",
                         font=("Arial", 20, "bold"),
                         bg='#faf8ef', fg='#776e65')
        header.pack(pady=10)
        
        # Главный контейнер
        main_frame = tk.Frame(self.root, bg='#faf8ef')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Левая панель - игровое поле и управление
        left_panel = tk.Frame(main_frame, bg='#faf8ef')
        left_panel.pack(side=tk.LEFT, padx=10)
        
        tk.Label(left_panel, text="Current Game", font=("Arial", 12, "bold"),
                bg='#faf8ef', fg='#776e65').pack(pady=5)
        
        self.game_board = MiniGameBoard(left_panel)
        self.game_board.pack(pady=10)
        
        # Текущая статистика игры
        game_stats_frame = tk.Frame(left_panel, bg='#8f7a66', padx=10, pady=5)
        game_stats_frame.pack(pady=10, fill=tk.X)
        
        self.current_score = tk.Label(game_stats_frame, text="Score: 0",
                                      font=("Arial", 12), bg='#8f7a66', fg='#f9f6f2')
        self.current_score.pack()
        
        self.current_tile = tk.Label(game_stats_frame, text="Max Tile: 0",
                                     font=("Arial", 12), bg='#8f7a66', fg='#f9f6f2')
        self.current_tile.pack()
        
        # Кнопки управления
        control_frame = tk.Frame(left_panel, bg='#faf8ef')
        control_frame.pack(pady=10)
        
        self.start_button = tk.Button(control_frame, text="▶ Start Training",
                                      command=self._start_training,
                                      font=("Arial", 12, "bold"),
                                      bg='#8f7a66', fg='#f9f6f2',
                                      padx=20, pady=10)
        self.start_button.pack(pady=5)
        
        self.pause_button = tk.Button(control_frame, text="⏸ Pause",
                                      command=self._toggle_pause,
                                      font=("Arial", 12),
                                      bg='#bbada0', fg='#f9f6f2',
                                      padx=20, pady=10, state=tk.DISABLED)
        self.pause_button.pack(pady=5)
        
        self.stop_button = tk.Button(control_frame, text="⏹ Stop",
                                     command=self._stop_training,
                                     font=("Arial", 12),
                                     bg='#bbada0', fg='#f9f6f2',
                                     padx=20, pady=10, state=tk.DISABLED)
        self.stop_button.pack(pady=5)
        
        # Правая панель - статистика и графики
        right_panel = tk.Frame(main_frame, bg='#faf8ef')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Статистика
        self.stats_panel = TrainingStatsPanel(right_panel)
        self.stats_panel.pack(pady=10)
        
        # Графики
        graphs_frame = tk.Frame(right_panel, bg='#faf8ef')
        graphs_frame.pack(pady=10)
        
        self.score_graph = SimpleGraph(graphs_frame, width=250, height=80, title="Score")
        self.score_graph.pack(pady=5)
        
        self.tile_graph = SimpleGraph(graphs_frame, width=250, height=80, title="Max Tile")
        self.tile_graph.pack(pady=5)
        
        self.loss_graph = SimpleGraph(graphs_frame, width=250, height=80, title="Loss")
        self.loss_graph.pack(pady=5)
    
    def _start_training(self):
        """Начать обучение"""
        # Диалог параметров
        dialog = TrainingConfigDialog(self.root)
        self.root.wait_window(dialog.top)
        
        if not dialog.result:
            return
        
        config = dialog.result
        
        # Создать агента
        print(f"Creating agent on device: {device}")
        self.agent = DQNAgent(
            learning_rate=config['lr'],
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size'],
            epsilon_decay=config['episodes'] * 5
        )
        
        # Загрузить существующую модель если нужно
        if config['resume'] and os.path.exists("models/model_best.pt"):
            self.agent.load("models/model_best.pt")
        
        self.trainer = Trainer(self.agent)
        
        # Запустить обучение в отдельном потоке
        self.training = True
        self.paused = False
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        
        thread = threading.Thread(target=self._training_loop,
                                 args=(config['episodes'],), daemon=True)
        thread.start()
    
    def _training_loop(self, n_episodes: int):
        """Цикл обучения в отдельном потоке"""
        from collections import deque
        
        score_window = deque(maxlen=10)
        tile_window = deque(maxlen=10)
        moves_window = deque(maxlen=10)
        
        for episode in range(1, n_episodes + 1):
            if not self.training:
                break
            
            while self.paused:
                time.sleep(0.1)
                if not self.training:
                    break
            
            # Один эпизод
            game = Game2048()
            result = self.trainer.train_episode(game)
            
            # Обновляем окна
            score_window.append(result['score'])
            tile_window.append(result['max_tile'])
            moves_window.append(result['moves'])
            
            # Обновляем лучшие результаты
            if result['score'] > self.trainer.best_score:
                self.trainer.best_score = result['score']
            if result['max_tile'] > self.trainer.best_max_tile:
                self.trainer.best_max_tile = result['max_tile']
            
            # Получить скорость
            speed = self.stats_panel.speed_var.get()
            delay = {'fast': 0.001, 'normal': 0.05, 'slow': 0.2}.get(speed, 0.05)
            
            # Отправить обновление в GUI каждые N эпизодов или с задержкой
            if episode % 1 == 0:
                self.update_queue.put({
                    'type': 'game_update',
                    'board': game.board.copy(),
                    'score': game.score,
                    'max_tile': game.max_tile
                })
                
                time.sleep(delay)
            
            if episode % 10 == 0:
                self.update_queue.put({
                    'type': 'stats_update',
                    'episode': episode,
                    'total_episodes': n_episodes,
                    'avg_score': np.mean(score_window),
                    'avg_max_tile': np.mean(tile_window),
                    'avg_moves': np.mean(moves_window),
                    'best_score': self.trainer.best_score,
                    'best_max_tile': self.trainer.best_max_tile,
                    'loss': result['loss'],
                    'epsilon': self.agent.get_epsilon()
                })
            
            # Сохранение каждые 100 эпизодов
            if episode % 100 == 0:
                self.trainer.save_checkpoint(episode)
        
        # Финальное сохранение
        if self.training:
            self.trainer.save_checkpoint(n_episodes, final=True)
            self.update_queue.put({'type': 'training_complete'})
    
    def _check_updates(self):
        """Проверка обновлений из очереди"""
        try:
            while True:
                update = self.update_queue.get_nowait()
                
                if update['type'] == 'game_update':
                    self.game_board.update_board(update['board'])
                    self.current_score.config(text=f"Score: {update['score']}")
                    self.current_tile.config(text=f"Max Tile: {update['max_tile']}")
                
                elif update['type'] == 'stats_update':
                    self.stats_panel.update_stats(update)
                    self.score_graph.add_point(update['avg_score'])
                    self.tile_graph.add_point(update['avg_max_tile'])
                    if update['loss'] > 0:
                        self.loss_graph.add_point(update['loss'])
                
                elif update['type'] == 'training_complete':
                    messagebox.showinfo("Training Complete", 
                                       "Training has finished!\nModel saved to models/")
                    self._reset_buttons()
        
        except queue.Empty:
            pass
        
        self.root.after(50, self._check_updates)
    
    def _toggle_pause(self):
        """Пауза/продолжить"""
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(text="▶ Resume")
        else:
            self.pause_button.config(text="⏸ Pause")
    
    def _stop_training(self):
        """Остановить обучение"""
        if messagebox.askyesno("Stop Training", "Are you sure you want to stop training?"):
            self.training = False
            self._reset_buttons()
    
    def _reset_buttons(self):
        """Сбросить состояние кнопок"""
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED, text="⏸ Pause")
        self.stop_button.config(state=tk.DISABLED)
        self.paused = False
    
    def run(self):
        """Запустить GUI"""
        self.root.mainloop()


class TrainingConfigDialog:
    """Диалог настройки параметров обучения"""
    
    def __init__(self, parent):
        self.result = None
        
        self.top = tk.Toplevel(parent)
        self.top.title("Training Configuration")
        self.top.configure(bg='#faf8ef')
        self.top.transient(parent)
        self.top.grab_set()
        
        # Центрирование
        width, height = 400, 350
        x = parent.winfo_x() + (parent.winfo_width() - width) // 2
        y = parent.winfo_y() + (parent.winfo_height() - height) // 2
        self.top.geometry(f"{width}x{height}+{x}+{y}")
        
        tk.Label(self.top, text="Training Parameters",
                font=("Arial", 14, "bold"),
                bg='#faf8ef', fg='#776e65').pack(pady=10)
        
        # Параметры
        params_frame = tk.Frame(self.top, bg='#faf8ef')
        params_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.entries = {}
        
        params = [
            ("Episodes:", "episodes", "1000"),
            ("Learning Rate:", "lr", "0.0001"),
            ("Batch Size:", "batch_size", "64"),
            ("Buffer Size:", "buffer_size", "50000"),
        ]
        
        for i, (label, key, default) in enumerate(params):
            tk.Label(params_frame, text=label, font=("Arial", 11),
                    bg='#faf8ef', fg='#776e65').grid(row=i, column=0, sticky='w', pady=5)
            
            entry = tk.Entry(params_frame, font=("Arial", 11), width=15)
            entry.insert(0, default)
            entry.grid(row=i, column=1, pady=5, padx=10)
            self.entries[key] = entry
        
        # Чекбокс для resume
        self.resume_var = tk.BooleanVar(value=False)
        tk.Checkbutton(params_frame, text="Resume from existing model",
                      variable=self.resume_var, font=("Arial", 10),
                      bg='#faf8ef').grid(row=len(params), column=0, columnspan=2, pady=10)
        
        # Кнопки
        button_frame = tk.Frame(self.top, bg='#faf8ef')
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Start", command=self._ok,
                 font=("Arial", 12, "bold"),
                 bg='#8f7a66', fg='#f9f6f2', padx=30, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Cancel", command=self.top.destroy,
                 font=("Arial", 12),
                 bg='#bbada0', fg='#f9f6f2', padx=30, pady=5).pack(side=tk.LEFT, padx=5)
    
    def _ok(self):
        """Подтвердить параметры"""
        try:
            self.result = {
                'episodes': int(self.entries['episodes'].get()),
                'lr': float(self.entries['lr'].get()),
                'batch_size': int(self.entries['batch_size'].get()),
                'buffer_size': int(self.entries['buffer_size'].get()),
                'resume': self.resume_var.get()
            }
            self.top.destroy()
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid numbers: {e}")


def main():
    """Запуск GUI обучения"""
    print(f"Starting Training GUI on device: {device}")
    gui = TrainingGUI()
    gui.run()


if __name__ == "__main__":
    main()

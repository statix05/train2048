#!/usr/bin/env python3
"""
Терминальная версия Training GUI
Показывает прогресс обучения в консоли с обновлением в реальном времени
"""
import os
import sys
import time
import numpy as np
from collections import deque
from typing import List, Tuple

from game_2048 import Game2048, Direction
from neural_network import DQNAgent, device
from trainer import Trainer


class TerminalTrainingDisplay:
    """Отображение обучения в терминале"""
    
    def __init__(self):
        self.score_history = deque(maxlen=50)
        self.tile_history = deque(maxlen=50)
        self.loss_history = deque(maxlen=50)
    
    def clear_screen(self):
        """Очистка экрана"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def draw_progress_bar(self, progress: float, width: int = 40) -> str:
        """Рисование прогресс-бара"""
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {progress*100:.1f}%"
    
    def draw_mini_graph(self, data: List[float], width: int = 50, height: int = 8) -> List[str]:
        """Рисование мини-графика"""
        if len(data) < 2:
            return [" " * width] * height
        
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val > min_val else 1
        
        lines = []
        for h in range(height):
            line = ""
            threshold = 1.0 - (h / height)
            
            for i, val in enumerate(data[-width:]):
                normalized = (val - min_val) / range_val
                if normalized >= threshold:
                    line += "█"
                elif normalized >= threshold - (1.0 / height):
                    line += "▓"
                else:
                    line += " "
            
            lines.append(line)
        
        return lines
    
    def draw_board(self, board: np.ndarray) -> str:
        """Рисование игрового поля"""
        lines = []
        lines.append("┌────┬────┬────┬────┐")
        
        for i in range(4):
            row = "│"
            for j in range(4):
                val = board[i, j]
                if val == 0:
                    row += " ·  │"
                elif val < 1000:
                    row += f"{val:^4}│"
                else:
                    row += f"{val:4}│"
            lines.append(row)
            
            if i < 3:
                lines.append("├────┼────┼────┼────┤")
        
        lines.append("└────┴────┴────┴────┘")
        return "\n".join(lines)
    
    def display(self, episode: int, total_episodes: int, 
                game: Game2048, stats: dict):
        """Полное обновление дисплея"""
        self.clear_screen()
        
        # Заголовок
        print("=" * 70)
        print(" " * 20 + "🧠 2048 AI TRAINING")
        print("=" * 70)
        print()
        
        # Прогресс
        progress = episode / total_episodes
        print(f"Episode: {episode:,} / {total_episodes:,}")
        print(self.draw_progress_bar(progress))
        print()
        
        # Две колонки: игровое поле и статистика
        board_str = self.draw_board(game.board)
        board_lines = board_str.split('\n')
        
        stats_lines = [
            "╔══════════════════════════╗",
            "║    CURRENT GAME          ║",
            "╠══════════════════════════╣",
            f"║ Score:      {game.score:>12} ║",
            f"║ Max Tile:   {game.max_tile:>12} ║",
            f"║ Moves:      {game.moves:>12} ║",
            "╠══════════════════════════╣",
            "║    STATISTICS            ║",
            "╠══════════════════════════╣",
            f"║ Avg Score:  {stats['avg_score']:>12.0f} ║",
            f"║ Avg Tile:   {stats['avg_max_tile']:>12.0f} ║",
            f"║ Best Score: {stats['best_score']:>12} ║",
            f"║ Best Tile:  {stats['best_max_tile']:>12} ║",
            "╠══════════════════════════╣",
            "║    TRAINING              ║",
            "╠══════════════════════════╣",
            f"║ Loss:       {stats['loss']:>12.4f} ║",
            f"║ Epsilon:    {stats['epsilon']:>12.3f} ║",
            "╚══════════════════════════╝",
        ]
        
        # Выводим параллельно
        max_lines = max(len(board_lines), len(stats_lines))
        for i in range(max_lines):
            board_line = board_lines[i] if i < len(board_lines) else " " * 25
            # Pad board_line to fixed width to ensure right column is aligned
            # The board uses box drawing chars which might mess up len() if not careful,
            # but here they are 1-char width. 
            # The board width is 4 cells * 5 chars/cell + 1 char (start) + 1 char (end) = 22?
            # Let's check draw_board: 
            # "┌────┬────┬────┬────┐" -> len is 21
            # "│ 16 │ 8  │ 16 │ 2  │" -> len is 21
            # So we pad to 25
            
            # Clean padding using ljust with exact known width of board string (21)
            # Use a slightly larger padding to separate columns
            padding_len = 26 
            
            # Calculate visible length (len() works for these chars)
            current_len = len(board_line)
            padding = " " * (padding_len - current_len)
            
            stats_line = stats_lines[i] if i < len(stats_lines) else ""
            print(f"  {board_line}{padding}{stats_line}")
        
        print()
        
        # Графики
        if len(self.score_history) > 1:
            print("╔" + "═" * 68 + "╗")
            print("║" + " " * 25 + "SCORE HISTORY" + " " * 30 + "║")
            print("╠" + "═" * 68 + "╣")
            
            graph_lines = self.draw_mini_graph(list(self.score_history), width=66)
            for line in graph_lines:
                print(f"║ {line} ║")
            
            if len(self.score_history) >= 2:
                min_score = min(self.score_history)
                max_score = max(self.score_history)
                print(f"║  Min: {min_score:>6.0f}  Max: {max_score:>6.0f}  Current: {stats['avg_score']:>6.0f}" + " " * 24 + "║")
            
            print("╚" + "═" * 68 + "╝")
        
        print()
        print("Press Ctrl+C to stop training")
        print()


def train_terminal(n_episodes: int = 1000, 
                   learning_rate: float = 1e-4,
                   batch_size: int = 64,
                   buffer_size: int = 50000,
                   model_type: str = 'dueling',
                   game_mode: str = 'classic'):
    """Обучение с терминальным интерфейсом"""
    
    print(f"Starting training on device: {device}")
    print(f"Episodes: {n_episodes}")
    print(f"Model Type: {model_type}")
    print(f"Game Mode: {game_mode}")
    print()
    
    # Создание агента
    agent = DQNAgent(
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        epsilon_decay=n_episodes * 5,
        model_type=model_type
    )
    
    trainer = Trainer(agent)
    display = TerminalTrainingDisplay()
    
    # Окна для статистики
    score_window = deque(maxlen=10)
    tile_window = deque(maxlen=10)
    moves_window = deque(maxlen=10)
    
    try:
        for episode in range(1, n_episodes + 1):
            # Один эпизод
            game = Game2048(mode=game_mode)
            result = trainer.train_episode(game)
            
            # Обновляем окна
            score_window.append(result['score'])
            tile_window.append(result['max_tile'])
            moves_window.append(result['moves'])
            
            # Обновляем лучшие результаты
            if result['score'] > trainer.best_score:
                trainer.best_score = result['score']
            if result['max_tile'] > trainer.best_max_tile:
                trainer.best_max_tile = result['max_tile']
            
            # Добавляем в историю для графика
            display.score_history.append(np.mean(score_window))
            display.tile_history.append(np.mean(tile_window))
            if result['loss'] > 0:
                display.loss_history.append(result['loss'])
            
            # Обновляем дисплей
            if episode % 5 == 0 or episode == 1:
                stats = {
                    'avg_score': np.mean(score_window),
                    'avg_max_tile': np.mean(tile_window),
                    'avg_moves': np.mean(moves_window),
                    'best_score': trainer.best_score,
                    'best_max_tile': trainer.best_max_tile,
                    'loss': result['loss'],
                    'epsilon': agent.get_epsilon()
                }
                
                display.display(episode, n_episodes, game, stats)
                time.sleep(0.05)  # Небольшая задержка чтобы видеть обновления
            
            # Сохранение каждые 100 эпизодов
            if episode % 100 == 0:
                trainer.save_checkpoint(episode)
                print(f"\n💾 Model saved at episode {episode}")
                time.sleep(1)
        
        # Финальное сохранение
        trainer.save_checkpoint(n_episodes, final=True)
        
        display.clear_screen()
        print("=" * 70)
        print(" " * 25 + "🎉 TRAINING COMPLETE!")
        print("=" * 70)
        print()
        print(f"Total Episodes: {n_episodes:,}")
        print(f"Best Score: {trainer.best_score:,}")
        print(f"Best Max Tile: {trainer.best_max_tile}")
        print()
        print("Model saved to: models/model_final.pt")
        print("Best model saved to: models/model_best.pt")
        print()
        print("Try your trained model:")
        print("  python main.py play --ai")
        print("  python gui_terminal.py --ai")
        print()
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
        trainer.save_checkpoint(episode)
        print(f"Progress saved at episode {episode}")


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Terminal Training Interface")
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=50000, help='Buffer size')
    parser.add_argument('--model', type=str, default='dueling', choices=['simple', 'conv', 'dueling', 'hybrid'], help='Neural Network Architecture')
    
    args = parser.parse_args()
    
    train_terminal(
        n_episodes=args.episodes,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        model_type=args.model
    )


if __name__ == "__main__":
    main()

"""
Alpha2048 Trainer
=================

Система обучения Alpha2048 с:
- Self-play для сбора данных
- Curriculum learning
- Визуализация прогресса
- Сохранение лучших моделей

Оптимизировано для Apple Silicon (MPS).
"""

import numpy as np
import time
import os
import json
from typing import Optional, Dict, List
from collections import deque
from datetime import datetime

from game_2048 import Game2048
from alpha2048 import (
    Alpha2048Agent, 
    CurriculumManager,
    get_device_info,
    DEVICE
)


class Alpha2048Trainer:
    """
    Тренер для Alpha2048 с self-play и curriculum learning.
    """
    
    def __init__(
        self,
        agent: Alpha2048Agent,
        save_dir: str = "models",
        log_dir: str = "logs"
    ):
        self.agent = agent
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Метрики
        self.game_scores: List[int] = []
        self.game_tiles: List[int] = []
        self.game_moves: List[int] = []
        self.training_losses: List[float] = []
        
        # Moving averages
        self.score_window = deque(maxlen=100)
        self.tile_window = deque(maxlen=100)
        
        # Лучшие результаты
        self.best_score = 0
        self.best_tile = 0
    
    def train(
        self,
        n_games: int = 1000,
        games_per_training: int = 5,
        train_steps_per_batch: int = 50,
        save_every: int = 100,
        eval_every: int = 50,
        eval_games: int = 10,
        temperature: float = 1.0,
        verbose: bool = True
    ) -> Dict:
        """
        Основной цикл обучения.
        
        Args:
            n_games: общее количество игр
            games_per_training: игр перед каждой фазой обучения
            train_steps_per_batch: шагов обучения после каждого батча игр
            save_every: сохранять модель каждые N игр
            eval_every: оценивать модель каждые N игр
            eval_games: количество игр для оценки
            temperature: начальная температура для exploration
            verbose: подробный вывод
        """
        start_time = time.time()
        device_info = get_device_info()
        
        print("=" * 70)
        print("🧠 Alpha2048 Training")
        print("=" * 70)
        print(f"Device: {device_info['name']} ({device_info['device']})")
        print(f"Total games: {n_games}")
        print(f"Games per training: {games_per_training}")
        print(f"Training steps per batch: {train_steps_per_batch}")
        print(f"Temperature: {temperature}")
        
        if self.agent.curriculum:
            print(f"\n📚 Curriculum Learning: ENABLED")
            print(self.agent.curriculum.get_status())
        
        print("=" * 70 + "\n")
        
        games_played = 0
        batch_games = 0
        
        while games_played < n_games:
            # === Self-play ===
            trajectory, stats = self.agent.self_play(
                game_mode='infinite',
                temperature=temperature
            )
            self.agent.add_to_buffer(trajectory)
            
            # Записываем статистику
            self.game_scores.append(stats['score'])
            self.game_tiles.append(stats['max_tile'])
            self.game_moves.append(stats['moves'])
            self.score_window.append(stats['score'])
            self.tile_window.append(stats['max_tile'])
            
            if stats['score'] > self.best_score:
                self.best_score = stats['score']
            if stats['max_tile'] > self.best_tile:
                self.best_tile = stats['max_tile']
            
            games_played += 1
            batch_games += 1
            
            # === Обучение ===
            if batch_games >= games_per_training:
                batch_losses = []
                for _ in range(train_steps_per_batch):
                    losses = self.agent.train_step()
                    if losses:
                        batch_losses.append(losses['total'])
                
                if batch_losses:
                    avg_loss = np.mean(batch_losses)
                    self.training_losses.append(avg_loss)
                
                batch_games = 0
            
            # === Вывод прогресса ===
            if verbose and games_played % 10 == 0:
                avg_score = np.mean(self.score_window)
                avg_tile = np.mean(self.tile_window)
                elapsed = time.time() - start_time
                games_per_sec = games_played / elapsed
                
                curriculum_info = ""
                if self.agent.curriculum:
                    stage = self.agent.curriculum.current_stage
                    curriculum_info = f" | Stage: {stage.name} ({stage.current_success_rate:.0%})"
                
                loss_str = f" | Loss: {self.training_losses[-1]:.3f}" if self.training_losses else ""
                
                print(
                    f"Game {games_played}/{n_games} | "
                    f"Avg Score: {avg_score:.0f} | "
                    f"Avg Tile: {avg_tile:.0f} | "
                    f"Best: {self.best_score} ({self.best_tile})"
                    f"{loss_str}{curriculum_info} | "
                    f"Speed: {games_per_sec:.1f} g/s"
                )
            
            # === Оценка ===
            if games_played % eval_every == 0:
                eval_results = self.evaluate(eval_games)
                print(f"\n📊 Evaluation ({eval_games} games):")
                print(f"   Score: {eval_results['avg_score']:.0f} ± {eval_results['std_score']:.0f}")
                print(f"   Max Tile: {eval_results['avg_tile']:.0f}")
                print(f"   Best: {eval_results['best_score']} ({eval_results['best_tile']})")
                
                # Распределение тайлов
                dist = eval_results['tile_distribution']
                dist_str = ", ".join([f"{k}:{v}" for k, v in sorted(dist.items(), key=lambda x: -x[0])[:5]])
                print(f"   Tiles: {dist_str}\n")
            
            # === Сохранение ===
            if games_played % save_every == 0:
                self._save_checkpoint(games_played)
        
        # Финальное сохранение
        self._save_checkpoint(games_played, final=True)
        self._save_logs()
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("🎉 Training Complete!")
        print("=" * 70)
        print(f"Total games: {games_played}")
        print(f"Best score: {self.best_score}")
        print(f"Best tile: {self.best_tile}")
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Training steps: {self.agent.training_steps}")
        
        if self.agent.curriculum:
            print(f"\n📚 Final Curriculum Status:")
            print(self.agent.curriculum.get_status())
        
        return {
            'games_played': games_played,
            'best_score': self.best_score,
            'best_tile': self.best_tile,
            'training_steps': self.agent.training_steps,
            'elapsed_time': elapsed
        }
    
    def evaluate(self, n_games: int = 10) -> Dict:
        """Оценка модели без exploration"""
        scores = []
        tiles = []
        moves = []
        
        for _ in range(n_games):
            game = Game2048(mode='infinite')
            
            while not game.is_game_over():
                # Используем MCTS с temperature=0 для лучшего хода
                action, _ = self.agent.select_action(
                    game, 
                    use_mcts=False,  # Быстрый режим для eval
                    temperature=0.0
                )
                
                _, done, _ = game.move(game_2048.Direction(action))
                if done:
                    break
            
            scores.append(game.score)
            tiles.append(game.max_tile)
            moves.append(game.moves)
        
        tile_dist = {}
        for t in tiles:
            tile_dist[t] = tile_dist.get(t, 0) + 1
        
        return {
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'avg_tile': np.mean(tiles),
            'best_score': max(scores),
            'best_tile': max(tiles),
            'avg_moves': np.mean(moves),
            'tile_distribution': tile_dist
        }
    
    def _save_checkpoint(self, games: int, final: bool = False):
        """Сохранение контрольной точки"""
        filename = "alpha2048_final.pt" if final else f"alpha2048_g{games}.pt"
        path = os.path.join(self.save_dir, filename)
        self.agent.save(path)
        
        # Сохраняем лучшую модель
        if self.game_scores and self.game_scores[-1] >= self.best_score * 0.95:
            best_path = os.path.join(self.save_dir, "alpha2048_best.pt")
            self.agent.save(best_path)
    
    def _save_logs(self):
        """Сохранение логов"""
        log_data = {
            'game_scores': self.game_scores,
            'game_tiles': self.game_tiles,
            'game_moves': self.game_moves,
            'training_losses': self.training_losses,
            'best_score': self.best_score,
            'best_tile': self.best_tile,
            'training_steps': self.agent.training_steps,
            'games_played': self.agent.games_played,
            'timestamp': datetime.now().isoformat()
        }
        
        log_path = os.path.join(self.log_dir, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📝 Logs saved to {log_path}")


# Импорт Direction для evaluate
import game_2048


def quick_train(n_games: int = 100) -> Alpha2048Agent:
    """Быстрое обучение для демонстрации"""
    print("🚀 Quick Training Demo")
    print("-" * 50)
    
    agent = Alpha2048Agent(
        n_channels=64,
        n_residual_blocks=3,
        mcts_simulations=20,
        batch_size=64,
        use_curriculum=True
    )
    
    trainer = Alpha2048Trainer(agent)
    trainer.train(
        n_games=n_games,
        games_per_training=5,
        train_steps_per_batch=20,
        save_every=n_games,
        eval_every=n_games // 2,
        eval_games=5,
        verbose=True
    )
    
    return agent


if __name__ == "__main__":
    import sys
    
    n_games = 100
    if len(sys.argv) > 1:
        try:
            n_games = int(sys.argv[1])
        except ValueError:
            pass
    
    quick_train(n_games)

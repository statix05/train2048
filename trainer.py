"""
Система обучения нейросети для 2048
Включает:
- Цикл обучения с визуализацией прогресса
- Валидацию и тестирование
- Сохранение лучших моделей
- Логирование метрик
"""
import numpy as np
import time
import os
from typing import List, Tuple, Optional
from collections import deque
import json
from datetime import datetime

from game_2048 import Game2048, Direction
from neural_network import DQNAgent, device


class Trainer:
    """
    Класс для обучения агента играть в 2048
    """
    
    def __init__(
        self,
        agent: DQNAgent,
        save_dir: str = "models",
        log_dir: str = "logs"
    ):
        self.agent = agent
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Метрики
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_max_tiles = []
        self.episode_moves = []
        self.training_losses = []
        
        self.best_score = 0
        self.best_max_tile = 0
        
        # Moving averages для отображения прогресса
        self.reward_window = deque(maxlen=100)
        self.score_window = deque(maxlen=100)
        self.tile_window = deque(maxlen=100)
    
    def train_episode(self, game: Game2048, render: bool = False) -> dict:
        """
        Один эпизод обучения
        """
        state = game.get_state()
        features = game.get_features()
        
        total_reward = 0.0
        episode_loss = []
        
        while True:
            # Получаем допустимые ходы
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            valid_moves_int = [int(m) for m in valid_moves]
            
            # Выбираем действие
            action = self.agent.select_action(state, features, valid_moves_int)
            
            # Выполняем действие
            reward, done, info = game.move(Direction(action))
            
            # Получаем новое состояние
            next_state = game.get_state()
            next_features = game.get_features()
            
            # Сохраняем опыт
            self.agent.store_experience(
                state, features, action, reward,
                next_state, next_features, done
            )
            
            # Обучаем
            loss = self.agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            total_reward += reward
            state = next_state
            features = next_features
            
            if done:
                break
            
            if render:
                print(game)
                print(f"Action: {Direction(action).name}, Reward: {reward:.2f}")
                time.sleep(0.1)
        
        return {
            'reward': total_reward,
            'score': game.score,
            'max_tile': game.max_tile,
            'moves': game.moves,
            'loss': np.mean(episode_loss) if episode_loss else 0.0
        }
    
    def train(
        self,
        n_episodes: int = 10000,
        save_every: int = 500,
        eval_every: int = 100,
        eval_episodes: int = 10,
        verbose: bool = True
    ):
        """
        Основной цикл обучения
        """
        start_time = time.time()
        
        for episode in range(1, n_episodes + 1):
            # Создаем новую игру
            game = Game2048()
            
            # Обучаем эпизод
            result = self.train_episode(game)
            
            # Сохраняем метрики
            self.episode_rewards.append(result['reward'])
            self.episode_scores.append(result['score'])
            self.episode_max_tiles.append(result['max_tile'])
            self.episode_moves.append(result['moves'])
            if result['loss'] > 0:
                self.training_losses.append(result['loss'])
            
            self.reward_window.append(result['reward'])
            self.score_window.append(result['score'])
            self.tile_window.append(result['max_tile'])
            
            # Обновляем лучшие результаты
            if result['score'] > self.best_score:
                self.best_score = result['score']
            if result['max_tile'] > self.best_max_tile:
                self.best_max_tile = result['max_tile']
            
            # Логирование
            if verbose and episode % 10 == 0:
                avg_reward = np.mean(self.reward_window)
                avg_score = np.mean(self.score_window)
                avg_tile = np.mean(self.tile_window)
                epsilon = self.agent.get_epsilon()
                
                elapsed = time.time() - start_time
                eps_per_sec = episode / elapsed
                
                print(f"Episode {episode}/{n_episodes} | "
                      f"Avg Reward: {avg_reward:.1f} | "
                      f"Avg Score: {avg_score:.0f} | "
                      f"Avg Max Tile: {avg_tile:.0f} | "
                      f"Best: {self.best_score} ({self.best_max_tile}) | "
                      f"ε: {epsilon:.3f} | "
                      f"Speed: {eps_per_sec:.1f} ep/s")
            
            # Оценка
            if episode % eval_every == 0:
                eval_result = self.evaluate(eval_episodes)
                print(f"\n📊 Evaluation ({eval_episodes} games):")
                print(f"   Score: {eval_result['avg_score']:.0f} ± {eval_result['std_score']:.0f}")
                print(f"   Max Tile Avg: {eval_result['avg_max_tile']:.0f}")
                print(f"   Best in Eval: {eval_result['best_score']} ({eval_result['best_max_tile']})")
                
                # Распределение максимальных плиток
                tile_dist = eval_result['tile_distribution']
                dist_str = ", ".join([f"{k}: {v}" for k, v in sorted(tile_dist.items(), key=lambda x: -x[1])])
                print(f"   Tile Distribution: {dist_str}\n")
            
            # Сохранение
            if episode % save_every == 0:
                self.save_checkpoint(episode)
        
        # Финальное сохранение
        self.save_checkpoint(n_episodes, final=True)
        self.save_logs()
        
        print(f"\n🎉 Training completed!")
        print(f"   Total episodes: {n_episodes}")
        print(f"   Best score: {self.best_score}")
        print(f"   Best max tile: {self.best_max_tile}")
        print(f"   Total time: {time.time() - start_time:.1f}s")
    
    def evaluate(self, n_episodes: int = 10) -> dict:
        """
        Оценка текущей модели (без exploration)
        """
        scores = []
        max_tiles = []
        moves_list = []
        
        for _ in range(n_episodes):
            game = Game2048()
            state = game.get_state()
            features = game.get_features()
            
            while True:
                valid_moves = game.get_valid_moves()
                if not valid_moves:
                    break
                
                valid_moves_int = [int(m) for m in valid_moves]
                
                # Выбираем действие без exploration (epsilon=0)
                action = self.agent.policy_net.get_action(
                    state, features, valid_moves_int, epsilon=0.0
                )
                
                _, done, _ = game.move(Direction(action))
                
                state = game.get_state()
                features = game.get_features()
                
                if done:
                    break
            
            scores.append(game.score)
            max_tiles.append(game.max_tile)
            moves_list.append(game.moves)
        
        # Распределение максимальных плиток
        tile_dist = {}
        for tile in max_tiles:
            tile_dist[tile] = tile_dist.get(tile, 0) + 1
        
        return {
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'avg_max_tile': np.mean(max_tiles),
            'best_score': max(scores),
            'best_max_tile': max(max_tiles),
            'avg_moves': np.mean(moves_list),
            'tile_distribution': tile_dist
        }
    
    def save_checkpoint(self, episode: int, final: bool = False):
        """Сохранение контрольной точки"""
        filename = "model_final.pt" if final else f"model_ep{episode}.pt"
        path = os.path.join(self.save_dir, filename)
        self.agent.save(path)
        
        # Сохраняем также лучшую модель
        if self.episode_scores and self.episode_scores[-1] >= self.best_score * 0.95:
            best_path = os.path.join(self.save_dir, "model_best.pt")
            self.agent.save(best_path)
    
    def save_logs(self):
        """Сохранение логов обучения"""
        log_data = {
            'episode_rewards': self.episode_rewards,
            'episode_scores': self.episode_scores,
            'episode_max_tiles': self.episode_max_tiles,
            'episode_moves': self.episode_moves,
            'training_losses': self.training_losses,
            'best_score': self.best_score,
            'best_max_tile': self.best_max_tile,
            'timestamp': datetime.now().isoformat()
        }
        
        log_path = os.path.join(self.log_dir, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump(log_data, f)
        
        print(f"Logs saved to {log_path}")


def quick_train(episodes: int = 1000):
    """Быстрое обучение для демонстрации"""
    print(f"🚀 Starting quick training on {device}")
    print(f"   Episodes: {episodes}")
    print("-" * 50)
    
    agent = DQNAgent(
        learning_rate=5e-4,
        buffer_size=50000,
        batch_size=64,
        target_update=500,
        epsilon_decay=episodes * 5
    )
    
    trainer = Trainer(agent)
    trainer.train(
        n_episodes=episodes,
        save_every=episodes // 2,
        eval_every=episodes // 10,
        eval_episodes=5
    )
    
    return agent, trainer


if __name__ == "__main__":
    # Быстрое обучение для демонстрации
    agent, trainer = quick_train(500)

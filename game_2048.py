"""
2048 Game - Core Logic Module
Оригинальная реализация игры 2048 с оптимизированной логикой
"""
import numpy as np
import random
from typing import Tuple, List, Optional
from enum import IntEnum


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Game2048:
    """
    Класс игры 2048 с оптимизированной логикой и дополнительными метриками
    """
    
    def __init__(self, size: int = 4, mode: str = 'classic'):
        self.size = size
        self.mode = mode  # 'classic' or 'dynamic'
        self.board = np.zeros((size, size), dtype=np.int32)
        self.score = 0
        self.moves = 0
        self.max_tile = 0
        self.history = []
        self._spawn_tile()
        self._spawn_tile()
        self._update_max_tile()
    
    def reset(self) -> np.ndarray:
        """Сброс игры и возврат начального состояния"""
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.moves = 0
        self.max_tile = 0
        self.history = []
        self._spawn_tile()
        self._spawn_tile()
        self._update_max_tile()
        return self.get_state()
    
    def _spawn_tile(self) -> bool:
        """
        Добавление новой плитки.
        Classic mode: 90% - 2, 10% - 4.
        Dynamic mode: Increases spawn value based on max_tile.
        """
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            return False
        row, col = random.choice(empty_cells)
        
        value = 2
        if self.mode == 'dynamic' and self.max_tile >= 64:
             # Продвинутая логика для поздней игры (Arcade/Dynamic Mode)
            r = random.random()
            if r < 0.55:
                value = max(2, int(self.max_tile / 16))
            elif r < 0.90:
                value = max(2, int(self.max_tile / 8))
            else:
                value = max(2, int(self.max_tile / 4))
        else:
             # Classic 2048 logic
             value = 4 if random.random() < 0.1 else 2
             
        self.board[row, col] = value
        return True
    
    def _update_max_tile(self):
        """Обновление максимальной плитки"""
        self.max_tile = int(np.max(self.board))
    
    def _compress(self, row: np.ndarray) -> Tuple[np.ndarray, int]:
        """Сжатие строки влево и подсчет очков"""
        # Убираем нули
        non_zero = row[row != 0]
        score = 0
        result = []
        
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged = non_zero[i] * 2
                result.append(merged)
                score += merged
                i += 2
            else:
                result.append(non_zero[i])
                i += 1
        
        # Дополняем нулями
        result.extend([0] * (self.size - len(result)))
        return np.array(result, dtype=np.int32), score
    
    def _move_left(self) -> Tuple[np.ndarray, int]:
        """Движение влево"""
        new_board = np.zeros_like(self.board)
        total_score = 0
        for i in range(self.size):
            new_board[i], score = self._compress(self.board[i])
            total_score += score
        return new_board, total_score
    
    def _move_right(self) -> Tuple[np.ndarray, int]:
        """Движение вправо"""
        new_board = np.zeros_like(self.board)
        total_score = 0
        for i in range(self.size):
            new_board[i], score = self._compress(self.board[i][::-1])
            new_board[i] = new_board[i][::-1]
            total_score += score
        return new_board, total_score
    
    def _move_up(self) -> Tuple[np.ndarray, int]:
        """Движение вверх"""
        transposed = self.board.T.copy()
        new_board = np.zeros_like(transposed)
        total_score = 0
        for i in range(self.size):
            new_board[i], score = self._compress(transposed[i])
            total_score += score
        return new_board.T, total_score
    
    def _move_down(self) -> Tuple[np.ndarray, int]:
        """Движение вниз"""
        transposed = self.board.T.copy()
        new_board = np.zeros_like(transposed)
        total_score = 0
        for i in range(self.size):
            new_board[i], score = self._compress(transposed[i][::-1])
            new_board[i] = new_board[i][::-1]
            total_score += score
        return new_board.T, total_score
    
    def move(self, direction: Direction) -> Tuple[int, bool, dict]:
        """
        Выполнение хода
        Returns: (reward, done, info)
        """
        old_board = self.board.copy()
        old_score = self.score
        
        # Выполняем движение
        if direction == Direction.UP:
            new_board, move_score = self._move_up()
        elif direction == Direction.DOWN:
            new_board, move_score = self._move_down()
        elif direction == Direction.LEFT:
            new_board, move_score = self._move_left()
        else:
            new_board, move_score = self._move_right()
        
        # Проверяем, изменилось ли поле
        moved = not np.array_equal(old_board, new_board)
        
        if moved:
            self.board = new_board
            self.score += move_score
            self._spawn_tile()
            self._update_max_tile()
            self.moves += 1
            self.history.append(old_board.copy())
        
        # Проверяем конец игры
        done = self.is_game_over()
        
        # Рассчитываем награду
        reward = self._calculate_reward(moved, move_score, old_board, done)
        
        info = {
            'moved': moved,
            'score': self.score,
            'max_tile': self.max_tile,
            'moves': self.moves,
            'merge_score': move_score
        }
        
        return reward, done, info
    
    def _calculate_reward(self, moved: bool, merge_score: int, 
                          old_board: np.ndarray, done: bool) -> float:
        """
        Креативная система наград:
        - Награда за слияние плиток
        - Бонус за сохранение максимального тайла в углу
        - Бонус за монотонность (упорядоченность)
        - Бонус за пустые клетки
        - Штраф за невозможный ход
        """
        if not moved:
            return -20.0  # Штраф за невозможный ход
        
        reward = 0.0
        
        # Награда за слияние (логарифмическая)
        if merge_score > 0:
            reward += np.log2(merge_score + 1) * 2.5  # Increased weight
        
        # Бонус за пустые клетки
        empty_cells = np.sum(self.board == 0)
        reward += empty_cells * 1.0  # Increased weight
        
        # Бонус за максимальный тайл в углу
        max_val = np.max(self.board)
        corners = [self.board[0, 0], self.board[0, -1], 
                   self.board[-1, 0], self.board[-1, -1]]
        if max_val in corners:
            reward += np.log2(max_val + 1) * 2.0  # Increased weight
        
        # Бонус за монотонность
        reward += self._monotonicity_score() * 0.5  # Increased weight
        
        # Бонус за гладкость (меньше разница между соседями)
        reward += self._smoothness_score() * 0.2
        
        # Штраф за проигрыш
        if done:
            reward -= 100.0
        
        return reward
    
    def _monotonicity_score(self) -> float:
        """Оценка монотонности - насколько поле упорядочено"""
        score = 0.0
        
        # Проверяем строки
        for row in self.board:
            non_zero = row[row > 0]
            if len(non_zero) > 1:
                # Проверяем возрастание или убывание
                increasing = all(non_zero[i] <= non_zero[i+1] for i in range(len(non_zero)-1))
                decreasing = all(non_zero[i] >= non_zero[i+1] for i in range(len(non_zero)-1))
                if increasing or decreasing:
                    score += 1.0
        
        # Проверяем столбцы
        for col in self.board.T:
            non_zero = col[col > 0]
            if len(non_zero) > 1:
                increasing = all(non_zero[i] <= non_zero[i+1] for i in range(len(non_zero)-1))
                decreasing = all(non_zero[i] >= non_zero[i+1] for i in range(len(non_zero)-1))
                if increasing or decreasing:
                    score += 1.0
        
        return score
    
    def _smoothness_score(self) -> float:
        """Оценка гладкости - меньше резких переходов между соседними клетками"""
        score = 0.0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] > 0:
                    val = np.log2(self.board[i, j])
                    # Проверяем соседей
                    for di, dj in [(0, 1), (1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            if self.board[ni, nj] > 0:
                                neighbor_val = np.log2(self.board[ni, nj])
                                score -= abs(val - neighbor_val)
        return score
    
    def is_game_over(self) -> bool:
        """Проверка конца игры"""
        # Есть пустые клетки
        if np.any(self.board == 0):
            return False
        
        # Проверяем возможность слияния по горизонтали
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j + 1]:
                    return False
        
        # Проверяем возможность слияния по вертикали
        for i in range(self.size - 1):
            for j in range(self.size):
                if self.board[i, j] == self.board[i + 1, j]:
                    return False
        
        return True
    
    def get_valid_moves(self) -> List[Direction]:
        """Получение списка допустимых ходов"""
        valid = []
        for direction in Direction:
            # Проверяем, изменит ли этот ход поле
            old_board = self.board.copy()
            if direction == Direction.UP:
                new_board, _ = self._move_up()
            elif direction == Direction.DOWN:
                new_board, _ = self._move_down()
            elif direction == Direction.LEFT:
                new_board, _ = self._move_left()
            else:
                new_board, _ = self._move_right()
            
            if not np.array_equal(old_board, new_board):
                valid.append(direction)
        
        return valid
    
    def get_state(self) -> np.ndarray:
        """
        Получение состояния для нейросети
        Нормализованное представление: log2(value) / 17 (для max = 131072)
        """
        state = np.zeros_like(self.board, dtype=np.float32)
        mask = self.board > 0
        state[mask] = np.log2(self.board[mask]) / 17.0
        return state
    
    def get_features(self) -> np.ndarray:
        """
        Дополнительные признаки для нейросети:
        - Количество пустых клеток (нормализовано)
        - Максимальный тайл (log2, нормализовано)
        - Монотонность
        - Гладкость
        - Максимальный тайл в углу (0 или 1)
        - Возможные ходы (4 значения)
        """
        features = []
        
        # Пустые клетки
        empty_ratio = np.sum(self.board == 0) / (self.size * self.size)
        features.append(empty_ratio)
        
        # Максимальный тайл
        max_tile = np.max(self.board)
        max_tile_norm = np.log2(max_tile + 1) / 17.0 if max_tile > 0 else 0
        features.append(max_tile_norm)
        
        # Монотонность (нормализовано)
        features.append(self._monotonicity_score() / 8.0)
        
        # Гладкость (нормализовано, примерно)
        features.append(max(min(self._smoothness_score() / 50.0 + 0.5, 1.0), 0.0))
        
        # Максимальный тайл в углу
        corners = [self.board[0, 0], self.board[0, -1], 
                   self.board[-1, 0], self.board[-1, -1]]
        features.append(1.0 if max_tile in corners else 0.0)
        
        # Возможные ходы
        valid_moves = self.get_valid_moves()
        for d in Direction:
            features.append(1.0 if d in valid_moves else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def copy(self) -> 'Game2048':
        """Создание копии игры"""
        game_copy = Game2048(self.size)
        game_copy.board = self.board.copy()
        game_copy.score = self.score
        game_copy.moves = self.moves
        game_copy.max_tile = self.max_tile
        return game_copy
    
    def __str__(self) -> str:
        """Строковое представление игры"""
        lines = [f"Score: {self.score} | Max: {self.max_tile} | Moves: {self.moves}"]
        lines.append("-" * 25)
        for row in self.board:
            line = "|"
            for val in row:
                if val == 0:
                    line += "    .|"
                else:
                    line += f"{val:5}|"
            lines.append(line)
        lines.append("-" * 25)
        return "\n".join(lines)


if __name__ == "__main__":
    # Тестирование игры
    game = Game2048()
    print("Начальное состояние:")
    print(game)
    
    # Тест нескольких ходов
    for i in range(10):
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            print("Игра окончена!")
            break
        
        move = random.choice(valid_moves)
        reward, done, info = game.move(move)
        print(f"\nХод {i+1}: {move.name}, Reward: {reward:.2f}")
        print(game)
        
        if done:
            print("Игра окончена!")
            break

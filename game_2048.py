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
    
    Режимы игры:
    - 'classic': Стандартные правила 2048
    - 'dynamic': Динамическое изменение минимального тайла на основе рекорда
    - 'infinite': Бесконечный режим с динамическими правилами и бонусами
    
    Формула динамического минимального тайла:
    minTile = max(2, record / 128)
    
    256  → min=2  (2 исчезает)
    512  → min=4  (4 исчезает) 
    1024 → min=8  (8 исчезает)
    2048 → min=16 и т.д.
    
    Система бонусов (infinite mode):
    - При достижении 2048, 4096, 8192... игрок получает бонус удаления блока
    - Бонусы накапливаются и могут использоваться в любое время
    """
    
    # Пороги для бонусов (степени двойки начиная с 2048)
    BONUS_THRESHOLDS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    
    def __init__(self, size: int = 4, mode: str = 'classic'):
        self.size = size
        self.mode = mode  # 'classic', 'dynamic', or 'infinite'
        self.board = np.zeros((size, size), dtype=np.int32)
        self.score = 0
        self.moves = 0
        self.max_tile = 0
        self.record = 0  # Максимальный тайл за всю игру (для расчета minTile)
        self.history = []
        
        # Система бонусов
        self.bonus_count = 0  # Накопленные бонусы удаления
        self.claimed_bonuses = set()  # Уже полученные бонусы (пороги)
        self.total_bonuses_earned = 0  # Всего заработано бонусов
        self.total_bonuses_used = 0  # Всего использовано бонусов
        
        self._spawn_tile()
        self._spawn_tile()
        self._update_max_tile()
    
    def reset(self) -> np.ndarray:
        """Сброс игры и возврат начального состояния"""
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.moves = 0
        self.max_tile = 0
        self.record = 0
        self.history = []
        
        # Сброс бонусов
        self.bonus_count = 0
        self.claimed_bonuses = set()
        self.total_bonuses_earned = 0
        self.total_bonuses_used = 0
        
        self._spawn_tile()
        self._spawn_tile()
        self._update_max_tile()
        return self.get_state()
    
    def get_min_tile(self) -> int:
        """
        Вычисление минимального тайла на основе текущего рекорда.
        
        Формула: minTile = record / 128
        
        Примеры:
        - record < 256:  minTile = 2 (стандартно)
        - record = 256:  minTile = 2 (256/128 = 2)
        - record = 512:  minTile = 4 (512/128 = 4)
        - record = 1024: minTile = 8 (1024/128 = 8)
        - record = 2048: minTile = 16
        - record = 4096: minTile = 32
        
        Returns:
            int: Минимальное значение тайла для спавна
        """
        if self.mode == 'classic':
            return 2
        
        if self.record < 256:
            return 2
        
        # Формула: minTile = record / 128
        min_tile = self.record // 128
        
        # Убедимся, что это степень двойки
        # (record всегда степень двойки, так что деление на 128 даст степень двойки)
        return max(2, min_tile)
    
    def get_spawn_tiles(self) -> List[int]:
        """
        Получение списка возможных тайлов для спавна.
        
        В режиме infinite/dynamic минимальный тайл зависит от рекорда.
        Спавнятся: minTile (90%) или minTile*2 (10%)
        
        Returns:
            List[int]: [основной_тайл, редкий_тайл]
        """
        min_tile = self.get_min_tile()
        return [min_tile, min_tile * 2]
    
    def _spawn_tile(self) -> bool:
        """
        Добавление новой плитки.
        
        Classic mode: 90% - 2, 10% - 4
        Dynamic/Infinite mode: 90% - minTile, 10% - minTile*2
        
        где minTile = max(2, record / 128)
        """
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            return False
        row, col = random.choice(empty_cells)
        
        if self.mode in ('dynamic', 'infinite'):
            spawn_tiles = self.get_spawn_tiles()
            # 90% основной тайл, 10% редкий
            value = spawn_tiles[1] if random.random() < 0.1 else spawn_tiles[0]
        else:
            # Classic 2048 logic
            value = 4 if random.random() < 0.1 else 2
             
        self.board[row, col] = value
        return True
    
    def _update_max_tile(self):
        """Обновление максимальной плитки и проверка бонусов"""
        old_max = self.max_tile
        self.max_tile = int(np.max(self.board))
        
        # Обновляем рекорд (только растёт)
        if self.max_tile > self.record:
            self.record = self.max_tile
            
            # Проверяем бонусы в режиме infinite
            if self.mode == 'infinite':
                self._check_and_award_bonus()
    
    def _check_and_award_bonus(self):
        """
        Проверка и выдача бонуса за достижение нового порога.
        Бонус выдаётся один раз за каждый порог (2048, 4096, 8192...).
        """
        for threshold in self.BONUS_THRESHOLDS:
            if self.record >= threshold and threshold not in self.claimed_bonuses:
                self.claimed_bonuses.add(threshold)
                self.bonus_count += 1
                self.total_bonuses_earned += 1
    
    def can_use_bonus(self) -> bool:
        """Проверка, есть ли доступные бонусы"""
        return self.bonus_count > 0
    
    def use_bonus_remove_tile(self, row: int, col: int) -> bool:
        """
        Использование бонуса для удаления тайла.
        
        Args:
            row: Строка тайла (0-3)
            col: Столбец тайла (0-3)
            
        Returns:
            bool: True если бонус успешно использован
        """
        if not self.can_use_bonus():
            return False
        
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        
        if self.board[row, col] == 0:
            return False  # Нельзя удалить пустую клетку
        
        # Удаляем тайл
        self.board[row, col] = 0
        self.bonus_count -= 1
        self.total_bonuses_used += 1
        
        return True
    
    def get_bonus_info(self) -> dict:
        """
        Получение информации о бонусах.
        
        Returns:
            dict: Информация о бонусах
        """
        next_bonus = None
        for threshold in self.BONUS_THRESHOLDS:
            if threshold not in self.claimed_bonuses:
                next_bonus = threshold
                break
        
        return {
            'available': self.bonus_count,
            'total_earned': self.total_bonuses_earned,
            'total_used': self.total_bonuses_used,
            'claimed_thresholds': sorted(list(self.claimed_bonuses)),
            'next_bonus_at': next_bonus,
            'progress_to_next': self.record / next_bonus if next_bonus else 1.0
        }
    
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
            'record': self.record,
            'moves': self.moves,
            'merge_score': move_score,
            'min_tile': self.get_min_tile(),
            'spawn_tiles': self.get_spawn_tiles(),
            'bonus_count': self.bonus_count,
            'mode': self.mode
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
        game_copy = Game2048(self.size, self.mode)
        game_copy.board = self.board.copy()
        game_copy.score = self.score
        game_copy.moves = self.moves
        game_copy.max_tile = self.max_tile
        game_copy.record = self.record
        game_copy.bonus_count = self.bonus_count
        game_copy.claimed_bonuses = self.claimed_bonuses.copy()
        game_copy.total_bonuses_earned = self.total_bonuses_earned
        game_copy.total_bonuses_used = self.total_bonuses_used
        return game_copy
    
    def __str__(self) -> str:
        """Строковое представление игры"""
        min_tile = self.get_min_tile()
        spawn_info = f"Spawn: {min_tile}/{min_tile*2}" if self.mode != 'classic' else "Spawn: 2/4"
        bonus_info = f" | Bonuses: {self.bonus_count}" if self.mode == 'infinite' else ""
        
        lines = [f"Score: {self.score} | Max: {self.max_tile} | Moves: {self.moves} | {spawn_info}{bonus_info}"]
        lines.append("-" * 45)
        for row in self.board:
            line = "|"
            for val in row:
                if val == 0:
                    line += "      .|"
                else:
                    line += f"{val:7}|"
            lines.append(line)
        lines.append("-" * 45)
        return "\n".join(lines)


def demonstrate_dynamic_mechanics():
    """
    Демонстрация динамической механики minTile.
    """
    print("="*60)
    print("ДЕМОНСТРАЦИЯ ДИНАМИЧЕСКОЙ МЕХАНИКИ МИНИМАЛЬНОГО ТАЙЛА")
    print("="*60)
    print()
    print("Формула: minTile = record / 128")
    print()
    print("Таблица соответствия:")
    print("-" * 40)
    print(f"{'Record':>10} | {'minTile':>8} | {'Spawn':>12}")
    print("-" * 40)
    
    records = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    
    for record in records:
        if record < 256:
            min_tile = 2
        else:
            min_tile = record // 128
        spawn = f"{min_tile}/{min_tile*2}"
        marker = " ← исчезает " + str(min_tile // 2) if record >= 256 else ""
        print(f"{record:>10} | {min_tile:>8} | {spawn:>12}{marker}")
    
    print("-" * 40)
    print()


def demonstrate_bonus_system():
    """
    Демонстрация системы бонусов.
    """
    print("="*60)
    print("ДЕМОНСТРАЦИЯ СИСТЕМЫ БОНУСОВ (INFINITE MODE)")
    print("="*60)
    print()
    print("Бонусы выдаются при достижении:")
    for i, threshold in enumerate(Game2048.BONUS_THRESHOLDS[:8], 1):
        print(f"  {i}. {threshold:>7} - бонус #{i}")
    print("  ...")
    print()
    print("Каждый бонус позволяет удалить один любой блок с поля.")
    print("Бонусы накапливаются и могут быть использованы в любое время.")
    print()


def test_game_modes():
    """
    Тестирование всех режимов игры.
    """
    print("="*60)
    print("ТЕСТИРОВАНИЕ РЕЖИМОВ ИГРЫ")
    print("="*60)
    print()
    
    for mode in ['classic', 'dynamic', 'infinite']:
        print(f"\n--- Режим: {mode.upper()} ---")
        game = Game2048(mode=mode)
        
        # Симулируем высокий рекорд для демонстрации
        game.record = 1024
        game.max_tile = 1024
        
        if mode == 'infinite':
            # Симулируем получение бонуса
            game.record = 2048
            game.max_tile = 2048
            game._check_and_award_bonus()
        
        print(f"Record: {game.record}")
        print(f"Min tile: {game.get_min_tile()}")
        print(f"Spawn tiles: {game.get_spawn_tiles()}")
        
        if mode == 'infinite':
            bonus_info = game.get_bonus_info()
            print(f"Bonuses available: {bonus_info['available']}")
            print(f"Next bonus at: {bonus_info['next_bonus_at']}")


if __name__ == "__main__":
    # Демонстрация механик
    demonstrate_dynamic_mechanics()
    demonstrate_bonus_system()
    test_game_modes()
    
    print("\n" + "="*60)
    print("ИНТЕРАКТИВНЫЙ ТЕСТ INFINITE MODE")
    print("="*60)
    
    # Тестирование игры в infinite режиме
    game = Game2048(mode='infinite')
    print("\nНачальное состояние:")
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
        print(f"Info: minTile={info['min_tile']}, spawn={info['spawn_tiles']}, bonuses={info['bonus_count']}")
        print(game)
        
        if done:
            print("Игра окончена!")
            break

"""
2048 Game - Core Logic Module
=============================

Полная реализация игры 2048 с:
- Оригинальным подсчётом очков
- Динамическим режимом (minTile scaling)
- Системой Record Combo
- Супер-бонусом сортировки

Подсчёт очков (оригинальный):
- При слиянии двух плиток очки = значение новой плитки
- Например: 2+2=4 → +4 очка, 4+4=8 → +8 очков
"""

import numpy as np
import random
from typing import Tuple, List, Optional, Dict
from enum import IntEnum
from dataclasses import dataclass, field


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


@dataclass
class RecordEvent:
    """Событие установки рекорда"""
    tile: int           # Значение нового рекордного тайла
    move_number: int    # Номер хода
    score: int          # Очки на момент рекорда
    is_combo: bool      # Был ли это combo (рекорд в течение 2 ходов)


class Game2048:
    """
    Игра 2048 с расширенной механикой.
    
    Режимы:
    - 'classic': Стандартные правила 2048
    - 'dynamic': Динамический minTile на основе рекорда
    - 'infinite': Dynamic + бонусы + combo
    
    Формула minTile: max(2, record / 128)
    
    Record Combo:
    - Срабатывает если новый рекорд установлен в течение 2 ходов после предыдущего
    - Даёт дополнительный бонус для тайлов от 256
    - После 2048 даёт супер-бонус сортировки
    
    Очки (оригинальная формула):
    - score += merged_tile_value при каждом слиянии
    """
    
    # Пороги для обычных бонусов
    BONUS_THRESHOLDS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    
    # Пороги для record combo (от 256)
    COMBO_THRESHOLDS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    
    # Максимальное количество ходов для combo
    COMBO_WINDOW = 2
    
    def __init__(self, size: int = 4, mode: str = 'classic'):
        self.size = size
        self.mode = mode
        self.board = np.zeros((size, size), dtype=np.int64)
        self.score = 0
        self.moves = 0
        self.max_tile = 0
        self.record = 0
        self.history: List[np.ndarray] = []
        
        # Система бонусов
        self.bonus_count = 0
        self.claimed_bonuses: set = set()
        self.total_bonuses_earned = 0
        self.total_bonuses_used = 0
        
        # Система Record Combo
        self.record_events: List[RecordEvent] = []
        self.combo_bonuses: List[int] = []  # Тайлы, за которые получен combo bonus
        self.sort_bonuses = 0  # Супер-бонусы сортировки (combo после 2048)
        self.last_record_move = -999  # Ход последнего рекорда
        
        # Статистика combo
        self.total_combos = 0
        self.total_sort_bonuses_earned = 0
        self.total_sort_bonuses_used = 0
        
        self._spawn_tile()
        self._spawn_tile()
        self._update_max_tile()
    
    def reset(self) -> np.ndarray:
        """Полный сброс игры"""
        self.board = np.zeros((self.size, self.size), dtype=np.int64)
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
        
        # Сброс combo
        self.record_events = []
        self.combo_bonuses = []
        self.sort_bonuses = 0
        self.last_record_move = -999
        self.total_combos = 0
        self.total_sort_bonuses_earned = 0
        self.total_sort_bonuses_used = 0
        
        self._spawn_tile()
        self._spawn_tile()
        self._update_max_tile()
        return self.get_state()
    
    # ========================================================================
    # DYNAMIC TILE SYSTEM
    # ========================================================================
    
    def get_min_tile(self) -> int:
        """
        Минимальный тайл для спавна.
        
        Формула: minTile = max(2, record / 128)
        
        Record  | minTile | Spawn
        --------|---------|-------
        < 256   | 2       | 2/4
        256     | 2       | 2/4
        512     | 4       | 4/8
        1024    | 8       | 8/16
        2048    | 16      | 16/32
        4096    | 32      | 32/64
        """
        if self.mode == 'classic':
            return 2
        
        if self.record < 256:
            return 2
        
        return max(2, self.record // 128)
    
    def get_spawn_tiles(self) -> List[int]:
        """Возможные тайлы для спавна: [common, rare]"""
        min_tile = self.get_min_tile()
        return [min_tile, min_tile * 2]
    
    def _spawn_tile(self) -> bool:
        """Спавн нового тайла (90% common, 10% rare)"""
        empty = list(zip(*np.where(self.board == 0)))
        if not empty:
            return False
        
        row, col = random.choice(empty)
        
        if self.mode in ('dynamic', 'infinite'):
            tiles = self.get_spawn_tiles()
            value = tiles[1] if random.random() < 0.1 else tiles[0]
        else:
            value = 4 if random.random() < 0.1 else 2
        
        self.board[row, col] = value
        return True
    
    # ========================================================================
    # RECORD & COMBO SYSTEM
    # ========================================================================
    
    def _update_max_tile(self):
        """Обновление max_tile и проверка record/combo"""
        old_max = self.max_tile
        self.max_tile = int(np.max(self.board))
        
        if self.max_tile > self.record:
            old_record = self.record
            self.record = self.max_tile
            
            # Проверяем combo
            is_combo = (self.moves - self.last_record_move) <= self.COMBO_WINDOW
            
            # Записываем событие рекорда
            event = RecordEvent(
                tile=self.record,
                move_number=self.moves,
                score=self.score,
                is_combo=is_combo and old_record >= 128  # Combo только от 256 (после 128)
            )
            self.record_events.append(event)
            
            # Обрабатываем combo
            if is_combo and self.record in self.COMBO_THRESHOLDS:
                self._handle_combo(self.record)
            
            self.last_record_move = self.moves
            
            # Проверяем стандартные бонусы (infinite mode)
            if self.mode == 'infinite':
                self._check_standard_bonus()
    
    def _handle_combo(self, tile: int):
        """Обработка Record Combo"""
        if tile < 256:
            return
        
        self.total_combos += 1
        self.combo_bonuses.append(tile)
        
        # После 2048 даём супер-бонус сортировки
        if tile >= 2048:
            self.sort_bonuses += 1
            self.total_sort_bonuses_earned += 1
    
    def _check_standard_bonus(self):
        """Проверка и выдача стандартных бонусов (удаление тайла)"""
        for threshold in self.BONUS_THRESHOLDS:
            if self.record >= threshold and threshold not in self.claimed_bonuses:
                self.claimed_bonuses.add(threshold)
                self.bonus_count += 1
                self.total_bonuses_earned += 1
    
    # ========================================================================
    # BONUS ACTIONS
    # ========================================================================
    
    def can_use_bonus(self) -> bool:
        """Есть ли бонус удаления"""
        return self.bonus_count > 0
    
    def can_use_sort_bonus(self) -> bool:
        """Есть ли супер-бонус сортировки"""
        return self.sort_bonuses > 0
    
    def use_bonus_remove_tile(self, row: int, col: int) -> bool:
        """Использовать бонус для удаления тайла"""
        if not self.can_use_bonus():
            return False
        
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        
        if self.board[row, col] == 0:
            return False
        
        self.board[row, col] = 0
        self.bonus_count -= 1
        self.total_bonuses_used += 1
        return True
    
    def use_sort_bonus(self) -> bool:
        """
        Использовать супер-бонус сортировки.
        
        Сортирует все тайлы по убыванию:
        - Наибольший в левом верхнем углу
        - Наименьший ближе к правому нижнему
        - Пустые клетки концентрируются справа внизу
        - Градиентное расположение (змейка)
        """
        if not self.can_use_sort_bonus():
            return False
        
        # Собираем все значения
        values = self.board.flatten().tolist()
        
        # Сортируем по убыванию (нули в конец)
        non_zero = sorted([v for v in values if v > 0], reverse=True)
        zeros = [0] * (self.size * self.size - len(non_zero))
        sorted_values = non_zero + zeros
        
        # Заполняем змейкой для градиента
        new_board = np.zeros_like(self.board)
        idx = 0
        for i in range(self.size):
            if i % 2 == 0:
                # Чётная строка: слева направо
                for j in range(self.size):
                    new_board[i, j] = sorted_values[idx]
                    idx += 1
            else:
                # Нечётная строка: справа налево
                for j in range(self.size - 1, -1, -1):
                    new_board[i, j] = sorted_values[idx]
                    idx += 1
        
        self.board = new_board
        self.sort_bonuses -= 1
        self.total_sort_bonuses_used += 1
        self._update_max_tile()
        
        return True
    
    def get_bonus_info(self) -> Dict:
        """Полная информация о бонусах"""
        next_bonus = None
        for t in self.BONUS_THRESHOLDS:
            if t not in self.claimed_bonuses:
                next_bonus = t
                break
        
        return {
            'remove_available': self.bonus_count,
            'sort_available': self.sort_bonuses,
            'total_earned': self.total_bonuses_earned,
            'total_used': self.total_bonuses_used,
            'total_combos': self.total_combos,
            'combo_tiles': self.combo_bonuses.copy(),
            'sort_earned': self.total_sort_bonuses_earned,
            'sort_used': self.total_sort_bonuses_used,
            'next_bonus_at': next_bonus,
            'claimed_thresholds': sorted(list(self.claimed_bonuses))
        }
    
    # ========================================================================
    # GAME MECHANICS
    # ========================================================================
    
    def _compress(self, row: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Сжатие строки влево с подсчётом очков.
        
        Очки = сумма значений новых тайлов после слияния.
        Это оригинальная формула подсчёта очков в 2048.
        """
        non_zero = row[row != 0]
        score = 0
        result = []
        
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                # Слияние: новый тайл = сумма
                merged = non_zero[i] * 2
                result.append(merged)
                score += merged  # Очки = значение нового тайла
                i += 2
            else:
                result.append(non_zero[i])
                i += 1
        
        # Дополняем нулями
        result.extend([0] * (self.size - len(result)))
        return np.array(result, dtype=np.int64), score
    
    def _move_left(self) -> Tuple[np.ndarray, int]:
        new_board = np.zeros_like(self.board)
        total_score = 0
        for i in range(self.size):
            new_board[i], score = self._compress(self.board[i])
            total_score += score
        return new_board, total_score
    
    def _move_right(self) -> Tuple[np.ndarray, int]:
        new_board = np.zeros_like(self.board)
        total_score = 0
        for i in range(self.size):
            new_board[i], score = self._compress(self.board[i][::-1])
            new_board[i] = new_board[i][::-1]
            total_score += score
        return new_board, total_score
    
    def _move_up(self) -> Tuple[np.ndarray, int]:
        transposed = self.board.T.copy()
        new_board = np.zeros_like(transposed)
        total_score = 0
        for i in range(self.size):
            new_board[i], score = self._compress(transposed[i])
            total_score += score
        return new_board.T, total_score
    
    def _move_down(self) -> Tuple[np.ndarray, int]:
        transposed = self.board.T.copy()
        new_board = np.zeros_like(transposed)
        total_score = 0
        for i in range(self.size):
            new_board[i], score = self._compress(transposed[i][::-1])
            new_board[i] = new_board[i][::-1]
            total_score += score
        return new_board.T, total_score
    
    def move(self, direction: Direction) -> Tuple[float, bool, Dict]:
        """
        Выполнение хода.
        
        Returns:
            reward: Награда для AI
            done: Игра окончена
            info: Информация о ходе
        """
        old_board = self.board.copy()
        old_record = self.record
        
        # Выполняем движение
        if direction == Direction.UP:
            new_board, move_score = self._move_up()
        elif direction == Direction.DOWN:
            new_board, move_score = self._move_down()
        elif direction == Direction.LEFT:
            new_board, move_score = self._move_left()
        else:
            new_board, move_score = self._move_right()
        
        moved = not np.array_equal(old_board, new_board)
        new_record = False
        combo_triggered = False
        
        if moved:
            self.board = new_board
            self.score += move_score
            self._spawn_tile()
            
            old_max = self.max_tile
            self._update_max_tile()
            
            self.moves += 1
            self.history.append(old_board.copy())
            
            new_record = self.record > old_record
            
            # Проверяем, был ли combo
            if new_record and len(self.record_events) > 0:
                combo_triggered = self.record_events[-1].is_combo
        
        done = self.is_game_over()
        reward = self._calculate_reward(moved, move_score, old_board, done)
        
        info = {
            'moved': moved,
            'score': self.score,
            'move_score': move_score,
            'max_tile': self.max_tile,
            'record': self.record,
            'moves': self.moves,
            'new_record': new_record,
            'combo_triggered': combo_triggered,
            'min_tile': self.get_min_tile(),
            'spawn_tiles': self.get_spawn_tiles(),
            'bonus_count': self.bonus_count,
            'sort_bonuses': self.sort_bonuses,
            'mode': self.mode
        }
        
        return reward, done, info
    
    def _calculate_reward(self, moved: bool, merge_score: int,
                          old_board: np.ndarray, done: bool) -> float:
        """Система наград для AI"""
        if not moved:
            return -10.0
        
        reward = 0.0
        
        # Награда за слияние
        if merge_score > 0:
            reward += np.log2(merge_score + 1) * 2.0
        
        # Бонус за пустые клетки
        empty = np.sum(self.board == 0)
        reward += empty * 0.5
        
        # Бонус за угловое расположение max tile
        max_val = np.max(self.board)
        corners = [
            self.board[0, 0], self.board[0, -1],
            self.board[-1, 0], self.board[-1, -1]
        ]
        if max_val in corners:
            reward += np.log2(max_val + 1) * 1.5
        
        # Бонус за монотонность
        reward += self._monotonicity_score() * 0.3
        
        # Штраф за проигрыш
        if done:
            reward -= 50.0
        
        return reward
    
    def _monotonicity_score(self) -> float:
        """Оценка упорядоченности доски"""
        score = 0.0
        
        for row in self.board:
            non_zero = row[row > 0]
            if len(non_zero) > 1:
                if all(non_zero[i] <= non_zero[i+1] for i in range(len(non_zero)-1)):
                    score += 1.0
                elif all(non_zero[i] >= non_zero[i+1] for i in range(len(non_zero)-1)):
                    score += 1.0
        
        for col in self.board.T:
            non_zero = col[col > 0]
            if len(non_zero) > 1:
                if all(non_zero[i] <= non_zero[i+1] for i in range(len(non_zero)-1)):
                    score += 1.0
                elif all(non_zero[i] >= non_zero[i+1] for i in range(len(non_zero)-1)):
                    score += 1.0
        
        return score
    
    def is_game_over(self) -> bool:
        """Проверка окончания игры"""
        if np.any(self.board == 0):
            return False
        
        # Проверяем возможные слияния
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j + 1]:
                    return False
        
        for i in range(self.size - 1):
            for j in range(self.size):
                if self.board[i, j] == self.board[i + 1, j]:
                    return False
        
        return True
    
    def get_valid_moves(self) -> List[Direction]:
        """Список допустимых ходов"""
        valid = []
        for direction in Direction:
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
        """Нормализованное состояние для нейросети"""
        state = np.zeros_like(self.board, dtype=np.float32)
        mask = self.board > 0
        state[mask] = np.log2(self.board[mask]) / 20.0  # Нормализация до ~1
        return state
    
    def get_features(self) -> np.ndarray:
        """Дополнительные признаки для нейросети"""
        features = []
        
        # Пустые клетки
        features.append(np.sum(self.board == 0) / 16.0)
        
        # Max tile
        max_tile = np.max(self.board)
        features.append(np.log2(max_tile + 1) / 20.0 if max_tile > 0 else 0)
        
        # Монотонность
        features.append(self._monotonicity_score() / 8.0)
        
        # Max в углу
        corners = [self.board[0,0], self.board[0,-1], self.board[-1,0], self.board[-1,-1]]
        features.append(1.0 if max_tile in corners else 0.0)
        
        # Валидные ходы
        valid = self.get_valid_moves()
        for d in Direction:
            features.append(1.0 if d in valid else 0.0)
        
        # Бонусы (если infinite)
        features.append(min(self.bonus_count / 5.0, 1.0))
        
        return np.array(features, dtype=np.float32)
    
    def copy(self) -> 'Game2048':
        """Копия игры"""
        game = Game2048(self.size, self.mode)
        game.board = self.board.copy()
        game.score = self.score
        game.moves = self.moves
        game.max_tile = self.max_tile
        game.record = self.record
        game.bonus_count = self.bonus_count
        game.claimed_bonuses = self.claimed_bonuses.copy()
        game.sort_bonuses = self.sort_bonuses
        game.last_record_move = self.last_record_move
        game.combo_bonuses = self.combo_bonuses.copy()
        return game
    
    def __str__(self) -> str:
        """Строковое представление"""
        min_t = self.get_min_tile()
        spawn = f"{min_t}/{min_t*2}" if self.mode != 'classic' else "2/4"
        
        bonus_str = ""
        if self.mode == 'infinite':
            bonus_str = f" | 🎁{self.bonus_count}"
            if self.sort_bonuses > 0:
                bonus_str += f" ⚡{self.sort_bonuses}"
        
        combo_str = f" | 🔥{self.total_combos}" if self.total_combos > 0 else ""
        
        lines = [
            f"Score: {self.score:,} | Max: {self.max_tile:,} | "
            f"Moves: {self.moves} | Spawn: {spawn}{bonus_str}{combo_str}"
        ]
        lines.append("─" * 50)
        
        for row in self.board:
            line = "│"
            for val in row:
                if val == 0:
                    line += "      ·│"
                else:
                    line += f"{val:>7,}│"
            lines.append(line)
        
        lines.append("─" * 50)
        return "\n".join(lines)


# ============================================================================
# DEMO
# ============================================================================

def demo_score_system():
    """Демонстрация системы очков"""
    print("=" * 60)
    print("СИСТЕМА ОЧКОВ (ОРИГИНАЛЬНАЯ)")
    print("=" * 60)
    print()
    print("При слиянии двух плиток:")
    print("  2 + 2 = 4   →  +4 очка")
    print("  4 + 4 = 8   →  +8 очков")
    print("  8 + 8 = 16  →  +16 очков")
    print("  ...")
    print("  1024 + 1024 = 2048  →  +2048 очков")
    print()


def demo_combo_system():
    """Демонстрация системы combo"""
    print("=" * 60)
    print("СИСТЕМА RECORD COMBO")
    print("=" * 60)
    print()
    print("Record Combo срабатывает когда:")
    print("  Новый рекорд установлен в течение 2 ходов после предыдущего")
    print()
    print("Награды за combo:")
    print("  256-combo   → +1 обычный бонус")
    print("  512-combo   → +1 обычный бонус")
    print("  1024-combo  → +1 обычный бонус")
    print("  2048-combo  → +1 СУПЕР-БОНУС СОРТИРОВКИ ⚡")
    print("  4096-combo  → +1 СУПЕР-БОНУС СОРТИРОВКИ ⚡")
    print("  ...")
    print()
    print("Супер-бонус сортировки:")
    print("  Автоматически сортирует все тайлы:")
    print("  • Наибольший → левый верхний угол")
    print("  • Наименьший → правый нижний угол")
    print("  • Змейкообразный градиент")
    print("  • Пустые клетки концентрируются справа внизу")
    print()


def demo_sort_bonus():
    """Демонстрация супер-бонуса сортировки"""
    print("=" * 60)
    print("ДЕМО: СУПЕР-БОНУС СОРТИРОВКИ")
    print("=" * 60)
    
    game = Game2048(mode='infinite')
    
    # Создаём хаотичную доску
    game.board = np.array([
        [4, 32, 2, 8],
        [256, 2, 64, 4],
        [16, 128, 8, 2],
        [2, 4, 16, 32]
    ], dtype=np.int64)
    game.sort_bonuses = 1
    game._update_max_tile()
    
    print("\nДО сортировки:")
    print(game)
    
    game.use_sort_bonus()
    
    print("\nПОСЛЕ сортировки:")
    print(game)
    print()


if __name__ == "__main__":
    demo_score_system()
    demo_combo_system()
    demo_sort_bonus()
    
    print("=" * 60)
    print("ТЕСТ ИГРЫ")
    print("=" * 60)
    
    game = Game2048(mode='infinite')
    print("\nНачало игры:")
    print(game)
    
    # Несколько случайных ходов
    for i in range(10):
        valid = game.get_valid_moves()
        if not valid:
            break
        move = random.choice(valid)
        reward, done, info = game.move(move)
        
        if info.get('new_record'):
            print(f"\n🎯 Новый рекорд: {info['record']}!")
            if info.get('combo_triggered'):
                print("🔥 COMBO!")
    
    print(f"\nПосле 10 ходов:")
    print(game)

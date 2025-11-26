"""
Alpha2048 - AlphaZero-подобная архитектура для игры 2048
=========================================================

Оптимизировано для Apple Silicon (MPS) и бесконечного режима игры.

Ключевые особенности:
1. Policy + Value + Planning головы
2. MCTS с Chance Nodes для стохастичности спавна
3. Curriculum Learning для постепенного усложнения
4. Оптимизации под Apple Silicon MPS

Отличия от классического AlphaZero:
- Expectimax вместо minimax (случайный спавн)
- Chance nodes моделируют вероятности спавна тайлов
- Planning head предсказывает последствия каждого из 4 ходов
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict, Set
from dataclasses import dataclass, field
from collections import deque
import random
import math
import os
import time
from enum import IntEnum

from game_2048 import Game2048, Direction


# ============================================================================
# ОПТИМИЗАЦИЯ ПОД APPLE SILICON
# ============================================================================

def get_optimal_device() -> torch.device:
    """
    Выбор оптимального устройства с приоритетом Apple Silicon.
    
    Порядок приоритета:
    1. MPS (Apple Silicon M1/M2/M3/M4)
    2. CUDA (NVIDIA GPU)
    3. CPU
    """
    # Apple Silicon MPS
    if torch.backends.mps.is_available():
        try:
            # Проверяем работоспособность MPS
            test_tensor = torch.zeros(1, device='mps')
            del test_tensor
            return torch.device('mps')
        except Exception as e:
            print(f"MPS available but failed: {e}")
    
    # NVIDIA CUDA
    if torch.cuda.is_available():
        return torch.device('cuda')
    
    return torch.device('cpu')


def get_device_info() -> Dict:
    """Получение информации об устройстве"""
    device = get_optimal_device()
    info = {
        'device': str(device),
        'type': device.type,
    }
    
    if device.type == 'mps':
        info['name'] = 'Apple Silicon (MPS)'
        info['memory'] = 'Unified Memory'
    elif device.type == 'cuda':
        info['name'] = torch.cuda.get_device_name(0)
        info['memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    else:
        info['name'] = 'CPU'
        info['memory'] = 'System RAM'
    
    return info


# Глобальное устройство
DEVICE = get_optimal_device()


# ============================================================================
# АРХИТЕКТУРА НЕЙРОСЕТИ (оптимизированная для MPS)
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Residual блок с оптимизациями для MPS.
    Использует GroupNorm вместо BatchNorm для лучшей совместимости с MPS.
    """
    def __init__(self, channels: int, use_group_norm: bool = True):
        super().__init__()
        
        # GroupNorm лучше работает на MPS и с маленькими батчами
        if use_group_norm:
            norm_layer = lambda c: nn.GroupNorm(min(32, c), c)
        else:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = norm_layer(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = norm_layer(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.norm1(self.conv1(x)))  # GELU лучше чем ReLU
        out = self.norm2(self.conv2(out))
        return F.gelu(out + residual)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation блок для channel attention"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Global average pooling
        y = x.view(b, c, -1).mean(dim=2)
        y = F.gelu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1)


class Alpha2048Network(nn.Module):
    """
    Нейросеть Alpha2048 с тремя головами:
    - Policy: вероятности для 4 направлений
    - Value: оценка позиции [-1, 1]
    - Planning: прогноз для каждого из 4 ходов
    
    Оптимизирована для Apple Silicon MPS.
    """
    
    # Максимальное значение тайла (log2)
    MAX_TILE_POWER = 20  # До 2^20 = 1,048,576
    
    def __init__(
        self,
        board_size: int = 4,
        n_features: int = 9,
        n_channels: int = 128,
        n_residual_blocks: int = 6,
        use_se_blocks: bool = True
    ):
        super().__init__()
        
        self.board_size = board_size
        self.n_features = n_features
        self.n_channels = n_channels
        
        # One-hot encoding: 0 + log2(2..2^MAX_TILE_POWER) = MAX_TILE_POWER + 1 channels
        self.n_tile_types = self.MAX_TILE_POWER + 1
        
        # === Input Encoding ===
        self.input_conv = nn.Sequential(
            nn.Conv2d(self.n_tile_types, n_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, n_channels), n_channels),
            nn.GELU()
        )
        
        # === Residual Tower ===
        blocks = []
        for _ in range(n_residual_blocks):
            blocks.append(ResidualBlock(n_channels))
            if use_se_blocks:
                blocks.append(SEBlock(n_channels))
        self.residual_tower = nn.Sequential(*blocks)
        
        # === Feature Encoder ===
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU()
        )
        
        # Размер после conv: n_channels * 4 * 4 = n_channels * 16
        conv_output_size = n_channels * board_size * board_size
        combined_size = n_channels + 64  # После global pooling + features
        
        # === Policy Head ===
        self.policy_conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )
        self.policy_fc = nn.Sequential(
            nn.Linear(32 * board_size * board_size + 64, 128),
            nn.GELU(),
            nn.Linear(128, 4)
        )
        
        # === Value Head ===
        self.value_conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )
        self.value_fc = nn.Sequential(
            nn.Linear(32 * board_size * board_size + 64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        # === Planning Head ===
        # Для каждого из 4 ходов: [reward, future_value, confidence, valid_prob]
        self.planning_conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, 1, bias=False),
            nn.GroupNorm(16, 64),
            nn.GELU()
        )
        self.planning_fc = nn.Sequential(
            nn.Linear(64 * board_size * board_size + 64, 256),
            nn.GELU(),
            nn.Linear(256, 4 * 4)  # 4 moves × 4 predictions
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов для стабильного обучения"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode_board(self, board_values: torch.Tensor) -> torch.Tensor:
        """
        Эффективное one-hot кодирование доски.
        
        Args:
            board_values: [batch, 4, 4] с реальными значениями тайлов
        
        Returns:
            [batch, n_tile_types, 4, 4] one-hot encoding
        """
        batch_size = board_values.size(0)
        device = board_values.device
        
        # Вычисляем индексы каналов: 0 для пустых, log2(value) для остальных
        # Используем clamp для безопасности
        log_values = torch.zeros_like(board_values, dtype=torch.long)
        non_zero_mask = board_values > 0
        log_values[non_zero_mask] = torch.log2(
            board_values[non_zero_mask].float()
        ).long().clamp(0, self.MAX_TILE_POWER)
        
        # One-hot encoding через scatter
        encoded = torch.zeros(
            batch_size, self.n_tile_types, self.board_size, self.board_size,
            device=device
        )
        
        # Reshape для scatter: [batch, 1, 4, 4]
        indices = log_values.unsqueeze(1)
        encoded.scatter_(1, indices, 1.0)
        
        return encoded
    
    def forward(
        self,
        state: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: [batch, 4, 4] нормализованная доска (log2(value)/17)
            features: [batch, n_features] дополнительные признаки
        
        Returns:
            policy: [batch, 4] логиты действий
            value: [batch, 1] оценка позиции
            planning: [batch, 4, 4] прогнозы для каждого хода
        """
        batch_size = state.size(0)
        
        # Декодируем нормализованное состояние в значения тайлов
        # state * 17 = log2(value), 2^(state*17) = value
        board_values = torch.zeros_like(state, dtype=torch.long)
        mask = state > 0
        board_values[mask] = torch.pow(
            2, (state[mask] * 17).clamp(0, self.MAX_TILE_POWER)
        ).long()
        
        # One-hot encoding
        x = self.encode_board(board_values)
        
        # Convolutions
        x = self.input_conv(x)
        x = self.residual_tower(x)
        
        # Feature encoding
        feat = self.feature_encoder(features)
        
        # === Policy ===
        policy_x = self.policy_conv(x)
        policy_x = policy_x.view(batch_size, -1)
        policy_x = torch.cat([policy_x, feat], dim=1)
        policy = self.policy_fc(policy_x)
        
        # === Value ===
        value_x = self.value_conv(x)
        value_x = value_x.view(batch_size, -1)
        value_x = torch.cat([value_x, feat], dim=1)
        value = self.value_fc(value_x)
        
        # === Planning ===
        plan_x = self.planning_conv(x)
        plan_x = plan_x.view(batch_size, -1)
        plan_x = torch.cat([plan_x, feat], dim=1)
        planning_raw = self.planning_fc(plan_x)
        planning = planning_raw.view(batch_size, 4, 4)
        
        # Активации для planning:
        # [0] reward - as is
        # [1] future_value - tanh
        # [2] confidence - sigmoid
        # [3] valid_prob - sigmoid
        planning_out = torch.zeros_like(planning)
        planning_out[:, :, 0] = planning[:, :, 0]
        planning_out[:, :, 1] = torch.tanh(planning[:, :, 1])
        planning_out[:, :, 2] = torch.sigmoid(planning[:, :, 2])
        planning_out[:, :, 3] = torch.sigmoid(planning[:, :, 3])
        
        return policy, value, planning_out


# ============================================================================
# MCTS С CHANCE NODES (Expectimax)
# ============================================================================

class NodeType(IntEnum):
    """Типы узлов в дереве MCTS"""
    DECISION = 0   # Узел выбора действия игрока
    CHANCE = 1     # Узел случайности (спавн тайла)


class MCTSNode:
    """Узел дерева MCTS с поддержкой Chance Nodes"""
    
    __slots__ = [
        'node_type', 'state', 'features', 'parent', 'action',
        'visit_count', 'value_sum', 'prior', 'children', 'spawn_probability',
        '_id'
    ]
    
    _counter = 0
    
    def __init__(
        self,
        node_type: NodeType,
        state: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        parent: Optional['MCTSNode'] = None,
        action = None,
        prior: float = 0.0,
        spawn_probability: float = 1.0
    ):
        self.node_type = node_type
        self.state = state
        self.features = features
        self.parent = parent
        self.action = action
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children: Dict = {}
        self.spawn_probability = spawn_probability
        
        # Уникальный ID для сравнения
        MCTSNode._counter += 1
        self._id = MCTSNode._counter
    
    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visits: int, c_puct: float = 2.0) -> float:
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.q_value
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return exploitation + exploration
    
    def __eq__(self, other):
        if not isinstance(other, MCTSNode):
            return False
        return self._id == other._id
    
    def __hash__(self):
        return hash(self._id)


class Alpha2048MCTS:
    """
    MCTS с Chance Nodes для правильной обработки стохастичности.
    
    Структура дерева:
    DECISION (выбор хода) -> CHANCE (спавн тайла) -> DECISION -> ...
    
    Chance nodes моделируют:
    - Позицию спавна (все пустые клетки равновероятны)
    - Значение тайла (90% minTile, 10% minTile*2)
    """
    
    def __init__(
        self,
        network: Alpha2048Network,
        n_simulations: int = 100,
        c_puct: float = 2.0,
        device: torch.device = DEVICE,
        # Оптимизации для скорости
        max_chance_samples: int = 4,  # Макс. сэмплов chance outcomes
        use_virtual_loss: bool = True
    ):
        self.network = network
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.device = device
        self.max_chance_samples = max_chance_samples
        self.use_virtual_loss = use_virtual_loss
    
    @torch.no_grad()
    def get_network_prediction(
        self,
        state: np.ndarray,
        features: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Получение предсказания от сети"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        features_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        policy_logits, value, planning = self.network(state_t, features_t)
        
        policy = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
        value = value.squeeze().item()
        planning = planning.squeeze().cpu().numpy()
        
        return policy, value, planning
    
    def search(self, game: Game2048) -> Tuple[np.ndarray, Dict]:
        """
        Выполнение MCTS поиска.
        
        Returns:
            action_probs: [4] распределение по ходам
            info: дополнительная информация
        """
        # Создаём корневой узел (DECISION)
        root = MCTSNode(
            node_type=NodeType.DECISION,
            state=game.get_state(),
            features=game.get_features()
        )
        
        # Получаем prior от сети
        policy, root_value, planning = self.get_network_prediction(root.state, root.features)
        
        # Инициализируем детей для валидных ходов
        valid_moves = game.get_valid_moves()
        valid_mask = np.zeros(4)
        for m in valid_moves:
            m_int = m.value if isinstance(m, Direction) else m
            valid_mask[m_int] = 1.0
        
        if valid_mask.sum() == 0:
            # Игра окончена
            return np.zeros(4), {'value': -1.0, 'game_over': True}
        
        masked_policy = policy * valid_mask
        masked_policy /= masked_policy.sum()
        
        for m in valid_moves:
            action = m.value if isinstance(m, Direction) else m
            root.children[action] = MCTSNode(
                node_type=NodeType.DECISION,  # После chance станет DECISION
                parent=root,
                action=action,
                prior=masked_policy[action]
            )
        
        # Выполняем симуляции
        for _ in range(self.n_simulations):
            self._simulate(root, game.copy())
        
        # Формируем распределение
        action_probs = np.zeros(4)
        total_visits = sum(c.visit_count for c in root.children.values())
        
        if total_visits > 0:
            for action, child in root.children.items():
                action_probs[action] = child.visit_count / total_visits
        
        info = {
            'value': root_value,
            'planning': planning,
            'visit_counts': {a: c.visit_count for a, c in root.children.items()},
            'q_values': {a: c.q_value for a, c in root.children.items()},
            'game_over': False
        }
        
        return action_probs, info
    
    def _simulate(self, root: MCTSNode, game: Game2048):
        """Одна симуляция MCTS с Chance Nodes"""
        node = root
        path = [node]
        
        # === Selection ===
        while node.children:
            if node.node_type == NodeType.DECISION:
                # Выбираем по UCB
                best_action = max(
                    node.children.keys(),
                    key=lambda a: node.children[a].ucb_score(
                        node.visit_count + 1, self.c_puct
                    )
                )
                child = node.children[best_action]
                
                # Применяем virtual loss
                if self.use_virtual_loss:
                    child.visit_count += 1
                    child.value_sum -= 1.0
                
                # Делаем ход
                if child.state is None:
                    # Ход ещё не был выполнен
                    reward, done, info = game.move(Direction(best_action))
                    if not info['moved']:
                        # Ход невозможен - удаляем
                        del node.children[best_action]
                        continue
                    
                    if done:
                        # Игра окончена
                        child.state = game.get_state()
                        child.features = game.get_features()
                        node = child
                        path.append(node)
                        break
                    
                    # Создаём Chance Node
                    child.node_type = NodeType.CHANCE
                    child.state = game.get_state()
                    child.features = game.get_features()
                else:
                    # Уже посещённый узел
                    game.board = self._decode_state(child.state, game)
                
                node = child
                path.append(node)
            
            else:  # CHANCE node
                # Сэмплируем исход спавна
                outcome = self._sample_chance_outcome(game)
                
                if outcome in node.children:
                    child = node.children[outcome]
                else:
                    # Создаём новый узел
                    row, col, value = outcome
                    game.board[row, col] = value
                    game._update_max_tile()
                    
                    spawn_prob = self._get_spawn_probability(game, outcome)
                    
                    child = MCTSNode(
                        node_type=NodeType.DECISION,
                        state=game.get_state(),
                        features=game.get_features(),
                        parent=node,
                        action=outcome,
                        spawn_probability=spawn_prob
                    )
                    node.children[outcome] = child
                
                node = child
                path.append(node)
        
        # === Expansion & Evaluation ===
        if not game.is_game_over():
            if node.state is None:
                node.state = game.get_state()
                node.features = game.get_features()
            
            policy, value, _ = self.get_network_prediction(node.state, node.features)
            
            # Создаём детей для DECISION узла
            if node.node_type == NodeType.DECISION and not node.children:
                valid_moves = game.get_valid_moves()
                if valid_moves:
                    valid_mask = np.zeros(4)
                    for m in valid_moves:
                        m_int = m.value if isinstance(m, Direction) else m
                        valid_mask[m_int] = 1.0
                    
                    masked_policy = policy * valid_mask
                    if masked_policy.sum() > 0:
                        masked_policy /= masked_policy.sum()
                    
                    for m in valid_moves:
                        action = m.value if isinstance(m, Direction) else m
                        node.children[action] = MCTSNode(
                            node_type=NodeType.DECISION,
                            parent=node,
                            action=action,
                            prior=masked_policy[action]
                        )
        else:
            value = -1.0
        
        # === Backpropagation ===
        for n in reversed(path):
            # Убираем virtual loss
            if self.use_virtual_loss and n != root:
                n.visit_count -= 1
                n.value_sum += 1.0
            
            n.visit_count += 1
            n.value_sum += value
            
            # Для chance nodes взвешиваем по вероятности
            if n.node_type == NodeType.CHANCE:
                value *= n.spawn_probability
    
    def _sample_chance_outcome(self, game: Game2048) -> Tuple[int, int, int]:
        """Сэмплирование исхода спавна"""
        empty_cells = list(zip(*np.where(game.board == 0)))
        if not empty_cells:
            return (0, 0, 0)
        
        row, col = random.choice(empty_cells)
        spawn_tiles = game.get_spawn_tiles()
        value = spawn_tiles[1] if random.random() < 0.1 else spawn_tiles[0]
        
        return (row, col, value)
    
    def _get_spawn_probability(
        self, 
        game: Game2048, 
        outcome: Tuple[int, int, int]
    ) -> float:
        """Вычисление вероятности конкретного исхода спавна"""
        row, col, value = outcome
        empty_cells = np.sum(game.board == 0)
        
        if empty_cells == 0:
            return 0.0
        
        spawn_tiles = game.get_spawn_tiles()
        value_prob = 0.9 if value == spawn_tiles[0] else 0.1
        position_prob = 1.0 / empty_cells
        
        return value_prob * position_prob
    
    def _decode_state(self, state: np.ndarray, game: Game2048) -> np.ndarray:
        """Декодирование состояния обратно в доску"""
        board = np.zeros_like(game.board)
        mask = state > 0
        board[mask] = np.power(2, state[mask] * 17).astype(np.int32)
        return board


# ============================================================================
# CURRICULUM LEARNING
# ============================================================================

class CurriculumStage:
    """Этап curriculum learning"""
    def __init__(
        self,
        name: str,
        target_tile: int,
        min_games: int,
        success_rate: float,
        game_mode: str = 'infinite',
        mcts_simulations: int = 50
    ):
        self.name = name
        self.target_tile = target_tile
        self.min_games = min_games
        self.success_rate = success_rate
        self.game_mode = game_mode
        self.mcts_simulations = mcts_simulations
        
        # Статистика
        self.games_played = 0
        self.successes = 0
        self.recent_results = deque(maxlen=100)
    
    def record_game(self, max_tile: int):
        """Записать результат игры"""
        self.games_played += 1
        success = max_tile >= self.target_tile
        if success:
            self.successes += 1
        self.recent_results.append(success)
    
    @property
    def current_success_rate(self) -> float:
        if len(self.recent_results) == 0:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)
    
    def is_complete(self) -> bool:
        """Проверка завершения этапа"""
        return (
            self.games_played >= self.min_games and
            self.current_success_rate >= self.success_rate
        )


class CurriculumManager:
    """
    Менеджер Curriculum Learning для постепенного усложнения.
    
    Этапы:
    1. Достичь 256 (базовый)
    2. Достичь 512 (продвинутый)
    3. Достичь 1024 (эксперт)
    4. Достичь 2048 (мастер)
    5. Достичь 4096+ (бесконечность)
    """
    
    DEFAULT_CURRICULUM = [
        CurriculumStage("Beginner", 256, 100, 0.7, mcts_simulations=20),
        CurriculumStage("Intermediate", 512, 200, 0.5, mcts_simulations=30),
        CurriculumStage("Advanced", 1024, 300, 0.4, mcts_simulations=50),
        CurriculumStage("Expert", 2048, 500, 0.3, mcts_simulations=75),
        CurriculumStage("Master", 4096, 1000, 0.2, mcts_simulations=100),
        CurriculumStage("Infinite", 8192, 2000, 0.1, mcts_simulations=100),
    ]
    
    def __init__(self, stages: List[CurriculumStage] = None):
        self.stages = stages or self.DEFAULT_CURRICULUM.copy()
        self.current_stage_idx = 0
    
    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[min(self.current_stage_idx, len(self.stages) - 1)]
    
    def record_game(self, max_tile: int) -> bool:
        """
        Записать результат игры.
        
        Returns:
            True если перешли на новый этап
        """
        stage = self.current_stage
        stage.record_game(max_tile)
        
        if stage.is_complete() and self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            return True
        
        return False
    
    def get_training_config(self) -> Dict:
        """Получить конфигурацию обучения для текущего этапа"""
        stage = self.current_stage
        return {
            'stage_name': stage.name,
            'target_tile': stage.target_tile,
            'mcts_simulations': stage.mcts_simulations,
            'game_mode': stage.game_mode,
            'progress': f"{stage.current_success_rate:.1%}",
            'games': stage.games_played
        }
    
    def get_status(self) -> str:
        """Получить статус curriculum"""
        stage = self.current_stage
        return (
            f"Stage {self.current_stage_idx + 1}/{len(self.stages)}: {stage.name}\n"
            f"  Target: {stage.target_tile}\n"
            f"  Progress: {stage.current_success_rate:.1%} / {stage.success_rate:.1%}\n"
            f"  Games: {stage.games_played} / {stage.min_games}"
        )


# ============================================================================
# АГЕНТ
# ============================================================================

class Alpha2048Agent:
    """
    Агент Alpha2048 с:
    - Self-play обучением
    - Curriculum learning
    - Оптимизацией для Apple Silicon
    """
    
    def __init__(
        self,
        board_size: int = 4,
        n_features: int = 9,
        n_channels: int = 128,
        n_residual_blocks: int = 6,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        mcts_simulations: int = 50,
        buffer_size: int = 100000,
        batch_size: int = 256,
        use_curriculum: bool = True,
        device: torch.device = None
    ):
        self.device = device or DEVICE
        self.batch_size = batch_size
        self.mcts_simulations = mcts_simulations
        
        # Сеть
        self.network = Alpha2048Network(
            board_size=board_size,
            n_features=n_features,
            n_channels=n_channels,
            n_residual_blocks=n_residual_blocks
        ).to(self.device)
        
        # Оптимизатор с cosine annealing
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )
        
        # Replay buffer
        self.buffer = deque(maxlen=buffer_size)
        
        # Curriculum
        self.use_curriculum = use_curriculum
        self.curriculum = CurriculumManager() if use_curriculum else None
        
        # Статистика
        self.training_steps = 0
        self.games_played = 0
        self.best_tile = 0
        self.best_score = 0
    
    def self_play(
        self,
        game_mode: str = 'infinite',
        temperature: float = 1.0,
        temperature_drop_move: int = 30
    ) -> Tuple[List[Dict], Dict]:
        """
        Self-play для сбора данных обучения.
        
        Returns:
            trajectory: список позиций для обучения
            stats: статистика игры
        """
        # Используем mcts_simulations из curriculum если включен
        n_sims = self.mcts_simulations
        if self.curriculum:
            n_sims = self.curriculum.current_stage.mcts_simulations
        
        game = Game2048(mode=game_mode)
        mcts = Alpha2048MCTS(
            self.network,
            n_simulations=n_sims,
            device=self.device
        )
        
        trajectory = []
        move_count = 0
        
        while not game.is_game_over():
            # Temperature scheduling
            temp = temperature if move_count < temperature_drop_move else 0.1
            
            # MCTS поиск
            action_probs, info = mcts.search(game)
            
            if info.get('game_over', False):
                break
            
            # Сохраняем данные
            trajectory.append({
                'state': game.get_state().copy(),
                'features': game.get_features().copy(),
                'policy': action_probs.copy(),
                'planning': info['planning'].copy()
            })
            
            # Выбор действия
            if temp > 0:
                # Добавляем Dirichlet noise для exploration
                noise = np.random.dirichlet([0.3] * 4)
                probs = 0.75 * action_probs + 0.25 * noise
                probs = np.power(probs, 1.0 / temp)
                probs /= probs.sum()
                action = np.random.choice(4, p=probs)
            else:
                action = np.argmax(action_probs)
            
            # Выполняем ход
            reward, done, _ = game.move(Direction(action))
            move_count += 1
            
            if done:
                break
        
        # Вычисляем value targets
        final_value = -1.0 if game.is_game_over() else 0.0
        gamma = 0.99
        
        for i in range(len(trajectory) - 1, -1, -1):
            trajectory[i]['value'] = final_value
            final_value = gamma * final_value
        
        # Статистика
        stats = {
            'score': game.score,
            'max_tile': game.max_tile,
            'moves': game.moves,
            'record': game.record,
            'bonuses_earned': game.total_bonuses_earned
        }
        
        # Обновляем лучшие результаты
        if game.max_tile > self.best_tile:
            self.best_tile = game.max_tile
        if game.score > self.best_score:
            self.best_score = game.score
        
        # Curriculum
        if self.curriculum:
            advanced = self.curriculum.record_game(game.max_tile)
            stats['curriculum_advanced'] = advanced
            stats['curriculum_stage'] = self.curriculum.current_stage.name
        
        self.games_played += 1
        
        return trajectory, stats
    
    def add_to_buffer(self, trajectory: List[Dict]):
        """Добавление траектории в буфер"""
        for data in trajectory:
            self.buffer.append(data)
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """Один шаг обучения"""
        if len(self.buffer) < self.batch_size:
            return None
        
        # Выборка батча
        batch = random.sample(list(self.buffer), self.batch_size)
        
        states = torch.FloatTensor(
            np.array([d['state'] for d in batch])
        ).to(self.device)
        features = torch.FloatTensor(
            np.array([d['features'] for d in batch])
        ).to(self.device)
        target_policies = torch.FloatTensor(
            np.array([d['policy'] for d in batch])
        ).to(self.device)
        target_values = torch.FloatTensor(
            np.array([d['value'] for d in batch])
        ).to(self.device)
        target_planning = torch.FloatTensor(
            np.array([d['planning'] for d in batch])
        ).to(self.device)
        
        # Forward
        policy_logits, values, planning = self.network(states, features)
        
        # Losses
        policy_loss = F.cross_entropy(policy_logits, target_policies)
        value_loss = F.mse_loss(values.squeeze(), target_values)
        planning_loss = F.mse_loss(planning, target_planning)
        
        total_loss = policy_loss + value_loss + 0.5 * planning_loss
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        self.training_steps += 1
        
        return {
            'total': total_loss.item(),
            'policy': policy_loss.item(),
            'value': value_loss.item(),
            'planning': planning_loss.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def select_action(
        self,
        game: Game2048,
        use_mcts: bool = True,
        temperature: float = 0.0
    ) -> Tuple[int, Dict]:
        """Выбор действия для игры"""
        if use_mcts:
            mcts = Alpha2048MCTS(
                self.network,
                n_simulations=self.mcts_simulations,
                device=self.device
            )
            action_probs, info = mcts.search(game)
            
            if temperature == 0:
                action = np.argmax(action_probs)
            else:
                probs = np.power(action_probs + 1e-8, 1.0 / temperature)
                probs /= probs.sum()
                action = np.random.choice(4, p=probs)
            
            return action, info
        else:
            # Быстрый режим без MCTS
            with torch.no_grad():
                state = torch.FloatTensor(
                    game.get_state()
                ).unsqueeze(0).to(self.device)
                features = torch.FloatTensor(
                    game.get_features()
                ).unsqueeze(0).to(self.device)
                
                policy_logits, value, planning = self.network(state, features)
                policy = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
                
                valid_moves = game.get_valid_moves()
                mask = np.zeros(4)
                for m in valid_moves:
                    mask[m.value if isinstance(m, Direction) else m] = 1.0
                
                masked_policy = policy * mask
                if masked_policy.sum() > 0:
                    masked_policy /= masked_policy.sum()
                
                action = np.argmax(masked_policy)
                
                return action, {
                    'value': value.item(),
                    'policy': masked_policy,
                    'planning': planning.squeeze().cpu().numpy()
                }
    
    def get_planning_visualization(self, game: Game2048) -> Dict:
        """Визуализация планирования"""
        with torch.no_grad():
            state = torch.FloatTensor(
                game.get_state()
            ).unsqueeze(0).to(self.device)
            features = torch.FloatTensor(
                game.get_features()
            ).unsqueeze(0).to(self.device)
            
            policy_logits, value, planning = self.network(state, features)
            policy = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
            planning = planning.squeeze().cpu().numpy()
        
        valid_moves = set(
            m.value if isinstance(m, Direction) else m 
            for m in game.get_valid_moves()
        )
        
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        result = {
            'current_value': value.item(),
            'moves': {}
        }
        
        for i, name in enumerate(directions):
            result['moves'][name] = {
                'policy': float(policy[i]),
                'reward': float(planning[i, 0]),
                'future_value': float(planning[i, 1]),
                'confidence': float(planning[i, 2]),
                'valid_prob': float(planning[i, 3]),
                'is_valid': i in valid_moves
            }
        
        return result
    
    def save(self, path: str):
        """Сохранение модели"""
        state = {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'training_steps': self.training_steps,
            'games_played': self.games_played,
            'best_tile': self.best_tile,
            'best_score': self.best_score,
        }
        
        if self.curriculum:
            state['curriculum_stage'] = self.curriculum.current_stage_idx
        
        torch.save(state, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str) -> bool:
        """Загрузка модели"""
        if not os.path.exists(path):
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        self.training_steps = checkpoint.get('training_steps', 0)
        self.games_played = checkpoint.get('games_played', 0)
        self.best_tile = checkpoint.get('best_tile', 0)
        self.best_score = checkpoint.get('best_score', 0)
        
        if self.curriculum and 'curriculum_stage' in checkpoint:
            self.curriculum.current_stage_idx = checkpoint['curriculum_stage']
        
        print(f"✓ Model loaded from {path}")
        print(f"  Steps: {self.training_steps}, Games: {self.games_played}")
        print(f"  Best: {self.best_score} (tile: {self.best_tile})")
        
        return True


# ============================================================================
# ДЕМОНСТРАЦИЯ
# ============================================================================

def demo():
    """Демонстрация Alpha2048"""
    print("=" * 70)
    print("ALPHA2048 - AlphaZero для 2048")
    print("=" * 70)
    
    # Информация об устройстве
    info = get_device_info()
    print(f"\n🖥️  Device: {info['name']} ({info['device']})")
    print(f"   Memory: {info['memory']}")
    
    # Создаём агента
    agent = Alpha2048Agent(
        n_channels=64,
        n_residual_blocks=3,
        mcts_simulations=20,
        use_curriculum=True
    )
    
    params = sum(p.numel() for p in agent.network.parameters())
    print(f"\n📊 Network: {params:,} parameters")
    
    if agent.curriculum:
        print(f"\n📚 Curriculum:")
        print(agent.curriculum.get_status())
    
    # Игра
    game = Game2048(mode='infinite')
    for _ in range(30):
        valid = game.get_valid_moves()
        if not valid:
            break
        game.move(random.choice(valid))
    
    print(f"\n🎮 Game State:")
    print(game)
    
    # Планирование
    planning = agent.get_planning_visualization(game)
    print(f"\n🧠 Planning (value: {planning['current_value']:.3f}):")
    print("-" * 60)
    print(f"{'Move':<8} | {'Valid':<6} | {'Policy':<8} | {'Reward':<8} | {'Future':<8}")
    print("-" * 60)
    
    for name, info in planning['moves'].items():
        valid = "✓" if info['is_valid'] else "✗"
        print(f"{name:<8} | {valid:<6} | {info['policy']:.4f}  | "
              f"{info['reward']:>7.2f} | {info['future_value']:>7.3f}")
    
    # MCTS
    print("\n🔍 MCTS Search...")
    action, mcts_info = agent.select_action(game, use_mcts=True)
    
    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    print(f"   Best move: {directions[action]}")
    print(f"   Visits: {mcts_info.get('visit_counts', {})}")


def train_demo(n_games: int = 5, n_steps: int = 20):
    """Демонстрация обучения"""
    print("\n" + "=" * 70)
    print("TRAINING DEMO")
    print("=" * 70)
    
    agent = Alpha2048Agent(
        n_channels=32,
        n_residual_blocks=2,
        mcts_simulations=10,
        batch_size=32,
        use_curriculum=True
    )
    
    print(f"\n🎮 Self-play ({n_games} games)...")
    
    for i in range(n_games):
        trajectory, stats = agent.self_play(temperature=1.0)
        agent.add_to_buffer(trajectory)
        
        stage = stats.get('curriculum_stage', 'N/A')
        print(f"   Game {i+1}: {stats['moves']} moves, "
              f"max={stats['max_tile']}, score={stats['score']} "
              f"[{stage}]")
    
    print(f"\n📚 Buffer: {len(agent.buffer)} positions")
    
    print(f"\n🎓 Training ({n_steps} steps)...")
    for step in range(n_steps):
        losses = agent.train_step()
        if losses and step % 5 == 0:
            print(f"   Step {step+1}: loss={losses['total']:.4f} "
                  f"(p={losses['policy']:.3f}, v={losses['value']:.3f})")
    
    print("\n✅ Done!")
    
    if agent.curriculum:
        print(f"\n📊 Curriculum Status:")
        print(agent.curriculum.get_status())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_demo()
    else:
        demo()

"""
Креативная нейросетевая архитектура для 2048
Использует:
- Attention механизм для анализа паттернов
- Dueling DQN для эффективного обучения
- Convolutional слои для пространственных паттернов
- Residual connections для глубокого обучения
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import random
from collections import deque, namedtuple
import os
from models import SimpleDQN, ConvDQN, DuelingDQN, HybridDQN

# Определяем устройство с поддержкой Apple Silicon (MPS)
def get_device():
    """
    Выбор лучшего доступного устройства:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon - M1/M2/M3)
    3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Проверяем, что MPS действительно работает
        try:
            torch.zeros(1).to(torch.device("mps"))
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")

device = get_device()

# Структура для хранения опыта
Experience = namedtuple('Experience', 
    ['state', 'features', 'action', 'reward', 'next_state', 'next_features', 'done'])



class PrioritizedReplayBuffer:
    """
    Priority Experience Replay Buffer
    Приоритизирует опыт на основе TD-ошибки для более эффективного обучения
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Степень приоритизации
        self.beta = beta    # Importance sampling
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, experience: Experience):
        """Добавление опыта с максимальным приоритетом"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Выборка с приоритетами
        Returns: (experiences, indices, importance_weights)
        """
        n = len(self.buffer)
        
        # Вычисляем вероятности выборки
        priorities = self.priorities[:n]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Выбираем индексы
        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        
        # Importance sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Увеличиваем beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        experiences = [self.buffer[i] for i in indices]
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Обновление приоритетов на основе TD-ошибки"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6  # Добавляем epsilon для стабильности
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Агент с Double DQN и Priority Experience Replay
    """
    
    def __init__(
        self,
        board_size: int = 4,
        n_features: int = 9,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 50000,
        model_type: str = "dueling"
    ):
        self.board_size = board_size
        self.n_features = n_features
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.model_type = model_type
        
        # Epsilon scheduling
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Network Factory
        def create_model(m_type):
            if m_type == "simple": return SimpleDQN(board_size, n_features)
            if m_type == "conv": return ConvDQN(board_size, n_features)
            if m_type == "hybrid": return HybridDQN(board_size, n_features)
            return DuelingDQN(board_size, n_features) # Default

        # Networks
        self.policy_net = create_model(model_type).to(device)
        self.target_net = create_model(model_type).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.95)
        
        # Replay buffer
        self.memory = PrioritizedReplayBuffer(buffer_size)
        
        # Training stats
        self.steps = 0
        self.training_losses = []
        self.episode_rewards = []
        self.episode_scores = []
        self.max_tiles = []
    
    def get_epsilon(self) -> float:
        """Получение текущего epsilon"""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps / self.epsilon_decay)
    
    def select_action(self, state: np.ndarray, features: np.ndarray, 
                      valid_moves: List[int], epsilon: Optional[float] = None) -> int:
        """Выбор действия"""
        if epsilon is None:
            epsilon = self.get_epsilon()
        
        if random.random() < epsilon:
            return random.choice(valid_moves)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            features_t = torch.FloatTensor(features).unsqueeze(0).to(device)
            # Handle different model signatures if needed (but they all match now)
            q_values = self.policy_net(state_t, features_t).squeeze()
            
            masked_q = torch.full((4,), float('-inf'), device=device)
            for move in valid_moves:
                masked_q[move] = q_values[move]
            
            return masked_q.argmax().item()
    
    def store_experience(self, state: np.ndarray, features: np.ndarray,
                         action: int, reward: float, 
                         next_state: np.ndarray, next_features: np.ndarray,
                         done: bool):
        """Сохранение опыта в буфер"""
        exp = Experience(state, features, action, reward, next_state, next_features, done)
        self.memory.push(exp)
    
    def train_step(self) -> Optional[float]:
        """
        Один шаг обучения
        Returns: loss или None если недостаточно опыта
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Выборка из буфера
        experiences, indices, weights = self.memory.sample(self.batch_size)
        
        # Подготовка батча
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(device)
        features = torch.FloatTensor(np.array([e.features for e in experiences])).to(device)
        actions = torch.LongTensor([e.action for e in experiences]).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(device)
        next_features = torch.FloatTensor(np.array([e.next_features for e in experiences])).to(device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(device)
        weights_t = torch.FloatTensor(weights).to(device)
        
        # Текущие Q-значения
        current_q = self.policy_net(states, features).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN: выбираем действия с policy_net, оцениваем с target_net
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states, next_features)
            next_actions = next_q_policy.argmax(1)
            next_q_target = self.target_net(next_states, next_features)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze()
            
            # Целевые Q-значения
            target_q = rewards + self.gamma * next_q * (~dones)
        
        # TD-ошибка для приоритизации
        td_errors = (current_q - target_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        
        # Huber loss с importance sampling weights
        loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = (loss * weights_t).mean()
        
        # Оптимизация
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        self.scheduler.step()
        
        self.steps += 1
        
        # Обновление target network
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, path: str):
        """Сохранение модели"""
        torch.save({
            'model_type': self.model_type,
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'steps': self.steps,
            'training_losses': self.training_losses,
            'episode_rewards': self.episode_rewards,
            'episode_scores': self.episode_scores,
            'max_tiles': self.max_tiles
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Загрузка модели"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            
            # Check for model type compatibility if loading
            if 'model_type' in checkpoint:
                 # Could warn if different, but for now we assume user loads correct one or we just load weights
                 pass

            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.steps = checkpoint['steps']
            self.training_losses = checkpoint.get('training_losses', [])
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.episode_scores = checkpoint.get('episode_scores', [])
            self.max_tiles = checkpoint.get('max_tiles', [])
            print(f"Model loaded from {path}, steps: {self.steps}")
            return True
        return False



if __name__ == "__main__":
    # Тестирование архитектуры
    print(f"Using device: {device}")
    
    # Создаем модель
    model = DuelingDQN()
    print(f"\nModel architecture:")
    print(model)
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Тестовый forward pass
    batch_size = 4
    state = torch.randn(batch_size, 4, 4)
    features = torch.randn(batch_size, 9)
    
    output = model(state, features)
    print(f"\nInput state shape: {state.shape}")
    print(f"Input features shape: {features.shape}")
    print(f"Output Q-values shape: {output.shape}")
    print(f"Sample Q-values: {output[0].detach().numpy()}")

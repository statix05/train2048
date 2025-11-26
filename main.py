#!/usr/bin/env python3
"""
Alpha2048 - AI для игры 2048
============================

Реализация AlphaZero-подобного алгоритма для игры 2048:
- Policy + Value + Planning нейросеть
- MCTS с Chance Nodes для стохастичности
- Curriculum Learning
- Оптимизация под Apple Silicon (MPS)

Режимы запуска:
    python main.py              - Графический интерфейс (GUI)
    python main.py gui          - Графический интерфейс (GUI)
    python main.py demo         - Демонстрация AI
    python main.py train        - Обучение модели
    python main.py train 500    - Обучение на 500 игр
    python main.py play         - Консольная игра
    python main.py play --ai    - Наблюдение за AI
    python main.py info         - Информация о системе
"""

import argparse
import os
import sys
import time


def setup_environment():
    """Настройка окружения"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def show_info():
    """Показать информацию о системе"""
    from alpha2048 import get_device_info, Alpha2048Network
    import torch
    
    print("=" * 60)
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 60)
    
    device_info = get_device_info()
    print(f"\nDevice: {device_info['name']}")
    print(f"Type: {device_info['device']}")
    print(f"Memory: {device_info['memory']}")
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Network info
    net = Alpha2048Network(n_channels=128, n_residual_blocks=6)
    params = sum(p.numel() for p in net.parameters())
    print(f"\nNetwork parameters: {params:,}")
    
    # Check for saved models
    print("\n📁 Saved models:")
    if os.path.exists("models"):
        models = [f for f in os.listdir("models") if f.endswith('.pt')]
        if models:
            for m in sorted(models):
                size = os.path.getsize(f"models/{m}") / 1024 / 1024
                print(f"   {m} ({size:.1f} MB)")
        else:
            print("   (none)")
    else:
        print("   (models directory not found)")


def run_gui():
    """Запуск графического интерфейса"""
    try:
        from gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"❌ Ошибка импорта GUI: {e}")
        print("Установите pygame: pip install pygame")
        sys.exit(1)


def demo():
    """Демонстрация AI"""
    from alpha2048 import demo as alpha_demo
    alpha_demo()


def train(args):
    """Обучение модели"""
    from alpha2048 import Alpha2048Agent, get_device_info
    from trainer import Alpha2048Trainer
    
    device_info = get_device_info()
    
    # Определяем размер сети в зависимости от устройства
    if device_info['device'] == 'cpu':
        # Меньшая сеть для CPU
        n_channels = 64
        n_blocks = 3
        mcts_sims = 20
    else:
        # Полная сеть для GPU/MPS
        n_channels = 128
        n_blocks = 6
        mcts_sims = 50
    
    print("=" * 60)
    print("🧠 Alpha2048 Training")
    print("=" * 60)
    print(f"Device: {device_info['name']}")
    print(f"Network: {n_channels} channels, {n_blocks} blocks")
    print(f"MCTS simulations: {mcts_sims}")
    print(f"Games: {args.games}")
    print("=" * 60 + "\n")
    
    agent = Alpha2048Agent(
        n_channels=n_channels,
        n_residual_blocks=n_blocks,
        mcts_simulations=mcts_sims,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        use_curriculum=not args.no_curriculum
    )
    
    # Загрузка существующей модели
    if args.resume:
        model_path = args.model or "models/alpha2048_best.pt"
        if os.path.exists(model_path):
            agent.load(model_path)
    
    trainer = Alpha2048Trainer(agent)
    trainer.train(
        n_games=args.games,
        games_per_training=args.games_per_train,
        train_steps_per_batch=args.train_steps,
        save_every=max(10, args.games // 10),
        eval_every=max(10, args.games // 20),
        eval_games=5,
        temperature=args.temperature,
        verbose=True
    )
    
    return agent


def play(args):
    """Игра (ручная или с AI)"""
    from game_2048 import Game2048, Direction
    from alpha2048 import Alpha2048Agent, get_device_info
    import random
    
    game = Game2048(mode='infinite')
    
    if args.ai:
        # AI играет
        print("🤖 AI Playing...")
        
        agent = Alpha2048Agent(
            n_channels=64,
            n_residual_blocks=3,
            mcts_simulations=args.mcts
        )
        
        model_path = args.model or "models/alpha2048_best.pt"
        if os.path.exists(model_path):
            agent.load(model_path)
        else:
            print("⚠️  No trained model found, using untrained network")
        
        while not game.is_game_over():
            os.system('clear' if os.name == 'posix' else 'cls')
            print(game)
            
            action, info = agent.select_action(
                game, 
                use_mcts=args.mcts > 0,
                temperature=0.0
            )
            
            directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            print(f"\n🎯 AI move: {directions[action]}")
            print(f"   Value: {info.get('value', 0):.3f}")
            
            game.move(Direction(action))
            time.sleep(args.delay)
        
        print("\n" + "=" * 40)
        print("🏁 GAME OVER")
        print(f"Score: {game.score}")
        print(f"Max tile: {game.max_tile}")
        print(f"Moves: {game.moves}")
        
    else:
        # Ручная игра
        print("🎮 Manual Play")
        print("Controls: W/↑=UP, S/↓=DOWN, A/←=LEFT, D/→=RIGHT, Q=Quit")
        print("In infinite mode: B=Use Bonus (if available), T=Sort Bonus")
        
        key_map = {
            'w': Direction.UP, 'W': Direction.UP,
            's': Direction.DOWN, 'S': Direction.DOWN,
            'a': Direction.LEFT, 'A': Direction.LEFT,
            'd': Direction.RIGHT, 'D': Direction.RIGHT,
        }
        
        while not game.is_game_over():
            os.system('clear' if os.name == 'posix' else 'cls')
            print(game)
            
            if game.bonus_count > 0:
                print(f"\n🎁 Remove bonuses: {game.bonus_count} (Press B)")
            if game.sort_bonuses > 0:
                print(f"⚡ Sort bonuses: {game.sort_bonuses} (Press T)")
            
            try:
                key = input("\nMove (WASD/Q): ").strip()
                
                if key.lower() == 'q':
                    print("Quit")
                    break
                
                if key.lower() == 'b' and game.bonus_count > 0:
                    pos = input("Enter row,col to remove (e.g. 1,2): ").strip()
                    try:
                        row, col = map(int, pos.split(','))
                        if game.use_bonus_remove_tile(row, col):
                            print(f"✅ Removed tile at ({row}, {col})")
                        else:
                            print("❌ Invalid position")
                    except:
                        print("❌ Invalid input")
                    continue
                
                if key.lower() == 't' and game.sort_bonuses > 0:
                    if game.use_sort_bonus():
                        print("⚡ Tiles sorted!")
                    continue
                
                if key in key_map:
                    game.move(key_map[key])
                
            except (EOFError, KeyboardInterrupt):
                break
        
        print("\n" + "=" * 40)
        print("🏁 GAME OVER")
        print(game)


def main():
    parser = argparse.ArgumentParser(
        description="Alpha2048 - AlphaZero for 2048",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # GUI command (default)
    subparsers.add_parser('gui', help='Launch graphical interface')
    
    # Info command
    subparsers.add_parser('info', help='Show system information')
    
    # Demo command
    subparsers.add_parser('demo', help='Run AI demonstration')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('games', type=int, nargs='?', default=100,
                             help='Number of games (default: 100)')
    train_parser.add_argument('--lr', type=float, default=1e-3,
                             help='Learning rate (default: 1e-3)')
    train_parser.add_argument('--batch-size', type=int, default=128,
                             help='Batch size (default: 128)')
    train_parser.add_argument('--games-per-train', type=int, default=5,
                             help='Games between training (default: 5)')
    train_parser.add_argument('--train-steps', type=int, default=50,
                             help='Training steps per batch (default: 50)')
    train_parser.add_argument('--temperature', type=float, default=1.0,
                             help='Exploration temperature (default: 1.0)')
    train_parser.add_argument('--resume', action='store_true',
                             help='Resume from saved model')
    train_parser.add_argument('--model', type=str,
                             help='Path to model file')
    train_parser.add_argument('--no-curriculum', action='store_true',
                             help='Disable curriculum learning')
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Play the game (console)')
    play_parser.add_argument('--ai', action='store_true',
                            help='Watch AI play')
    play_parser.add_argument('--model', type=str,
                            help='Path to model file')
    play_parser.add_argument('--mcts', type=int, default=20,
                            help='MCTS simulations (0 for policy only)')
    play_parser.add_argument('--delay', type=float, default=0.3,
                            help='Delay between moves (seconds)')
    
    args = parser.parse_args()
    
    setup_environment()
    
    if args.command == 'gui':
        run_gui()
    elif args.command == 'info':
        show_info()
    elif args.command == 'demo':
        demo()
    elif args.command == 'train':
        train(args)
    elif args.command == 'play':
        play(args)
    else:
        # Default: launch GUI
        run_gui()


if __name__ == "__main__":
    main()

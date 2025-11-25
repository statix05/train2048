#!/usr/bin/env python3
"""
2048 AI - Главный модуль
========================

Игра 2048 с оригинальным интерфейсом и нейросетью на основе:
- Dueling DQN архитектуры
- Spatial Attention механизма
- Priority Experience Replay
- Креативной системы наград

Режимы запуска:
    python main.py play          - Ручная игра
    python main.py play --ai     - Игра с AI
    python main.py train         - Обучение модели
    python main.py train --quick - Быстрое обучение (демо)
    python main.py demo          - Демонстрация AI без GUI
"""

import argparse
import os
import sys


def setup_environment():
    """Настройка окружения"""
    # Создаем директории
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def train_model(args):
    """Обучение модели"""
    from neural_network import DQNAgent, device
    from trainer import Trainer
    
    print("=" * 60)
    print("🧠 2048 Neural Network Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Model Type: {args.model_type}")
    print("=" * 60)
    
    agent = DQNAgent(
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
        epsilon_decay=args.episodes * 5,
        model_type=args.model_type
    )
    
    # Загрузка существующей модели если есть
    if args.resume and os.path.exists("models/model_best.pt"):
        agent.load("models/model_best.pt")
        print("Resumed from existing model")
    
    trainer = Trainer(agent)
    trainer.train(
        n_episodes=args.episodes,
        save_every=max(1, args.episodes // 10),
        eval_every=max(1, args.episodes // 20),
        eval_episodes=5
    )
    
    return agent


def play_game(args):
    """Запуск игры"""
    # Пробуем GUI (Tkinter)
    try:
        print("Trying to load GUI (Tkinter)...")
        from gui import play_with_ai, play_manual
        
        print("✓ GUI loaded successfully!")
        
        if args.ai:
            model_path = getattr(args, 'model', None) or "models/model_best.pt"
            play_with_ai(model_path)
        else:
            play_manual()
        return
    
    except ImportError as e:
        if "_tkinter" in str(e):
            print("✗ Tkinter not available (GUI requires Tkinter)")
            print("  Install Python with Tkinter support or use terminal mode")
        else:
            print(f"✗ GUI failed: {e}")
    
    # Запасной вариант: терминальный интерфейс
    try:
        print("\n→ Using Terminal Interface instead...")
        from gui_terminal import play_terminal, play_terminal_with_ai
        
        if args.ai:
            model_path = getattr(args, 'model', None) or "models/model_best.pt"
            play_terminal_with_ai(model_path)
        else:
            play_terminal()
        return
    
    except Exception as e:
        print(f"✗ Terminal interface failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Последний вариант: консольная демонстрация
    print("\n→ Running console demo...")
    demo_console(args)


def demo_console(args):
    """Демонстрация в консоли"""
    from game_2048 import Game2048, Direction
    from neural_network import DQNAgent, device
    import time
    
    print("=" * 60)
    print("🎮 2048 AI Console Demo")
    print("=" * 60)
    
    # Загружаем модель
    agent = DQNAgent()
    model_path = getattr(args, 'model', None) or "models/model_best.pt"
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("No trained model found, using random agent")
    
    agent.policy_net.eval()
    
    # Играем несколько игр
    n_games = getattr(args, 'games', 5)
    verbose = getattr(args, 'verbose', False)
    results = []
    
    for game_num in range(1, n_games + 1):
        game = Game2048()
        print(f"\n{'='*40}")
        print(f"Game {game_num}/{n_games}")
        print('='*40)
        
        while not game.is_game_over():
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            state = game.get_state()
            features = game.get_features()
            valid_moves_int = [int(m) for m in valid_moves]
            
            action = agent.select_action(state, features, valid_moves_int, epsilon=0.0)
            game.move(Direction(action))
            
            # Отображение
            if verbose:
                os.system('clear' if os.name == 'posix' else 'cls')
                print(f"Game {game_num}/{n_games}")
                print(game)
                time.sleep(0.05)
        
        results.append({
            'score': game.score,
            'max_tile': game.max_tile,
            'moves': game.moves
        })
        
        print(f"\nGame {game_num} finished!")
        print(f"Score: {game.score}")
        print(f"Max tile: {game.max_tile}")
        print(f"Moves: {game.moves}")
    
    # Итоговая статистика
    print("\n" + "=" * 60)
    print("📊 Final Statistics")
    print("=" * 60)
    
    avg_score = sum(r['score'] for r in results) / len(results)
    avg_tile = sum(r['max_tile'] for r in results) / len(results)
    best_score = max(r['score'] for r in results)
    best_tile = max(r['max_tile'] for r in results)
    
    print(f"Average Score: {avg_score:.0f}")
    print(f"Average Max Tile: {avg_tile:.0f}")
    print(f"Best Score: {best_score}")
    print(f"Best Max Tile: {best_tile}")
    
    # Распределение плиток
    tile_dist = {}
    for r in results:
        tile = r['max_tile']
        tile_dist[tile] = tile_dist.get(tile, 0) + 1
    
    print("\nMax Tile Distribution:")
    for tile in sorted(tile_dist.keys(), reverse=True):
        count = tile_dist[tile]
        pct = count / len(results) * 100
        print(f"  {tile}: {count} ({pct:.1f}%)")


def quick_demo():
    """Быстрая демонстрация без обучения"""
    from game_2048 import Game2048, Direction
    import random
    
    print("=" * 60)
    print("🎮 2048 Quick Demo (Random Agent)")
    print("=" * 60)
    
    game = Game2048()
    print("\nInitial state:")
    print(game)
    
    move_count = 0
    while not game.is_game_over() and move_count < 100:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            break
        
        move = random.choice(valid_moves)
        reward, done, info = game.move(move)
        move_count += 1
        
        if move_count % 20 == 0:
            print(f"\nAfter {move_count} moves:")
            print(game)
    
    print(f"\n{'='*40}")
    print("Final state:")
    print(game)
    print(f"\nGame Over! Score: {game.score}, Max Tile: {game.max_tile}")


def main():
    parser = argparse.ArgumentParser(
        description="2048 AI Game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Play the game')
    play_parser.add_argument('--ai', action='store_true', help='Let AI play')
    play_parser.add_argument('--model', type=str, help='Path to model file')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the neural network')
    train_parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    train_parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')
    train_parser.add_argument('--target-update', type=int, default=1000, help='Target network update frequency')
    train_parser.add_argument('--quick', action='store_true', help='Quick training (500 episodes)')
    train_parser.add_argument('--resume', action='store_true', help='Resume from existing model')
    train_parser.add_argument('--gui', action='store_true', help='Use GUI for training visualization')
    train_parser.add_argument('--model-type', type=str, default='dueling', choices=['simple', 'conv', 'dueling', 'hybrid'], help='Network architecture')
    train_parser.add_argument('--mode', type=str, default='classic', choices=['classic', 'dynamic'], help='Game mode: classic (90%% 2 / 10%% 4) or dynamic (scaling values)')
    
    # Train GUI command
    subparsers.add_parser('train-gui', help='Train with GUI visualization')
    
    # Train terminal command
    train_term_parser = subparsers.add_parser('train-terminal', help='Train with terminal visualization')
    train_term_parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    train_term_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_term_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    train_term_parser.add_argument('--buffer-size', type=int, default=50000, help='Buffer size')
    train_term_parser.add_argument('--model-type', type=str, default='dueling', choices=['simple', 'conv', 'dueling', 'hybrid'], help='Network architecture')
    train_term_parser.add_argument('--mode', type=str, default='classic', choices=['classic', 'dynamic'], help='Game mode')
    
    # Replay command
    replay_parser = subparsers.add_parser('replay', help='Watch best game replay')
    replay_parser.add_argument('--file', default='logs/best_replay.json', help='Replay file path')
    replay_parser.add_argument('--speed', type=float, default=0.2, help='Playback speed')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run AI demo in console')
    demo_parser.add_argument('--games', type=int, default=5, help='Number of games')
    demo_parser.add_argument('--model', type=str, help='Path to model file')
    demo_parser.add_argument('--verbose', action='store_true', help='Show game progress')
    
    # Quick demo command
    subparsers.add_parser('quick', help='Quick demo without training')
    
    args = parser.parse_args()
    
    setup_environment()
    
    if args.command == 'play':
        play_game(args)
    elif args.command == 'train':
        if args.quick:
            args.episodes = 500
        
        # Проверяем флаг GUI
        if hasattr(args, 'gui') and args.gui:
            try:
                from training_gui import TrainingGUI
                print("Starting training with GUI...")
                gui = TrainingGUI()
                gui.run()
            except ImportError as e:
                print(f"Training GUI not available: {e}")
                print("Falling back to console training...")
                train_model(args)
        else:
            train_model(args)
    elif args.command == 'train-gui':
        # Прямой запуск GUI обучения
        try:
            from training_gui import TrainingGUI
            print("Starting training GUI...")
            gui = TrainingGUI()
            gui.run()
        except Exception as e:
            print(f"Training GUI not available: {e}")
            print("Falling back to terminal training...")
            from training_terminal import train_terminal
            train_terminal()
    elif args.command == 'train-terminal':
        # Терминальное обучение с визуализацией
        from training_terminal import train_terminal
        train_terminal(
            n_episodes=args.episodes,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            model_type=args.model_type,
            game_mode=args.mode
        )
    elif args.command == 'replay':
        from replay_viewer import play_replay
        play_replay(args.file, args.speed)
    elif args.command == 'demo':
        demo_console(args)
    elif args.command == 'quick':
        quick_demo()
    else:
        # По умолчанию - быстрая демонстрация
        parser.print_help()
        print("\n" + "=" * 60)
        print("Running quick demo...")
        print("=" * 60)
        quick_demo()


if __name__ == "__main__":
    main()

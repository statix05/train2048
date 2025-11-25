#!/usr/bin/env python3
import json
import os
import time
import sys
import numpy as np

COLORS = {
    0: '\033[90m',      # Dark Gray
    2: '\033[97m',      # White
    4: '\033[93m',      # Yellow
    8: '\033[33m',      # Orange (Yellow-ish)
    16: '\033[31m',     # Red
    32: '\033[91m',     # Light Red
    64: '\033[35m',     # Magenta
    128: '\033[95m',    # Light Magenta
    256: '\033[34m',    # Blue
    512: '\033[94m',    # Light Blue
    1024: '\033[36m',   # Cyan
    2048: '\033[96m',   # Light Cyan
    4096: '\033[32m',   # Green
    8192: '\033[92m',   # Light Green
}
RESET = '\033[0m'
BOLD = '\033[1m'

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def get_color(val):
    return COLORS.get(val, '\033[97m')

def draw_board(board):
    lines = []
    lines.append("┌────┬────┬────┬────┐")
    for i in range(4):
        row = "│"
        for j in range(4):
            val = int(board[i][j])
            color = get_color(val)
            if val == 0:
                row += f" {color} ·  {RESET}│"
            elif val < 100:
                row += f"{color}{val:^4}{RESET}│"
            elif val < 1000:
                row += f"{color}{val:^4}{RESET}│"
            else:
                row += f"{color}{val:4}{RESET}│"
        lines.append(row)
        if i < 3:
            lines.append("├────┼────┼────┼────┤")
    lines.append("└────┴────┴────┴────┘")
    return "\n".join(lines)

def play_replay(path="logs/best_replay.json", delay=0.3):
    if not os.path.exists(path):
        print(f"No replay file found at {path}")
        return

    with open(path, 'r') as f:
        data = json.load(f)

    history = data['history']
    score = data['score']
    max_tile = data['max_tile']

    print(f"Loaded Replay: Score {score} | Max Tile {max_tile} | Moves {len(history)}")
    input("Press Enter to start...")

    for i, board in enumerate(history):
        clear_screen()
        print(f"Move: {i}/{len(history)}")
        print(draw_board(board))
        print(f"\nStats: Score {score} (Final)")
        time.sleep(delay)

    print("\nReplay Finished!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='logs/best_replay.json')
    parser.add_argument('--speed', type=float, default=0.2)
    args = parser.parse_args()
    play_replay(args.file, args.speed)

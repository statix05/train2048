#!/usr/bin/env python3
"""
Упрощённая версия GUI для отладки
"""
import tkinter as tk
from tkinter import messagebox
from game_2048 import Game2048, Direction

# Простые цвета
COLORS = {
    0: "#cdc1b4",
    2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563",
    32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61",
    512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
}

class SimpleGUI:
    def __init__(self):
        print("Creating window...")
        self.root = tk.Tk()
        self.root.title("2048")
        self.root.configure(bg='#faf8ef')
        
        # Game
        self.game = Game2048()
        
        # Score
        self.score_label = tk.Label(self.root, text=f"Score: {self.game.score}", 
                                     font=("Arial", 20), bg='#faf8ef')
        self.score_label.pack(pady=10)
        
        # Board frame
        self.board_frame = tk.Frame(self.root, bg='#bbada0', padx=5, pady=5)
        self.board_frame.pack(padx=10, pady=10)
        
        # Create cells
        self.cells = []
        for i in range(4):
            row = []
            for j in range(4):
                cell = tk.Label(
                    self.board_frame,
                    text="",
                    font=("Arial", 32, "bold"),
                    width=4,
                    height=2,
                    bg="#cdc1b4"
                )
                cell.grid(row=i, column=j, padx=5, pady=5)
                row.append(cell)
            self.cells.append(row)
        
        # Instructions
        tk.Label(self.root, text="Use arrow keys | R: restart | Q: quit",
                font=("Arial", 12), bg='#faf8ef').pack(pady=10)
        
        # Key bindings
        self.root.bind("<Up>", lambda e: self.move(Direction.UP))
        self.root.bind("<Down>", lambda e: self.move(Direction.DOWN))
        self.root.bind("<Left>", lambda e: self.move(Direction.LEFT))
        self.root.bind("<Right>", lambda e: self.move(Direction.RIGHT))
        self.root.bind("<r>", lambda e: self.restart())
        self.root.bind("<R>", lambda e: self.restart())
        self.root.bind("<q>", lambda e: self.root.quit())
        self.root.bind("<Q>", lambda e: self.root.quit())
        
        print("Widgets created, updating display...")
        self.update_display()
        print("Ready! Window should be visible.")
    
    def move(self, direction):
        reward, done, info = self.game.move(direction)
        self.update_display()
        if done:
            messagebox.showinfo("Game Over", 
                              f"Score: {self.game.score}\nBest: {self.game.max_tile}")
    
    def restart(self):
        self.game.reset()
        self.update_display()
    
    def update_display(self):
        self.score_label.config(text=f"Score: {self.game.score} | Best: {self.game.max_tile}")
        
        for i in range(4):
            for j in range(4):
                value = self.game.board[i, j]
                cell = self.cells[i][j]
                
                if value == 0:
                    cell.config(text="", bg="#cdc1b4", fg="#776e65")
                else:
                    color = COLORS.get(value, "#3c3a32")
                    text_color = "#776e65" if value <= 4 else "#f9f6f2"
                    cell.config(text=str(value), bg=color, fg=text_color)
    
    def run(self):
        print("Starting mainloop...")
        self.root.mainloop()

if __name__ == "__main__":
    gui = SimpleGUI()
    gui.run()

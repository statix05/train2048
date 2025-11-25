#!/usr/bin/env python3
"""
Тест GUI для диагностики проблемы
"""
import sys

print("Testing GUI components...")
print(f"Python version: {sys.version}")

# Test 1: Tkinter import
try:
    import tkinter as tk
    print("✓ Tkinter imported successfully")
except Exception as e:
    print(f"✗ Tkinter import failed: {e}")
    sys.exit(1)

# Test 2: Create window
try:
    root = tk.Tk()
    root.title("Test Window")
    print("✓ Window created")
    
    label = tk.Label(root, text="If you see this window, GUI works!", font=("Helvetica", 20))
    label.pack(padx=50, pady=50)
    
    button = tk.Button(root, text="Close", command=root.quit)
    button.pack(pady=20)
    
    print("✓ Widgets created")
    print("\nWindow should appear now...")
    print("Close it to continue")
    
    root.mainloop()
    print("✓ GUI test passed!")
    
except Exception as e:
    print(f"✗ GUI test failed: {e}")
    import traceback
    traceback.print_exc()

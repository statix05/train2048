#!/bin/bash
# Проверка системы для 2048 AI

echo "🔍 Проверка системы для 2048 AI"
echo "=================================="
echo ""

# Python версия
echo "📌 Python:"
python3 --version 2>/dev/null || echo "  ✗ python3 не найден"
which python3

echo ""
echo "📌 Активный Python в venv:"
if [ -d "venv" ]; then
    source venv/bin/activate 2>/dev/null
    which python
    python --version
else
    echo "  ⚠ venv не найден"
fi

echo ""
echo "📌 Проверка зависимостей:"

# PyTorch
python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" 2>/dev/null || echo "  ✗ PyTorch не установлен"

# NumPy
python -c "import numpy; print(f'  ✓ NumPy {numpy.__version__}')" 2>/dev/null || echo "  ✗ NumPy не установлен"

# MPS
python -c "import torch; print(f'  ✓ MPS доступен: {torch.backends.mps.is_available()}')" 2>/dev/null

# Tkinter (опционально)
python -c "import tkinter; print('  ✓ Tkinter доступен (GUI будет работать)')" 2>/dev/null || echo "  ⚠ Tkinter недоступен (используйте терминальный интерфейс)"

echo ""
echo "📌 Готово к запуску:"
echo ""
echo "  Терминальный интерфейс (работает всегда):"
echo "    python gui_terminal.py"
echo ""
echo "  GUI (если Tkinter доступен):"
echo "    python main.py play"
echo ""
echo "  Обучение:"
echo "    python main.py train --quick"
echo ""

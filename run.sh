#!/bin/bash
# Универсальный launcher для 2048 AI
# Автоматически использует правильный Python

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🎮 2048 AI Launcher"
echo "==================="
echo ""

# Проверяем существует ли venv
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  venv не найден${NC}"
    echo "Создайте venv с правильным Python:"
    echo "  ./activate_correct_python.sh"
    echo ""
    exit 1
fi

# Активируем venv
source venv/bin/activate

# Проверяем Python и Tkinter
PYTHON_PATH=$(which python)
echo "📍 Используется Python: $PYTHON_PATH"
python --version

echo ""
echo "🔍 Проверка Tkinter..."
if python -c "import tkinter" 2>/dev/null; then
    echo -e "${GREEN}✅ Tkinter доступен${NC}"
    TKINTER_OK=true
else
    echo -e "${RED}❌ Tkinter недоступен${NC}"
    echo "Используйте терминальный интерфейс или пересоздайте venv:"
    echo "  ./activate_correct_python.sh"
    TKINTER_OK=false
fi

echo ""
echo "🔍 Проверка PyTorch..."
if python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    python -c "import torch; print(f'MPS доступен: {torch.backends.mps.is_available()}')"
else
    echo -e "${RED}❌ PyTorch не установлен${NC}"
    echo "Установите: pip install torch numpy"
    exit 1
fi

echo ""
echo "=================="
echo ""

# Запускаем команду
if [ $# -eq 0 ]; then
    # Без аргументов - показываем меню
    echo "Выберите режим:"
    echo ""
    echo "1) Играть (GUI)"
    echo "2) Играть (Терминал)"
    echo "3) AI играет (GUI)"
    echo "4) AI играет (Терминал)"
    echo "5) Обучение с визуализацией (Training GUI)"
    echo "6) Быстрое обучение (консоль)"
    echo "7) Выход"
    echo ""
    read -p "Выбор (1-7): " choice
    
    case $choice in
        1)
            if [ "$TKINTER_OK" = true ]; then
                python main.py play
            else
                echo "Tkinter недоступен. Используйте терминальный режим (опция 2)"
            fi
            ;;
        2)
            python gui_terminal.py
            ;;
        3)
            if [ "$TKINTER_OK" = true ]; then
                python main.py play --ai
            else
                echo "Tkinter недоступен. Используйте терминальный режим (опция 4)"
            fi
            ;;
        4)
            python gui_terminal.py --ai
            ;;
        5)
            if [ "$TKINTER_OK" = true ]; then
                python main.py train-gui
            else
                echo "Training GUI требует Tkinter"
                echo "Используйте консольное обучение: python main.py train --quick"
            fi
            ;;
        6)
            python main.py train --quick
            ;;
        7)
            echo "Выход..."
            exit 0
            ;;
        *)
            echo "Неверный выбор"
            exit 1
            ;;
    esac
else
    # С аргументами - передаём в main.py
    python main.py "$@"
fi

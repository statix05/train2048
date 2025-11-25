#!/bin/bash
# Скрипт для создания venv с правильным Python (не Homebrew)

echo "🔍 Поиск правильного Python с Tkinter..."
echo ""

# Массив возможных путей к Python
PYTHON_PATHS=(
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
    "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
    "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
    "/usr/bin/python3"
    "/usr/local/bin/python3"
)

FOUND_PYTHON=""

# Поиск Python с Tkinter
for py_path in "${PYTHON_PATHS[@]}"; do
    if [ -f "$py_path" ]; then
        echo "Проверяю: $py_path"
        if $py_path -c "import tkinter" 2>/dev/null; then
            echo "  ✅ Tkinter работает!"
            FOUND_PYTHON=$py_path
            break
        else
            echo "  ❌ Tkinter не найден"
        fi
    fi
done

if [ -z "$FOUND_PYTHON" ]; then
    echo ""
    echo "❌ Python с Tkinter не найден!"
    echo ""
    echo "Установите Python с https://www.python.org/downloads/macos/"
    echo "Или используйте терминальный интерфейс:"
    echo "  python gui_terminal.py"
    exit 1
fi

echo ""
echo "✅ Найден Python с Tkinter: $FOUND_PYTHON"
echo ""

# Удалить старый venv если существует
if [ -d "venv" ]; then
    echo "⚠️  Обнаружен старый venv, переименовываю в venv_old..."
    mv venv venv_old
fi

# Создать новый venv
echo "📦 Создаю новый venv с правильным Python..."
$FOUND_PYTHON -m venv venv

echo ""
echo "📥 Устанавливаю зависимости..."
source venv/bin/activate
pip install --upgrade pip -q
pip install torch numpy -q

echo ""
echo "✅ Готово!"
echo ""
echo "Теперь используйте:"
echo "  source venv/bin/activate"
echo "  python main.py play"
echo "  python main.py train-gui"
echo ""

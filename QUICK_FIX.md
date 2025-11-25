# 🚀 Быстрое решение: Использовать правильный Python

## ⚡ АВТОМАТИЧЕСКОЕ РЕШЕНИЕ (рекомендуется)

```bash
cd /Users/statix/2048_ai

# Этот скрипт найдёт правильный Python и создаст venv
./activate_correct_python.sh

# Затем используйте универсальный launcher
./run.sh
```

**Скрипт `activate_correct_python.sh` автоматически:**
1. Найдёт Python с Tkinter (не Homebrew)
2. Создаст новый venv с правильным Python
3. Установит все зависимости

---

## ✅ Терминальный интерфейс (работает всегда)

Если не хотите возиться с Python, используйте терминальный интерфейс:

```bash
cd /Users/statix/2048_ai
source venv/bin/activate  # Любой Python подойдёт

# ИГРАТЬ в терминале (работает с любым Python!)
python gui_terminal.py
```

**Это не требует Tkinter, GUI или переустановки Python!**

---

## 🎮 Запуск без venv (если что-то не так)

```bash
cd /Users/statix/2048_ai

# Прямо с системным Python
python3 gui_terminal.py

# Или через Homebrew Python
/opt/homebrew/bin/python3 gui_terminal.py
```

---

## 🔧 Если хотите GUI: Установить Python.org версию

### Способ 1: Скачать официальный установщик

1. Откройте: https://www.python.org/downloads/macos/
2. Скачайте **macOS 64-bit universal2 installer**
3. Установите (двойной клик на .pkg)
4. Затем:

```bash
cd /Users/statix/2048_ai

# Создать НОВЫЙ venv с Python.org версией
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv venv_gui

# Активировать
source venv_gui/bin/activate

# Проверить что это правильный Python
which python
# Должно быть: /Users/statix/2048_ai/venv_gui/bin/python

# Проверить Tkinter
python -c "import tkinter; print('✅ Tkinter работает!')"

# Установить зависимости
pip install torch numpy

# Запустить GUI
python main.py play
```

---

## 🍎 Способ 2: Использовать встроенный macOS Python

macOS обычно идёт с Python + Tkinter:

```bash
cd /Users/statix/2048_ai

# Проверить встроенный Python
/usr/bin/python3 --version
/usr/bin/python3 -c "import tkinter; print('OK')"

# Если работает, создать venv:
/usr/bin/python3 -m venv venv_system
source venv_system/bin/activate
pip install torch numpy
python main.py play
```

---

## 💡 Рекомендация

**Используйте терминальный интерфейс** — он работает отлично:

```bash
cd /Users/statix/2048_ai
source venv/bin/activate
python gui_terminal.py
```

Терминальная версия имеет:
- ✅ Цветные плитки
- ✅ Интерактивное управление стрелками
- ✅ Режим AI
- ✅ Красивый интерфейс
- ✅ Все функции как в GUI

**Никакой разницы в функциональности!**

---

## 🧠 Обучение работает с любым Python

```bash
cd /Users/statix/2048_ai
source venv/bin/activate  # Ваш существующий venv

# Быстрое обучение (2 мин)
python main.py train --quick

# Смотреть результаты
python gui_terminal.py --ai
```

---

## 🎯 Итоговая команда (работает сейчас)

```bash
cd /Users/statix/2048_ai
source venv/bin/activate
python gui_terminal.py
```

**Управление:** `↑↓←→` двигать | `r` рестарт | `a` AI режим | `q` выход

---

## ❓ Если что-то не работает

Напишите вывод этих команд:

```bash
cd /Users/statix/2048_ai
source venv/bin/activate
which python
python --version
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import numpy; print('NumPy OK')"
```

Но **gui_terminal.py должен работать с любым Python!**

---

## 🎯 Итоговая инструкция

### Вариант 1: Автоматическое исправление (для GUI)

```bash
cd /Users/statix/2048_ai
./activate_correct_python.sh  # Создаст правильный venv
./run.sh                        # Запустит с меню
```

### Вариант 2: Терминальный интерфейс (работает сейчас)

```bash
cd /Users/statix/2048_ai
source venv/bin/activate
python gui_terminal.py
```

### Вариант 3: Универсальный launcher

```bash
cd /Users/statix/2048_ai
./run.sh                # Интерактивное меню
./run.sh play          # Напрямую запустить игру
./run.sh train-gui     # Training GUI
```

**Launcher `run.sh` автоматически проверяет Python и Tkinter!**

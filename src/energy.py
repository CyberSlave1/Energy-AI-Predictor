import os
import platform
import subprocess
from pyfiglet import Figlet
from colorama import init, Fore, Style
from tabulate import tabulate
from features import get_feature_list


init(autoreset=True)

ARGS = [
    ["--mode",               "Режим роботи: train_auto, train_manual, predict, test, auto"],
    ["--train_path",         "Шлях до CSV з навчальними даними"],
    ["--predict_path",       "Шлях до CSV/XLSX з новими даними"],
    ["--model_path",         "Шлях до збереженої або нової моделі (.pkl)"],
    ["--params_path",        "JSON-файл з параметрами моделі"],
    ["--output_path",        "Шлях до Excel-файлу з прогнозом"],
    ["--n_trials",           "К-сть ітерацій для Optuna (default: 100)"],
    ["--lower_percentile",   "Нижній перцентиль для видалення викидів (default: 0.001)"],
    ["--upper_percentile",   "Верхній перцентиль для видалення викидів (default: 0.999)"],
    ["--plot_save",          "Зберегти графіки оцінки моделі"],
    ["--log_config",         "Логувати конфігурацію запуску"],
    ["--compare_models",     "Зберігати модель лише якщо вона краща за попередню"],
    ["--exclude_features",   "Список ознак для виключення з моделі (через пробіл)"],
    ["--normalize_features", "Нормалізувати числові ознаки"],
    ["--show_features",      "Вивести всі доступні ознаки без запуску моделі"]
]

MODES = [
    ["train_auto",   "Автоматичне тренування з Optuna"],
    ["train_manual", "Тренування з параметрами з JSON-файлу"],
    ["predict",      "Прогнозування для нових даних"],
    ["test",         "Оцінка моделі на класичних або останніх зразках"],
    ["auto",         "Повний цикл: тренування → прогноз → тест"]
]

def print_banner():
    os.system('cls' if os.name == 'nt' else 'clear')
    f = Figlet(font='slant')
    print(Fore.CYAN + f.renderText('ENERGY AI'))
    print(Fore.GREEN + Style.BRIGHT + "Інтелектуальна система прогнозування споживання електроенергії ⚡\n")

def print_help():
    print(Fore.YELLOW + Style.BRIGHT + "[ Доступні режими запуску ]")
    print(Fore.WHITE + tabulate(MODES, headers=["--mode", "Опис режиму"], tablefmt="fancy_grid"))
    print(Fore.YELLOW + Style.BRIGHT + "\n[ Аргументи командного рядка ]")
    print(Fore.WHITE + tabulate(ARGS, headers=["Аргумент", "Опис"], tablefmt="fancy_grid"))

def print_available_features():
    print(Fore.YELLOW + Style.BRIGHT + "\n[ Доступні ознаки для навчання ]")
    features = get_feature_list()
    for i, feat in enumerate(features):
        print(f" {i+1:2}. {feat}")

def launch():
    print_banner()
    print_help()

    while True:
        user_input = input(Fore.CYAN + Style.BRIGHT + "\nВведіть режим запуску з аргументами (або 'exit' / '--show_features'):\n> ").strip()
        if user_input.lower() == 'exit':
            print(Fore.MAGENTA + "Завершення програми...")
            break

        if user_input == '--show_features':
            print_available_features()
            continue

        if not user_input:
            continue

        try:
            python_cmd = "py" if platform.system() == "Windows" else "python"
            print(Fore.BLUE + f"\n>>> Виконання: {python_cmd} main.py {user_input}\n")
            subprocess.run(f"{python_cmd} main.py {user_input}", shell=True)
        except Exception as e:
            print(Fore.RED + f"[Помилка]: {e}")

if __name__ == '__main__':
    launch()

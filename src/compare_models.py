#!/usr/bin/env python3
"""
Сравнение двух JSON-файлов с числовыми метриками.
Вывод: транспонированная таблица (метрики — столбцы, файлы — строки)
с цветовым выделением (зелёный — больше, красный — меньше),
числа с 5 знаками после запятой.
"""

import sys
import json
from colorama import init, Fore, Style

# from colorama import init

# init(strip=False)

init(autoreset=True)


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    file1 = "metrics/metrics_of_trained.json"
    file2 = "metrics/metrics_of_post_trained.json"

    try:
        data1 = load_json(file1)
        data2 = load_json(file2)
    except Exception as e:
        print(f"Ошибка загрузки JSON: {e}")
        sys.exit(1)

    # Все метрики в алфавитном порядке
    metrics = sorted(set(data1.keys()) | set(data2.keys()))

    # Ширина первой колонки (имя файла)
    first_col_width = max(len(file1), len(file2)) + 2

    # Ширина колонок для метрик
    col_widths = [max(len(m), 12) + 2 for m in metrics]

    # Заголовок таблицы
    print(f"\n{'':<{first_col_width}}", end="")
    for m, w in zip(metrics, col_widths):
        print(f"{m:>{w}}", end="")
    print()
    print("-" * (first_col_width + sum(col_widths)))

    # Данные двух файлов
    for filename, file_data in [(file1, data1), (file2, data2)]:
        print(f"{filename:<{first_col_width}}", end="")
        for m, w in zip(metrics, col_widths):
            val = file_data.get(m)
            # Если значение — число, сравниваем с другим файлом
            if isinstance(val, (int, float)):
                val_str = f"{val:.5f}"
                other_data = data2 if file_data is data1 else data1
                other_val = other_data.get(m)
                if isinstance(other_val, (int, float)):
                    if val > other_val:
                        color = Fore.GREEN
                    elif val < other_val:
                        color = Fore.RED
                    else:
                        color = ""
                else:
                    color = ""
                # Форматируем с выравниванием по правому краю, затем раскрашиваем
                padded = f"{val_str:>{w}}"
                if color:
                    print(f"{color}{padded}{Style.RESET_ALL}", end="")
                else:
                    print(padded, end="")
            else:
                print(f"{'N/A':>{w}}", end="")
        print()


if __name__ == "__main__":
    main()

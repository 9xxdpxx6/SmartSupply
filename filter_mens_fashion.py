"""
Скрипт для фильтрации записей категории "Men's Fashion" из CSV файла
"""
import pandas as pd
import sys

def filter_mens_fashion(input_file, output_file=None):
    """
    Фильтрует записи категории "Men's Fashion" из CSV файла
    
    Args:
        input_file: Путь к входному CSV файлу
        output_file: Путь к выходному CSV файлу (по умолчанию: input_file_mens_fashion.csv)
    """
    print(f"Чтение файла: {input_file}")
    
    # Читаем CSV файл
    try:
        df = pd.read_csv(input_file)
        print(f"Всего записей в файле: {len(df)}")
    except FileNotFoundError:
        print(f"Ошибка: Файл {input_file} не найден!")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        sys.exit(1)
    
    # Фильтруем по категории "Men's Fashion"
    print(f"Фильтрация записей категории 'Men's Fashion'...")
    filtered_df = df[df['category'] == "Men's Fashion"]
    
    print(f"Найдено записей категории 'Men's Fashion': {len(filtered_df)}")
    
    # Определяем имя выходного файла
    if output_file is None:
        if input_file.endswith('.csv'):
            output_file = input_file.replace('.csv', '_mens_fashion.csv')
        else:
            output_file = input_file + '_mens_fashion.csv'
    
    # Сохраняем отфильтрованные данные
    try:
        filtered_df.to_csv(output_file, index=False)
        print(f"Результат сохранен в файл: {output_file}")
        print(f"Процент отфильтрованных записей: {len(filtered_df)/len(df)*100:.2f}%")
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Путь к входному файлу
    input_file = "sales_06_FY2020-21.csv"
    
    # Можно указать выходной файл как аргумент командной строки
    output_file = None
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    
    filter_mens_fashion(input_file, output_file)


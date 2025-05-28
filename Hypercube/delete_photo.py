import os
def delete_tif_files(folder_path):
    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не существует.")
        return
    # Рекурсивно проходимся по всем папкам и файлам
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # Проверяем, что файл имеет расширение .tif
            if filename.endswith(".tif"):
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    print(f"Файл {file_path} удален.")
                except Exception as e:
                    print(f"Ошибка при удалении файла {file_path}: {e}")
if __name__ == "__main__":
    folder_path = r'C:\NE - пигментный невус'
    delete_tif_files(folder_path)
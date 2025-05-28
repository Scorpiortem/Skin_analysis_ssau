import os
import numpy as np
from PIL import Image
from tkinter import Tk, filedialog

def select_input_folder():
    root = Tk()
    root.withdraw()
    return filedialog.askdirectory(title="Выберите папку с изображениями .tif")

def select_output_file():
    root = Tk()
    root.withdraw()
    return filedialog.asksaveasfilename(
        title="Сохранить гиперкуб как...",
        defaultextension=".npy",
        filetypes=[("NumPy files", "*.npy")]
    )

def load_images(folder_path):
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.tif')])
    if not image_files:
        raise ValueError("В выбранной папке нет .tif изображений")
    return image_files

def create_hypercube(folder_path, image_files):
    # Определение размеров и типа данных по первому изображению
    with Image.open(os.path.join(folder_path, image_files[0])) as img:
        width, height = img.size
        dtype = np.uint16 if img.mode == 'I;16' else np.uint8

    # Создание гиперкуба с правильным типом данных
    hypercube = np.zeros((height, width, len(image_files)), dtype=dtype)
    
    for i, filename in enumerate(image_files):
        with Image.open(os.path.join(folder_path, filename)) as img:
            if img.size != (width, height):
                raise ValueError(f"Изображение {filename} имеет разные размеры")
            hypercube[:, :, i] = np.array(img)
    
    return hypercube

def main():
    input_folder = select_input_folder()
    if not input_folder:
        return

    output_path = select_output_file()
    if not output_path:
        return

    try:
        image_files = load_images(input_folder)
        hypercube = create_hypercube(input_folder, image_files)
        np.save(output_path, hypercube)
        print(f"Гиперкуб успешно сохранен в {output_path}")
    except Exception as e:
        print(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main()
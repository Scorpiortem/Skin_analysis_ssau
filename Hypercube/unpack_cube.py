import os
import numpy as np
from PIL import Image
from tkinter import Tk, filedialog

def select_hypercube():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Выберите файл гиперкуба",
        filetypes=[("NumPy files", "*.npy")]
    )

def select_output_folder():
    root = Tk()
    root.withdraw()
    return filedialog.askdirectory(title="Выберите папку для сохранения изображений")

def decompose_hypercube(hypercube, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    num_channels = hypercube.shape[2]
    
    # Определение типа данных для сохранения
    dtype = hypercube.dtype
    mode = 'I;16' if dtype == np.uint16 else 'L'

    for i in range(num_channels):
        channel = hypercube[:, :, i]
        img = Image.fromarray(channel, mode=mode)
        img.save(os.path.join(output_folder, f"channel_{i:03d}.tif"), compression='tiff_lzw')

def main():
    hypercube_path = select_hypercube()
    if not hypercube_path:
        return

    output_folder = select_output_folder()
    if not output_folder:
        return

    try:
        hypercube = np.load(hypercube_path)
        decompose_hypercube(hypercube, output_folder)
        print(f"Изображения успешно сохранены в {output_folder}")
    except Exception as e:
        print(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main()
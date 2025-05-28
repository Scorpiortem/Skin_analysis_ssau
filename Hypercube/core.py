import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog

def load_hypercube(path):
    return np.load(path)

def find_optimal_channel(hypercube, mask):
    h, w, _ = hypercube.shape
    mask_resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    channel_means = np.mean(hypercube * mask_resized[..., np.newaxis], axis=(0, 1))
    return np.argmax(channel_means)

def select_file(title, filetypes):
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path

# Выбор модели и гиперкуба
model_path = select_file("Выберите модель", [("HDF5 files", "*.h5")])
hypercube_path = select_file("Выберите гиперкуб", [("Numpy files", "*.npy")])

# Загрузка данных
model = load_model(model_path)
hypercube = load_hypercube(hypercube_path)

# Сегментация
input_image = hypercube[..., [25, 15, 5]]  # Выбор RGB-каналов
original_h, original_w = hypercube.shape[:2]

# Масштабирование изображения для модели
model_input_size = (256, 256)
scaled_image = cv2.resize(input_image, model_input_size)
mask = model.predict(scaled_image[np.newaxis, ...])[0, ..., 0] > 0.5

# Масштабирование маски обратно
mask_fullres = cv2.resize(mask.astype(np.uint8), 
                        (original_w, original_h), 
                        interpolation=cv2.INTER_NEAREST)

# Находим оптимальный канал
optimal_ch = find_optimal_channel(hypercube, mask_fullres)

# Визуализация
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# 1. Исходное изображение с маской
resized_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
resized_image = cv2.normalize(resized_image, None, 0, 255, cv2.NORM_MINMAX)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced_gray = clahe.apply(resized_image.astype(np.uint8))

ax1.imshow(enhanced_gray, cmap='gray')
ax1.set_title(f"Оптимальная длина волны: {450 + (optimal_ch * 2)} нм")

# Получаем координаты маски
y_idx, x_idx = np.where(mask_fullres)

if len(x_idx) > 0 and len(y_idx) > 0:
    # Вычисляем центр
    cx, cy = int(np.mean(x_idx)), int(np.mean(y_idx))
    
    # Создаем цветовую маску
    overlay = np.zeros((*enhanced_gray.shape, 4))
    overlay[mask_fullres > 0] = [1, 0, 0, 0.4]  # RGBA
    
    # Отображаем элементы
    ax1.imshow(overlay)
    ax1.scatter(cx, cy, marker='x', color='red', s=200, linewidth=2)
    
    # 2. Спектр в центре опухоли
    spectrum = hypercube[cy, cx, :]
    wavelengths = np.arange(450, 450 + len(spectrum)*2, 2)  # Генерация длин волн
    ax2.plot(wavelengths, spectrum)
    ax2.set_xlabel("Длина волны (нм)")
    ax2.set_ylabel("Интенсивность")
    ax2.set_title("Спектр из центра опухоли")
    ax2.grid(True)
else:
    ax1.text(10, 10, 'Опухоль не найдена', color='red', fontsize=14,
             bbox=dict(facecolor='white', alpha=0.8))
    ax2.text(0.5, 0.5, 'Опухоль не найдена', 
             horizontalalignment='center', 
             verticalalignment='center', 
             transform=ax2.transAxes, 
             color='red', 
             fontsize=14)

plt.tight_layout()
plt.show()
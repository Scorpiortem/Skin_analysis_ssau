import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
import os
import cv2

def select_file(title, filetypes):
    """Выбор файла через диалоговое окно"""
    app = QApplication(sys.argv)
    file_path, _ = QFileDialog.getOpenFileName(None, title, "", filetypes)
    app.quit()
    return file_path

def select_directory(title):
    """Выбор директории через диалоговое окно"""
    app = QApplication(sys.argv)
    path = QFileDialog.getExistingDirectory(None, title)
    return path

def load_data(image_dir, mask_dir):
    """Загрузка и предобработка данных"""
    images = []
    masks = []

    for img_name in os.listdir(image_dir):
        if not img_name.endswith('.png'):  # или другой формат ваших изображений
            continue

        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")  # или другой формат масок

        if not os.path.exists(mask_path):
            print(f"Пропущен файл: {img_name} (маска не найдена)")
            continue

        try:
            # Загрузка изображения и маски
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Преобразование в массивы
            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))

            # Нормализация изображения
            img = img.astype(np.float32) / 255.0

            # Бинаризация маски
            mask = (mask > 127).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)

            images.append(img)
            masks.append(mask)

        except Exception as e:
            print(f"Ошибка загрузки {img_name}: {str(e)}")

    return np.array(images), np.array(masks)

def evaluate_model(model, X_val, y_val, threshold=0.5):
    """Оценка модели с вычислением различных метрик"""
    # Проверка на наличие NaN или Inf
    if np.any(np.isnan(X_val)) or np.any(np.isinf(X_val)):
        raise ValueError("Данные X_val содержат NaN или Inf значения.")

    # Проверка на пустые данные
    if X_val.size == 0 or y_val.size == 0:
        raise ValueError("Данные X_val или y_val пусты.")

    # Предсказание вероятностей
    y_pred_prob = model.predict(X_val).flatten()

    # Бинаризация предсказаний
    y_pred = (y_pred_prob > threshold).astype(int)
    y_true = y_val.flatten().astype(int)

    # ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    print(f"ROC AUC: {roc_auc:.4f}")

    # ROC кривая
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Полный отчет классификации
    cr = classification_report(y_true, y_pred, target_names=['Non-Tumor', 'Tumor'])
    print("Classification Report:")
    print(cr)

    return roc_auc, cm, cr

if __name__ == "__main__":
    # Выбор файла модели
    model_path = select_file("Выберите файл модели", "HDF5 files (*.h5)")

    if model_path:
        # Загрузка модели
        model = tf.keras.models.load_model(model_path)
        print(f"Модель успешно загружена из {model_path}")

        # Выбор директорий с данными для валидации
        image_dir = select_directory("Выберите папку с изображениями для валидации")
        mask_dir = select_directory("Выберите папку с масками для валидации")

        if image_dir and mask_dir:
            # Загрузка данных для валидации
            X_val, y_val = load_data(image_dir, mask_dir)

            # Оценка модели
            evaluate_model(model, X_val, y_val)
        else:
            print("Ошибка: не выбраны директории с данными для валидации.")
    else:
        print("Файл модели не выбран.")

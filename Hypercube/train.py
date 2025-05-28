import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
from tensorflow.keras.optimizers import SGD
from datetime import datetime  # Добавим импорт

class OverfittingCallback(Callback):
    def __init__(self, patience=3):
        super().__init__()
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.best_val_loss = np.inf
        self.prev_train_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val_loss = logs.get("val_loss")
        current_train_loss = logs.get("loss")

        # Проверка на переобучение: validation loss растет, а training loss уменьшается
        if current_val_loss > self.best_val_loss and current_train_loss < self.prev_train_loss:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(f"\nОбнаружено переобучение! Обучение остановлено на эпохе {epoch + 1}")
        else:
            self.wait = 0
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss

        self.prev_train_loss = current_train_loss

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Обучение остановлено досрочно из-за переобучения")
def select_file(title, filetypes):
        """Выбор файла через диалоговое окно"""
        app = QApplication(sys.argv)
        file_path, _ = QFileDialog.getOpenFileName(None, title, "", filetypes)
        app.quit()
        return file_path
def fine_tune_model(model_path, new_image_dir, new_mask_dir, output_path=None):
    try:
        model = load_model(model_path)
    except:
        raise ValueError(f"Не удалось загрузить модель из {model_path}")

    # Генерация уникального имени файла если не указан явно
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = f'fine_tuned_model_{timestamp}.h5'

    # Заморозка начальных слоев
    for layer in model.layers[:15]:
        layer.trainable = False

    # Компиляция с новыми параметрами
    model.compile(
        optimizer=SGD(learning_rate=1e-5, momentum=0.9),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)]
    )

    # Загрузка новых данных
    try:
        new_images, new_masks = load_data(new_image_dir, new_mask_dir)
        if len(new_images) == 0:
            raise ValueError("Новые данные не содержат валидных изображений")
    except Exception as e:
        print(f"Ошибка загрузки данных: {str(e)}")
        return None, None

    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(
        new_images, new_masks,
        test_size=0.1,
        random_state=42
    )

    # Коллбеки
    callbacks = [
        EarlyStopping(patience=5, verbose=1, restore_best_weights=True),
        ModelCheckpoint(output_path, save_best_only=True, verbose=1),
        OverfittingCallback(patience=3)
    ]

    # Дообучение
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=4,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nМодель успешно сохранена в файл: {output_path}")
    return model, history, output_path  # Возвращаем путь к файлу

def unet(input_size=(256, 256, 3)):
    """Архитектура U-Net с улучшенной инициализацией и нормализацией"""
    inputs = Input(input_size)
    
    # Encoder
    # Block 1
    c1 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)
    
    # Block 2
    c2 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)
    
    # Block 3
    c3 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)
    
    # Block 4
    c4 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2,2))(c4)
    
    # Bottleneck
    c5 = Conv2D(1024, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Conv2D(1024, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # Decoder
    # Block 6
    u6 = UpSampling2D((2,2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    # Block 7
    u7 = UpSampling2D((2,2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    # Block 8
    u8 = UpSampling2D((2,2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    # Block 9
    u9 = UpSampling2D((2,2))(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)
    
    model = Model(inputs, outputs)
    return model

def load_data(image_dir, mask_dir, test_size=0.2, random_state=42):
    """Улучшенная загрузка данных с проверкой"""
    images = []
    masks = []
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    
    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith(valid_extensions):
            continue
            
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, f"{base_name}.tif_mask.png")
        
        # Проверка существования маски
        if not os.path.exists(mask_path):
            print(f"Warning: Маска для {img_name} не найдена! Пропускаем.")
            continue
            
        # Загрузка изображения
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Не удалось загрузить {img_path}")
            continue
            
        # Загрузка маски
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: Не удалось загрузить {mask_path}")
            continue
            
        # Предобработка
        img = cv2.resize(img, (256, 256)) 
        img = img.astype(np.float32) / 255.0  # Нормализация
        
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 127).astype(np.float32)  # Бинаризация
        mask = np.expand_dims(mask, axis=-1)    # Добавляем размерность канала
        
        images.append(img)
        masks.append(mask)
    
    # Проверка данных
    if len(images) == 0:
        raise ValueError("Не найдено валидных данных для обучения!")
        
    return np.array(images), np.array(masks)

def visualize_augmentation(images, masks, num_samples=3):
    """Визуализация аугментаций"""
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i+1)
        plt.imshow(images[i])
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(2, num_samples, i+num_samples+1)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title("Mask")
        plt.axis('off')
    plt.show()

def select_directory(title):
    """Выбор директории через диалоговое окно"""
    app = QApplication(sys.argv)
    path = QFileDialog.getExistingDirectory(None, title)
    return path    

def train_model():
    # Настройки
    BATCH_SIZE = 8
    EPOCHS = 50
    LR = 1e-4
    MODEL_PATH = "tumor_segmentation_model_v2.h5"
    
     # Выбор директорий
    print("Выберите директорию с изображениями:")
    image_dir = select_directory("Выберите папку с изображениями")
    
    print("Выберите директорию с масками:")
    mask_dir = select_directory("Выберите папку с масками")
    
    # Проверка выбранных путей
    if not image_dir or not mask_dir:
        print("Ошибка: не выбраны необходимые директории!")
        return
    
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print("Ошибка: выбранные директории не существуют!")
        return
    
    # Инициализация GPU (если доступно)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Используется GPU:", physical_devices[0])
    else:
        print("Используется CPU")

    # Загрузка данных
    try:
        images, masks = load_data(image_dir, mask_dir)
    except Exception as e:
        print(f"Ошибка загрузки данных: {str(e)}")
        return
    
    images, masks = load_data(image_dir, mask_dir)
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, 
        test_size=0.2, 
        random_state=42
    )
    
    # Визуализация примеров
    visualize_augmentation(X_train, y_train)
    
    # Создание модели
    model = unet()
    model.summary()
    
    # Компиляция с улучшенными параметрами
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5),
            'accuracy'
        ]
    )
    
    # Коллбеки
    callbacks = [
        EarlyStopping(patience=10, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=5, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1)
    ]
    
    # Обучение
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Визуализация обучения
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Evolution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['binary_io_u'], label='Training IoU')
    plt.plot(history.history['val_binary_io_u'], label='Validation IoU')
    plt.title('IoU Metric')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--fine-tune':
        # Выбор модели через диалоговое окно
        model_path = select_file("Выберите модель для дообучения", "HDF5 files (*.h5)")
        
        # Выбор директорий через диалоговые окна
        app = QApplication(sys.argv)
        print("Выберите директорию с новыми изображениями:")
        image_dir = QFileDialog.getExistingDirectory(None, "Выберите папку с новыми изображениями")
        
        print("Выберите директорию с новыми масками:")
        mask_dir = QFileDialog.getExistingDirectory(None, "Выберите папку с новыми масками")
        app.quit()

        # Проверка выбранных путей
        if not all([model_path, image_dir, mask_dir]):
            print("Ошибка: не выбраны все необходимые файлы и директории!")
            sys.exit(1)
            
        if not os.path.exists(model_path):
            print(f"Ошибка: файл модели {model_path} не существует!")
            sys.exit(1)

        # Вызов функции дообучения
        _, _, saved_path = fine_tune_model(model_path, image_dir, mask_dir)
        
        # Вывод результатов
        print(f"\nРезультаты дообучения:")
        print(f"1. Модель для дообучения: {model_path}")
        print(f"2. Новые изображения: {image_dir}")
        print(f"3. Новые маски: {mask_dir}")
        print(f"4. Сохранённая модель: {saved_path}")
    else:
        train_model()
import sys
import cv2
import numpy as np
import os
import csv
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                            QLabel, QPushButton, QVBoxLayout, QWidget,
                            QListWidget, QMessageBox, QDialog, QLineEdit)
from PyQt5.QtGui import QPainter, QImage, QPen, QColor, QPixmap
from PyQt5.QtCore import Qt, QPoint, QSize

class ImageWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Viewer")
        self.image_label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        self.points = []
        self.original_image = None
        self.scale_factor = 1.0
        self.original_size = QSize(0, 0)  # Хранение оригинальных размеров

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Получаем точные координаты с плавающей точкой
            x = event.pos().x() / self.scale_factor
            y = event.pos().y() / self.scale_factor
            self.points.append(QPoint(int(x), int(y)))  # Округляем до пикселей изображения
            self.update_image()

    def update_image(self):
        pixmap = QPixmap(self.original_pixmap)

        # Масштабируем pixmap до текущего размера виджета
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        painter = QPainter(scaled_pixmap)
        painter.setPen(QPen(QColor(0, 255, 0), 2))

        # Рисуем линии в масштабированных координатах
        for i in range(1, len(self.points)):
            p1 = self.points[i-1] * self.scale_factor
            p2 = self.points[i] * self.scale_factor
            painter.drawLine(p1, p2)

        painter.end()
        self.image_label.setPixmap(scaled_pixmap)

    def show_image(self, image_path):
        self.points = []
        self.original_image = cv2.imread(image_path)
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        h, w = self.original_image.shape[:2]

        qimage = QImage(
            self.original_image.data,
            w,
            h,
            self.original_image.strides[0],
            QImage.Format_RGB888
        )
        self.original_pixmap = QPixmap.fromImage(qimage)

        # Новый расчет масштаба
        self.original_size = QSize(w, h)
        self.adjust_scale_factor()

        # Отображаем изображение с правильным масштабом
        self.image_label.setPixmap(
            self.original_pixmap.scaled(
                self.original_size * self.scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )
        self.show()

    def adjust_scale_factor(self):
        """ Автоматический подбор масштаба при изменении размера окна """
        if not self.original_size.isValid():
            return

        # Получаем доступный размер
        available_size = QApplication.primaryScreen().availableSize()

        # Рассчитываем масштаб отдельно для ширины и высоты
        width_scale = available_size.width() / self.original_size.width()
        height_scale = available_size.height() / self.original_size.height()

        # Выбираем минимальный масштаб
        self.scale_factor = min(width_scale, height_scale) * 0.9  # 10% отступ

    def resizeEvent(self, event):
        """ Обработчик изменения размера окна """
        self.adjust_scale_factor()
        self.update_image()
        super().resizeEvent(event)

class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Control Panel")
        self.setGeometry(100, 100, 300, 400)

        self.image_window = ImageWindow()
        self.init_ui()
        self.current_image_index = 0
        self.image_files = []
        self.output_dir = ""
        self.patient_id = ""
        self.current_class = ""
        self.block_size = 10

    def init_ui(self):
        self.layout = QVBoxLayout()

        # Кнопки для выбора класса
        self.class_buttons = {}
        classes = ["MM", "NE", "SK"]
        for cls in classes:
            button = QPushButton(cls)
            button.setCheckable(True)
            button.clicked.connect(lambda _, c=cls: self.class_button_clicked(c))
            self.class_buttons[cls] = button
            self.layout.addWidget(button)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_class)
        self.layout.addWidget(self.reset_btn)

        self.load_btn = QPushButton("Load Images")
        self.load_btn.clicked.connect(self.load_images)
        self.layout.addWidget(self.load_btn)

        self.load_folder_btn = QPushButton("Load Folder")
        self.load_folder_btn.clicked.connect(self.load_folder)
        self.layout.addWidget(self.load_folder_btn)

        self.save_btn = QPushButton("Save Mask")
        self.save_btn.clicked.connect(self.save_mask)
        self.save_btn.setEnabled(False)
        self.layout.addWidget(self.save_btn)

        self.apply_mask_btn = QPushButton("Apply Mask to Block")
        self.apply_mask_btn.clicked.connect(self.apply_mask_to_block)
        self.apply_mask_btn.setEnabled(False)
        self.layout.addWidget(self.apply_mask_btn)

        self.next_block_btn = QPushButton("Next Block")
        self.next_block_btn.clicked.connect(self.next_block)
        self.next_block_btn.setEnabled(False)
        self.layout.addWidget(self.next_block_btn)

        self.next_btn = QPushButton("Next Image")
        self.next_btn.clicked.connect(self.next_image)
        self.next_btn.setEnabled(False)
        self.layout.addWidget(self.next_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.close)
        self.layout.addWidget(self.stop_btn)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def class_button_clicked(self, cls):
        if self.current_class == cls:
            self.class_buttons[cls].setChecked(False)
            self.current_class = ""
        else:
            for button in self.class_buttons.values():
                button.setChecked(False)
            self.class_buttons[cls].setChecked(True)
            self.current_class = cls

    def reset_class(self):
        for button in self.class_buttons.values():
            button.setChecked(False)
        self.current_class = ""

    def load_images(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if directory:
            # Извлекаем ID пациента из имени папки
            folder_name = os.path.basename(directory)
            match = re.search(r'unpacked_(\d+)', folder_name)
            if match:
                self.patient_id = match.group(1)
                self.image_files = [os.path.join(directory, f)
                                  for f in os.listdir(directory)
                                  if f.lower().endswith(('.png', '.jpg', '.bmp', '.tif'))]
                self.current_image_index = 0
                self.show_current_image()
                self.next_btn.setEnabled(True)
                self.apply_mask_btn.setEnabled(True)
                self.next_block_btn.setEnabled(True)
                self.layout.removeWidget(self.stop_btn)
                self.layout.addWidget(self.stop_btn)
            else:
                QMessageBox.critical(self, "Error", "Invalid folder name format! Should be 'unpacked_<ID>'")

    def load_folder(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Existing Annotations Folder")
        if directory:
            self.output_dir = directory
            self.image_files = [os.path.join(directory, f)
                                for f in os.listdir(directory)
                                if f.lower().endswith(('.png', '.jpg', '.bmp', '.tif'))]
            self.current_image_index = 0
            self.show_current_image()
            self.next_btn.setEnabled(True)

    def show_current_image(self):
        if self.current_image_index < len(self.image_files):
            self.image_window.show_image(
                self.image_files[self.current_image_index]
            )
            self.save_btn.setEnabled(True)
        else:
            QMessageBox.information(self, "Info", "All images processed!")

    def save_mask(self):
        if not self.current_class:
            QMessageBox.warning(self, "Warning", "Please select a class first!")
            return

        # Выбор папки для сохранения масок
        if not self.output_dir:
            self.output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for Masks")
            if not self.output_dir:
                QMessageBox.warning(self, "Warning", "Please select an output directory!")
                return

        # Создаем папку для аннотаций, если она еще не существует
        os.makedirs(self.output_dir, exist_ok=True)

        # Создание маски
        mask = np.zeros(self.image_window.original_image.shape[:2], dtype=np.uint8)
        pts = np.array([[p.x(), p.y()] for p in self.image_window.points])
        cv2.fillPoly(mask, [pts], 255)

        # Сохранение маски
        base_name = os.path.basename(self.image_files[self.current_image_index])
        mask_path = os.path.join(self.output_dir, f"{base_name}_mask.png")
        cv2.imwrite(mask_path, mask)

        # Сохранение данных в CSV
        csv_path = os.path.join(self.output_dir, "labels.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Image", "Class"])  # Записываем заголовок, если файл новый
            writer.writerow([base_name, self.current_class])

        QMessageBox.information(self, "Saved", f"Mask saved to:\n{mask_path}")

    def apply_mask_to_block(self):
        if not self.current_class:
            QMessageBox.warning(self, "Warning", "Please select a class first!")
            return

        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Please select an output directory first!")
            return

        # Получаем координаты текущей маски
        pts = np.array([[p.x(), p.y()] for p in self.image_window.points])

        # Применяем маску ко всем изображениям в текущем блоке
        for i in range(self.block_size):
            if self.current_image_index + i < len(self.image_files):
                image_path = self.image_files[self.current_image_index + i]
                image = cv2.imread(image_path)
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)

                base_name = os.path.basename(image_path)
                mask_path = os.path.join(self.output_dir, f"{base_name}_mask.png")
                cv2.imwrite(mask_path, mask)

                # Обновляем CSV файл
                csv_path = os.path.join(self.output_dir, "labels.csv")
                file_exists = os.path.isfile(csv_path)
                with open(csv_path, "a", newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["Image", "Class"])  # Записываем заголовок, если файл новый
                    writer.writerow([base_name, self.current_class])

        QMessageBox.information(self, "Applied", "Mask applied to the current block of images.")

    def next_block(self):
        self.current_image_index += self.block_size
        if self.current_image_index >= len(self.image_files):
            self.current_image_index = len(self.image_files) - 1
        self.image_window.points = []
        self.show_current_image()

    def next_image(self):
        self.current_image_index += 1
        if self.current_image_index >= len(self.image_files):
            self.current_image_index = len(self.image_files) - 1
        self.image_window.points = []
        self.show_current_image()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnnotationTool()
    window.show()
    sys.exit(app.exec_())

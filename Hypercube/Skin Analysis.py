import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import sys
import subprocess

def resource_path(relative_path):
    # Получает абсолютный путь к ресурсу для работы с PyInstaller
    try:
        base_path = sys._MEIPASS  # Временная папка при запуске из .exe
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class UniversityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ гиперкубов - Самарский университет")
        
        # Загрузка логотипа с обработкой путей
        try:
            logo_path = resource_path("logo.jpg")
            pil_image = Image.open(logo_path)
            self.logo = ImageTk.PhotoImage(pil_image)
        except Exception as e:
            print(f"Ошибка загрузки логотипа: {str(e)}")
            self.logo = None

        # Конфигурация окна
        self.root.geometry("1280x720")
        self.root.minsize(800, 600)
        self.root.configure(bg='white')

        # Инициализация стилей
        self.init_styles()
        self.create_widgets()

    def init_styles(self):
        """ Настройка цветовой схемы и стилей """
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Цветовая палитра
        self.primary_color = "#0d3880"  # Основной синий
        self.secondary_color = "#e98300"  # Оранжевый акцент
        
        # Стиль кнопок
        self.style.configure('TButton', 
            font=('Arial', 12, 'bold'),
            padding=10,
            foreground='white',
            background=self.primary_color,
            borderwidth=0
        )
        self.style.map('TButton',
            background=[('active', self.secondary_color)],
            foreground=[('active', 'white')]
        )

    def create_widgets(self):
        """ Создание элементов интерфейса """
        # Шапка
        header_frame = tk.Frame(self.root, bg=self.primary_color)
        header_frame.pack(fill='x', padx=10, pady=10)

        if self.logo:
            logo_label = tk.Label(header_frame, 
                                image=self.logo, 
                                bg=self.primary_color)
            logo_label.pack(side='left', padx=20, pady=10)

        title_label = tk.Label(header_frame,
            text="Гиперспектральный анализатор",
            font=('Arial', 24, 'bold'),
            fg='white',
            bg=self.primary_color
        )
        title_label.pack(side='left', padx=20)

        # Основная область с кнопками
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Список кнопок и скриптов
        buttons = [
            ("Создать гиперкуб", "pack_cube.py"),
            ("Распаковать гиперкуб", "unpack_cube.py"),
            ("Анализ спектра", "pixels.py"),
            ("Разметка данных", "divider.py"),
            ("Обучение модели", "train.py"),
            ("Анализ опухоли", "core.py")
        ]

        # Расположение кнопок в сетке 3x2
        row, col = 0, 0
        for text, script in buttons:
            btn = ttk.Button(main_frame,
                text=text,
                command=lambda s=script: self.run_script(s),
                style='TButton'
            )
            btn.grid(row=row, column=col, padx=15, pady=15, sticky='nsew')
            col = (col + 1) % 2
            if col == 0:
                row += 1

        # Настройка адаптивности
        for i in range(2):
            main_frame.columnconfigure(i, weight=1)
        for i in range(3):
            main_frame.rowconfigure(i, weight=1)

        # Статус бар
        self.status_bar = ttk.Label(self.root,
            text="Готов к работе",
            relief='sunken',
            anchor='w'
        )
        self.status_bar.pack(side='bottom', fill='x')

    def run_script(self, script_name):
        """ Запуск дочерних скриптов """
        self.status_bar.config(text=f"Запуск: {script_name}")
        try:
            script_path = resource_path(script_name)
            subprocess.Popen([sys.executable, script_path])
            self.status_bar.config(text=f"Скрипт {script_name} успешно запущен")
        except Exception as e:
            self.status_bar.config(text=f"Ошибка: {str(e)}")
            self.show_error_message(f"Ошибка запуска {script_name}:\n{str(e)}")

    def show_error_message(self, message):
        """ Отображение ошибки в отдельном окне """
        error_window = tk.Toplevel(self.root)
        error_window.title("Ошибка")
        
        msg_label = ttk.Label(error_window, 
                            text=message, 
                            padding=10,
                            foreground='red')
        msg_label.pack(padx=20, pady=10)
        
        ttk.Button(error_window,
                 text="OK",
                 command=error_window.destroy).pack(pady=10)

    def on_closing(self):
        """ Обработчик закрытия окна """
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = UniversityApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
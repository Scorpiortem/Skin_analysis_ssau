import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from matplotlib.widgets import RectangleSelector

class HypercubeAnalyzer:
    def __init__(self):
        self.hypercube = None
        self.click_coords = None
        self.fig = None
        self.ax = None

    def load_hypercube(self):
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Выберите файл гиперкуба",
            filetypes=[("NumPy files", "*.npy")]
        )
        if file_path:
            self.hypercube = np.load(file_path)
            print(f"Гиперкуб загружен. Размеры: {self.hypercube.shape}")
            return True
        return False

    def create_pseudo_color_image(self):
        composite = np.mean(self.hypercube, axis=2)
        return (255 * (composite - composite.min()) / 
               (composite.max() - composite.min())).astype(np.uint8)

    def show_pseudo_color_image(self):
        composite = self.create_pseudo_color_image()
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_title("Hypercube Viewer - Щелкните ЛКМ по пикселю | Закройте окно для выхода")
        self.im = self.ax.imshow(composite, cmap='viridis')
        
        def on_click(event):
            if event.inaxes == self.ax:
                x = int(event.xdata)
                y = int(event.ydata)
                self.click_coords = (y, x)
                self.plot_spectrum()

        self.fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

    def plot_spectrum(self):
        if self.hypercube is None or self.click_coords is None:
            return

        y, x = self.click_coords
        spectrum = self.hypercube[y, x, :]
        
        plt.figure(figsize=(10, 6))
        plt.plot(spectrum, 'b-', linewidth=2)
        plt.title(f"Спектр пикселя ({x}, {y})")
        plt.xlabel("Номер канала")
        plt.ylabel("Интенсивность")
        plt.grid(True)
        
        plt.annotate(f'Координаты: ({x}, {y})', 
                    xy=(0.05, 0.95), 
                    xycoords='axes fraction',
                    fontsize=10)
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"График сохранен как: {save_path}")
        
        plt.show()

    def run(self):
        if self.load_hypercube():
            self.show_pseudo_color_image()

if __name__ == "__main__":
    analyzer = HypercubeAnalyzer()
    analyzer.run()
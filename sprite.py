import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import os
import json

Neutral = "D:/AI/photos/neutral.png"
MAGIC_CHROMA_KEY = "#ff00ff"
MAGIC_RGBA = (255, 0, 255, 255)

window_height = 768
window_width = 512

# Indentation from the bottom-right corner of the screen
OFFSET_X = 20
OFFSET_Y = 0


class AilynFaceWidget:
    def __init__(self, root):
        self.root = root
        self.emotion_face = ""
        self.old_state = "abvc"
        #Make the window completely frameless and without a title bar
        self.root.overrideredirect(True)

        self.root.attributes('-topmost', True)

        self.root.wait_visibility(self.root)
        self.root.config(bg=MAGIC_CHROMA_KEY)
        self.root.wm_attributes('-transparentcolor', MAGIC_CHROMA_KEY)

        self.current_photo = None
        self.label = tk.Label(self.root, bg=MAGIC_CHROMA_KEY)
        self.label.pack()

        self.update_emotion()
        self.load_face_image(self.emotion_face)

        self.root.after(1, self.position_window)

    def filter_background_smart(self, img):

        #Flood Fill filter

        img = img.convert("RGBA")

        corners = [
            (0, 0),
            (img.width - 1, 0),
            (0, img.height - 1),
            (img.width - 1, img.height - 1)
        ]

        # thresh — sensitivity to dirty colors
        for corner in corners:
            current_color = img.getpixel(corner)
            if current_color != MAGIC_RGBA:
                ImageDraw.floodfill(img, corner, MAGIC_RGBA, thresh=100)

        return img

    def load_face_image(self, file_path):
        if not os.path.exists(file_path):
            self.root.title("File is not found by path")
            img = Image.new('RGB', (window_width, window_height), color=(255, 0, 255))
        else:
            try:
                img = Image.open(file_path).convert("RGBA")

                img.thumbnail((window_width, window_height), Image.Resampling.LANCZOS)

                # Создаем подложку нужного размера, если картинка после thumbnail меньше окна
                background = Image.new('RGBA', (window_width, window_height), (255, 255, 255, 255))

                offset = ((window_width - img.width) // 2, (window_height - img.height) // 2)
                background.paste(img, offset)
                img = self.filter_background_smart(background)

            except Exception as e:
                print(f"loading error of {file_path}: {e}")
                return

        self.current_photo = ImageTk.PhotoImage(img)
        self.label.config(image=self.current_photo)

    def position_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = screen_width - window_width - OFFSET_X
        y = screen_height - window_height - OFFSET_Y

        # Установить геометрию (Ширина x Высота + X_поз + Y_поз)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def update_emotion(self):
        try:
            with open("D:/AI/PythonProject/emotion_state.json", "r", encoding="utf-8") as file:
                core_affect = json.load(file)

            self.state = core_affect["emotion"]

            if self.state != self.old_state:
                if self.state == "controlled positive":
                    self.emotion_face = "D:/AI/photos/happiness.png"
                    print(f"Ailyn changed her face to {self.state}")

                if self.state == "controlled negative":
                    self.emotion_face = "D:/AI/photos/angry.png"
                    print(f"Ailyn changed her face to {self.state}")

                if self.state == "low positive":
                    self.emotion_face = "D:/AI/photos/neutral.png"
                    print(f"Ailyn changed her face to {self.state}")

                if self.state == "low negative":
                    self.emotion_face = "D:/AI/photos/sadness.png"
                    print(f"Ailyn changed her face to {self.state}")

                if self.state == "high positive":
                    self.emotion_face = "D:/AI/photos/suprise.png"
                    print(f"Ailyn changed her face to {self.state}")

                if self.state == "high negative":
                    self.emotion_face = "D:/AI/photos/fear.png"
                    print(f"Ailyn changed her face to {self.state}")

                if self.state == "neutral":
                    self.emotion_face = "D:/AI/photos/neutral0.png"
                    print(f"Ailyn changed her face to {self.state}")

                self.old_state = self.state

                self.load_face_image(self.emotion_face)

        except (json.JSONDecodeError, PermissionError):
            # Защита от коллизий, если файл читается в момент записи основной программой
            pass

        except Exception as e:
            print(f"error in update cycle: {e}")

        self.root.after(200, self.update_emotion)

if __name__ == "__main__":
    root = tk.Tk()
    app = AilynFaceWidget(root)

    print("Ailyn woke up")
    root.mainloop()
import numpy as np
import json

class VisualCortex:
    def __init__(self, retina_file="retina.json"):
        self.num_receptors = 900
        self.radius = 10
        self.receptors = []
        self.retina_file = retina_file

        with open(self.retina_file, "r") as file:
            self.receptors = json.load(file)

        '''
        except FileNotFoundError:
            alpha = 0.80 #density (0.5 - density is same everywhere)
            c = 1.0 / math.pow(self.num_receptors, alpha) #scale

            for n in range(1, self.num_receptors + 1):
                teta = n * 137.5 * (math.pi / 180.0) #golden angle

                d = c * math.pow(n, alpha) #Eccentricity

                x = d * math.cos(teta)
                y = d * math.sin(teta)


                distance = np.sqrt(x ** 2 + y ** 2)

                radius = 0.025 + distance * 0.12

                self.receptors.append({
                    "x": x,
                    "y": y,
                    "radius": radius
                })

            with open(self.retina_file, "w") as file:
                json.dump(self.receptors, file, indent=4)
        '''

        self.retina = np.zeros(self.num_receptors, dtype=np.float32)
        self.pixel_patches = []

    def process_vision(self, image):
        height, width = image.shape[0], image.shape[1]

        assert width == height

        fov_size = width
        center = fov_size / 2
        fov_radius = fov_size / 2

        for i, r in enumerate(self.receptors):
            # normalization
            pixel_col = (r["x"] + 1.0) / 2.0 * fov_size
            pixel_row = (r["y"] + 1.0) / 2.0 * fov_size

            # clip — без этого крайние рецепторы (x=1.0 или y=-1.0) вылетают за границу массива
            pixel_col = int(np.clip(pixel_col, 0, fov_size  - 1))
            pixel_row = int(np.clip(pixel_row, 0, fov_size  - 1))

            # радиус рецептора -> радиус патча в пикселях (используем меньшую сторону,
            # чтобы патч не растягивался в эллипс на прямоугольном 600x300 изображении)
            scale = fov_size  / 2.0
            pixel_radius = max(1, int(r["radius"] * scale))

            # вырезаем патч вокруг центра рецептора
            row_lo = max(0, pixel_row - pixel_radius)
            row_hi = min(height, pixel_row + pixel_radius + 1)
            col_lo = max(0, pixel_col - pixel_radius)
            col_hi = min(width, pixel_col + pixel_radius + 1)
            patch = image[row_lo:row_hi, col_lo:col_hi]

            # RGB -> яркость (формула чувствительности человеческого глаза), затем среднее по патчу
            luminance_patch = (
                    0.299 * patch[:, :, 0] +
                    0.587 * patch[:, :, 1] +
                    0.114 * patch[:, :, 2]
            )
            self.retina[i] = luminance_patch.mean()

        return self.retina


#whatever screen size her receptor numbers are same, it is just focus she gives to things --completed
#receptors number = 30x30 (standard hd) =  900 --completed
#rods and cones? (we dont need rods now, only cones then) --cons ---not completed (rods was made)
#she can change focus (even look at my zone)
#centralized focus --completed


#formula for radius(Make such research) --completed
#better positioning of receptors --completed


# in future make real colourful vision
# reaction time
# create eyes that dont link directly to game
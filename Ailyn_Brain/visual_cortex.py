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

        gray_image = (
                0.299 * image[:, :, 0] +
                0.587 * image[:, :, 1] +
                0.114 * image[:, :, 2]
        )

        scale = min(width, height) / 2.0

        for i, r in enumerate(self.receptors):
            pixel_col = int(np.clip((r["x"] + 1.0) / 2.0 * width, 0, width - 1))
            pixel_row = int(np.clip((r["y"] + 1.0) / 2.0 * height, 0, height - 1))
            pixel_radius = max(1, int(r["radius"] * scale))

            row_lo = max(0, pixel_row - pixel_radius)
            row_hi = min(height, pixel_row + pixel_radius + 1)
            col_lo = max(0, pixel_col - pixel_radius)
            col_hi = min(width, pixel_col + pixel_radius + 1)
            patch = gray_image[row_lo:row_hi, col_lo:col_hi]

            self.retina[i] = patch.mean() / 255.0

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
import tkinter as tk
import random
import json
import math
import time

def scalar_sigmoid(a, x):
    try:
        return a / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else float(a)

class AilynCore:
    def __init__(self):

        self.frame_count = 0

        # ----- neurons -----
        self.neurons_0 = [0.0] * 192
        self.neurons_1 = [0.0] * 192
        self.neurons_2 = [0.0] * 42
        self.neurons_3 = [0.0] * 10 #--------------------------------------

        self.neuron_errors_0 = [0.0] * 192
        self.neuron_errors_1 = [0.0] * 192
        self.neuron_errors_2 = [0.0] * 42 #----------------------------------

        self.predicted_receptors = [0.0] * 192

        self.hippocampus = {}

        # ----- weights -----
        try:
            with open("D:/AI/PythonProject/weights.json", "r", encoding="utf-8") as file:
                weights = json.load(file)
            self.slow_weights_1 = weights["weights_1"]
            self.slow_weights_2 = weights["weights_2"]
            self.slow_weights_3 = weights["weights_3"] #--------------------------------------------

        except FileNotFoundError:
            limit_1 = 1.0 / math.sqrt(192)  # initialization of Xavier Glorot
            limit_2 = 1.0 / math.sqrt(192)
            limit_3 = 1.0 / math.sqrt(42) #--------------------------------------
            # receptor = row, hidden neuron = column
            self.slow_weights_1 = [[random.uniform(-limit_1, limit_1) for _ in range(192)] for _ in range(192)]
            self.slow_weights_2 = [[random.uniform(-limit_2, limit_2) for _ in range(192)] for _ in range(42)]
            self.slow_weights_3 = [[random.uniform(-limit_3, limit_3) for _ in range(42)] for _ in range(10)]

        self.fast_weights_1 = [[0.0] * 192 for _ in range(192)]
        self.fast_weights_2 = [[0.0] * 192 for _ in range(42)]
        self.fast_weights_3 = [[0.0] * 42 for _ in range(10)]

        # working weights
        self.weights_1 = [[self.slow_weights_1[r][h] + self.fast_weights_1[r][h] for h in range(192)] for r in range(192)]
        self.weights_2 = [[self.slow_weights_2[h2][h] + self.fast_weights_2[h2][h] for h in range(192)] for h2 in range(42)]
        self.weights_3 = [[self.slow_weights_3[n][h] + self.fast_weights_3[n][h] for h in range(42)] for n in range(10)]

        self.slow_lr_multiplier = 0.1  # Долгая память учится в 10 раз медленнее
        self.fast_decay = 0.999  # Скорость "остывания" быстрой памяти (ближе к 1.0 = дольше помнит)

        '''
        self.weights = {
            "weights_1": self.weights_1,
            "weights_2": self.weights_2,
            "weights_3": self.weights_3 #------------------------------------------------
        }
        '''

        # ----- learning -----
        self.inference_speed = 0.08
        self.learning_rate = 0.01
        self.precision_1 = 1.0
        self.precision_2 = 1.0 # ------------------------------------
        self.precision_3 = 1.0 # ------------------------------------
        self.precision_prior = 0.02  # confidence of prior
        self.teacher_precision = 0.0
        self.base_learning_rate = 0.0005
        self.weight_lr_multiplier = 50.0
        self.target = [0.0] * 10
        # make optimal parameters

        # ----- core affect -----
        self.previous_energy = 0.0
        self.arousal = 0.0
        self.valence = 0.5
        self.emotional_stability = 0.02
        self.stress_threshold = 0.6
        self.activation = 0.4

        # ----- screen -----
        self.canvas_width = 400
        self.canvas_height = 300

        # ----- eyes -----
        self.radius = 20
        self.receptors = []
        cols, rows = 16, 12
        step_x = self.canvas_width / cols
        step_y = self.canvas_height / rows

        # moving center to middle of array
        for r in range(rows):
            for c in range(cols):
                cx = (c * step_x) + (step_x / 2)
                cy = (r * step_y) + (step_y / 2)
                self.receptors.append((cx, cy))

        self.retina_input = [0.0] * len(self.receptors)

    def predictive_programming(self, retina_input):

        # ----- input -----
        self.neurons_0 = list(retina_input)
        self.restart_energy()

        self.neuron_0_error()
        self.neuron_1_error()
        self.neuron_2_error()

        self.Total_energy = self.errors_energy_0 + self.errors_energy_1 + self.errors_energy_2 + self.errors_energy_prior_1  + self.errors_energy_prior_2  #-----change for two hidden layers --------------

        initial_energy = self.Total_energy

        for _ in range(15):
            self.restart_energy()

            self.neuron_1_learning()
            self.neuron_2_learning()
            self.neuron_3_learning()

            self.neuron_0_error()
            self.neuron_1_error()
            self.neuron_2_error()

            self.Total_energy = self.errors_energy_0 + self.errors_energy_1 + self.errors_energy_2 + self.errors_energy_prior_1  + self.errors_energy_prior_2 #-----change for two hidden layers --------------

        # Apply the Emotional Inertia formula
        self.valence_calculation(initial_energy)
        baseline_arousal = 0.15
        dynamic_arousal = math.tanh(self.Total_energy / 5.0)
        self.arousal = baseline_arousal + (1.0 - baseline_arousal) * dynamic_arousal  # check formula again

        self.weights_update()

        self.previous_energy = self.Total_energy

    # ----- OUTPUT -----

    def restart_energy(self):
        self.errors_energy_0 = 0
        self.errors_energy_1 = 0
        self.errors_energy_2 = 0
        self.errors_energy_prior_1 = 0
        self.errors_energy_prior_2 = 0

    def neuron_0_error(self):
        self.predicted_receptors = [0.0] * 192

        for r in range(192): #recepter = row
            for h in range(192): #hidden = column
                self.predicted_receptors[r] += self.neurons_1[h] * self.weights_1[r][h]
            error = self.neurons_0[r] - self.predicted_receptors[r]
            self.neuron_errors_0[r] = error
            self.neuron_errors_0[r] *= self.precision_1

            self.errors_energy_0 += 0.5 * (self.neuron_errors_0[r] ** 2)

    def neuron_1_error(self):
        predicted_neurons_1 = [0.0] * 192
        for h2 in range(42):
            for h in range(192):
                predicted_neurons_1[h] += self.neurons_2[h2] * self.weights_2[h2][h]

        for h in range(192):
            error = self.neurons_1[h] - predicted_neurons_1[h]
            self.neuron_errors_1[h] = error
            self.neuron_errors_1[h] *= self.precision_2

            self.errors_energy_1 += 0.5 * (self.neuron_errors_1[h] ** 2)

            # Считаем энергию приора (штраф за слишком высокую активацию мыслей)
            self.errors_energy_prior_1 += 0.5 * self.precision_prior * (self.neurons_1[h] ** 2)


    def neuron_2_error(self):

        predicted_neurons_2 = [0.0] * 42
        for n in range(10):
            for h in range(42):
                predicted_neurons_2[h] += self.neurons_3[n] * self.weights_3[n][h]

        for h in range(42):
            error = self.neurons_2[h] - predicted_neurons_2[h]
            self.neuron_errors_2[h] = error
            self.neuron_errors_2[h] *= self.precision_3

            self.errors_energy_2 += 0.5 * (self.neuron_errors_2[h] ** 2)

            # Считаем энергию приора (штраф за слишком высокую активацию мыслей)
            self.errors_energy_prior_2 += 0.5 * self.precision_prior * (self.neurons_2[h] ** 2)

    # ----- LEARNING 1 -----
    def neuron_1_learning(self):
        for h in range(192):  # hidden neuron
            influence_from_below = 0
            for r in range(192):  # receptor
                influence_from_below += self.neuron_errors_0[r] * self.weights_1[r][h]

            # В формулу обновления внедрен приор (утечка/decay), удерживающий нейроны от взрыва
            prior_leak = self.precision_prior * self.neurons_1[h]

            self.neurons_1[h] -= self.inference_speed * (self.neuron_errors_1[h] - influence_from_below + prior_leak)

    # ----- LEARNING 2 -----
    def neuron_2_learning(self):
        for h2 in range(42):  # hidden neuron
            influence_from_below = 0
            for h in range(192):  # receptor
                influence_from_below += self.neuron_errors_1[h] * self.weights_2[h2][h]

            # В формулу обновления внедрен приор (утечка/decay), удерживающий нейроны от взрыва
            prior_leak = self.precision_prior * self.neurons_2[h2]

            self.neurons_2[h2] -= self.inference_speed * (self.neuron_errors_2[h2] - influence_from_below + prior_leak)

    # ----- LEARNING 3 -----
    def neuron_3_learning(self):
        # сначала считаем "сырые" градиенты
        raw_grads = []
        for n in range(10):
            grad = sum(self.neuron_errors_2[h] * self.weights_3[n][h] for h in range(42))
            grad += self.teacher_precision * (self.target[n] - self.neurons_3[n])
            raw_grads.append(grad)

        # латеральное торможение: каждый нейрон тормозит соседей пропорционально своей активности
        inhibition_strength = 0.05
        mean_activity = sum(self.neurons_3) / 10
        for n in range(10):
            lateral = inhibition_strength * (self.neurons_3[n] - mean_activity)
            self.neurons_3[n] += self.inference_speed * (raw_grads[n] - lateral)

    # ----- VALENCE -----
    def valence_calculation(self, initial_energy):
        if not hasattr(self, 'previous_energy') or self.previous_energy == 0.0: # --------------------------------------
            self.previous_energy = initial_energy

        energy_delta = self.previous_energy - self.Total_energy

        raw_stimulus = (math.tanh(energy_delta / 5.0) + 1.0) / 2.0
        self.valence = (1 - self.emotional_stability) * self.valence + self.emotional_stability * raw_stimulus

    # ----- WEIGHTS -----
    def weights_update(self): # -------------------------------------------------------

        current_lr = self.learning_rate
        if self.teacher_precision > 0:
            current_lr *= self.weight_lr_multiplier #КОСТЫЛ

        fast_lr = current_lr
        slow_lr = current_lr * self.slow_lr_multiplier

        for h in range(192):
             for r in range(192):
                 grad = self.neuron_errors_0[r] * self.neurons_1[h]

                 self.fast_weights_1[r][h] += fast_lr * grad
                 self.fast_weights_1[r][h] *= self.fast_decay  # Быстрый вес понемногу тает

                 self.slow_weights_1[r][h] += slow_lr * grad

                 self.weights_1[r][h] = self.slow_weights_1[r][h] + self.fast_weights_1[r][h]

        for h2 in range(42):
            for h in range(192):
                grad = self.neuron_errors_1[h] * self.neurons_2[h2]

                self.fast_weights_2[h2][h] += fast_lr * grad
                self.fast_weights_2[h2][h] *= self.fast_decay

                self.slow_weights_2[h2][h] += slow_lr * grad

                self.weights_2[h2][h] = self.slow_weights_2[h2][h] + self.fast_weights_2[h2][h]

        for n in range(10):
            for h in range(42):
                grad = self.neuron_errors_2[h] * self.neurons_3[n]

                self.fast_weights_3[n][h] += fast_lr * grad
                self.fast_weights_3[n][h] *= self.fast_decay

                self.slow_weights_3[n][h] += slow_lr * grad

                self.weights_3[n][h] = self.slow_weights_3[n][h] + self.fast_weights_3[n][h]

        self.weights = {
            "weights_1": self.slow_weights_1,
            "weights_2": self.slow_weights_2,
            "weights_3": self.slow_weights_3 #-------------------------------------------
        }

    def save_weights(self):
        weights_to_save = self.weights
        try:
            with open("D:/AI/PythonProject/weights.json", "w", encoding="utf-8") as file:
                json.dump(weights_to_save, file, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[SYSTEM ERROR] Не удалось сохранить веса: {e}")

    def softmax(self):
        neuros_sum = 0.0
        softmax = [0.0] * 10
        for n in range(10):
            neuros_sum += math.exp(self.neurons_3[n] - max(self.neurons_3))
        for n in range(10):
            softmax[n] = math.exp(self.neurons_3[n] - max(self.neurons_3)) / neuros_sum
        return softmax


    def amygdala(self):
        factor = (self.arousal * (2 - self.valence)) / 2
        emotional_modulator = 0.5 + factor  # эмоции усиливают/ослабляют base_lr на ±50%
        self.learning_rate = self.base_learning_rate * emotional_modulator

        # БИОЛОГИЧЕСКИЙ ХАРД-КЛИППИНГ ВМЕСТО СИГМОИДЫ:
        # Слой 1 (Сенсоры): Сильнее верим холсту при стрессе (arousal).
        p1_raw = 1.0 + (self.arousal * 1.5) - (self.valence * 0.5)
        self.precision_1 = max(0.1, min(3.0, p1_raw))

        # Слой 2 (Мост): Внутренняя геометрия и стабильность структур.
        p2_raw = 1.0 + (self.valence * 0.5)
        self.precision_2 = max(0.1, min(2.0, p2_raw))

        # Слой 3 (Концепты): Убрана нестабильная дробь!
        # Если есть паника (arousal), убираем жесткость мнений верхнего уровня,
        # чтобы линии снизу могли перестроить концепты.
        p3_raw = 1.5 + (self.valence * 1.0) - (self.arousal * 1.5)
        self.precision_3 = max(0.1, min(2.5, p3_raw))


        if 0.52 <= self.arousal <= self.stress_threshold:
            if self.valence > 0.52:
                self.state = "controlled positive"
            if self.valence < 0.48:
                self.state = "controlled negative"
        if self.arousal < 0.48:
            if self.valence > 0.52:
                self.state = "low positive"
            if self.valence < 0.48:
                self.state = "low negative"
        if self.arousal > self.stress_threshold:
            if self.valence > 0.52:
                self.state = "high positive"
            if self.valence < 0.48:
                self.state = "high negative"
        if .048 <= self.arousal <= 0.52 and 0.48 <= self.valence <= 0.52:
            self.state = "neutral"

        state_data = {"emotion": self.state, "arousal": self.arousal, "valence": self.valence}
        with open("D:/AI/PythonProject/emotion_state.json", "w", encoding="utf-8") as file:
            json.dump(state_data, file)

    def eyes(self, mouse_x, mouse_y):
        for i, (cx, cy) in enumerate(self.receptors):
            # distance between center of receptor and cursor
            distance = math.sqrt((mouse_x - cx) ** 2 + (mouse_y - cy) ** 2)

            if distance < self.radius:
                # closer to center = more active receptor
                activation = 1.0 - (distance / self.radius)

                # storing activities
                # between [0.0, 1.0]
                self.retina_input[i] = min(1.0, self.retina_input[i] + activation)

        return self.retina_input

class Whiteboard:
    def __init__(self, root, process):

        self.last_thought_time = time.time()

        self.root = root
        self.process = process
        self.root.title("Whiteboard")
        self.root.geometry("400x300")

        self.pen_color = "black"
        self.brush_size = 5
        self.matrix = []

        self.canvas = tk.Canvas(root, bg="white") # Creates the drawing area. Everything you draw goes onto this canvas.
        self.canvas.pack() # pack() tells Tkinter to display the widget. Without it, the canvas exists but is invisible. #fill=tk.both means stretch horizontally and vertically. Expand=True means give free space to canvas

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def paint(self, event):
        width = self.brush_size
        color = self.pen_color

        # hasattr = Does this object have an attribute named old_x/old_y?
        if hasattr(self, 'old_x') and hasattr(self, 'old_y'):
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                    width=width, fill=color,
                                    capstyle=tk.ROUND, smooth=tk.TRUE)
        self.old_x = event.x
        self.old_y = event.y

        self.process.eyes(event.x, event.y)
        current_time = time.time()
        if current_time - self.last_thought_time > 0.1:
            self.process.predictive_programming(self.process.retina_input)
            self.process.amygdala()
            self.last_thought_time = current_time



    # underscore means argument is required, but I intentionally don't use it (it is not python rule, it is just programmers rule)
    def reset(self, _event):
        # Reset previous coordinate so lines don't connect after lifting the mouse
        if hasattr(self, 'old_x'):
            del self.old_x
        if hasattr(self, 'old_y'):
            del self.old_y

        self.process.save_weights()

    def close_app(self, _event=None):
        root.destroy()  # Safely destroys the window and stops mainloop

    def submit(self, correct_answer):

        self.process.predictive_programming(self.process.retina_input)
        self.process.amygdala()
        self.process.save_weights()

        probability = self.process.softmax()
        probabilities = probability.index(max(probability))

        print(f"Prediction: {probabilities} | Energy: {self.process.Total_energy:.3f} | Valence: {self.process.valence:.3f} | Arousal: {self.process.arousal:.3f}")
        print(f"Correct answer: {correct_answer}")
        self.correct_answer = correct_answer

        self.process.teacher_precision = 3.0
        self.process.target[correct_answer] = 1.0

        for _ in range(10):
            self.process.predictive_programming(self.process.retina_input)
            self.process.amygdala()

        self.process.save_weights()

        self.process.predictive_programming(self.process.retina_input)
        self.process.amygdala()
        self.process.save_weights()

        self.process.retina_input = [0.0] * len(self.process.receptors)
        self.canvas.delete("all")
        self.process.teacher_precision = 0.0
        self.process.target = [0.0] * 10
        self.process.neurons_1 = [0.0] * 192
        self.process.neurons_2 = [0.0] * 42
        self.process.neurons_3 = [0.0] * 10

if __name__ == "__main__":
    root = tk.Tk() #Creates the main application window.
    process = AilynCore()

    app = Whiteboard(root, process)

    zero_button = tk.Button(root, text="0", command=lambda: app.submit(0))
    #In command, we don't write function, but what it returns

    one_button = tk.Button(root, text="1", command=lambda: app.submit(1))
    two_button = tk.Button(root, text="2", command=lambda: app.submit(2))
    three_button = tk.Button(root, text="3", command=lambda: app.submit(3))
    four_button = tk.Button(root, text="4", command=lambda: app.submit(4))
    five_button = tk.Button(root, text="5", command=lambda: app.submit(5))
    six_button = tk.Button(root, text="6", command=lambda: app.submit(6))
    seven_button = tk.Button(root, text="7", command=lambda: app.submit(7))
    eight_button = tk.Button(root, text="8", command=lambda: app.submit(8))
    nine_button = tk.Button(root, text="9", command=lambda: app.submit(9))


    exit_button = tk.Button(root, text="Exit", command=app.close_app)
    exit_button.pack(side=tk.LEFT, padx=5, pady=5)

    zero_button.pack(side=tk.LEFT, padx=5, pady=5)
    one_button.pack(side=tk.LEFT, padx=5, pady=5)
    two_button.pack(side=tk.LEFT, padx=5, pady=5)
    three_button.pack(side=tk.LEFT, padx=5, pady=5)
    four_button.pack(side=tk.LEFT, padx=5, pady=5)
    five_button.pack(side=tk.LEFT, padx=5, pady=5)
    six_button.pack(side=tk.LEFT, padx=5, pady=5)
    seven_button.pack(side=tk.LEFT, padx=5, pady=5)
    eight_button.pack(side=tk.LEFT, padx=5, pady=5)
    nine_button.pack(side=tk.LEFT, padx=5, pady=5)

    root.mainloop()
import json
import math
import numpy as np


def scalar_sigmoid(a, x):
    try:
        return a / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else float(a)

class AilynCore:
    def __init__(self):

        # ----- time -----
        self.frame_count = 0
        self.time_neurons = np.zeros(16)
        limit_time = 1.0 / math.sqrt(16)
        self.weights_time_to_3 = np.random.uniform(-limit_time, limit_time, (16, 692))
        self.weights_time_to_2 = np.random.uniform(-limit_time, limit_time, (16, 1500))

        # ----- neurons -----
        self.neurons_0 = np.zeros(900)
        self.neurons_1 = np.zeros(2500)
        self.neurons_2 = np.zeros(1500)
        self.neurons_3 = np.zeros(692)
        self.neurons_4 = np.zeros(10)

        self.neuron_errors_0 = np.zeros(900)
        self.neuron_errors_1 = np.zeros(2500)
        self.neuron_errors_2 = np.zeros(1500)
        self.neuron_errors_3 = np.zeros(692)

        self.predicted_receptors = np.zeros(900)

        #self.hippocampus = {}

        # ----- weights -----
        try:
            with open("D:/AI/PythonProject/Ailyn/weights.json", "r", encoding="utf-8") as file:
                weights = json.load(file)
            self.slow_weights_1 = np.array(weights["weights_1"])
            self.slow_weights_2 = np.array(weights["weights_2"])
            self.slow_weights_3 = np.array(weights["weights_3"])
            self.slow_weights_4 = np.array(weights["weights_4"])
            self.weights_time_to_2 = np.array(weights["weights_time_2"])
            self.weights_time_to_3 = np.array(weights["weights_time_3"])

        except FileNotFoundError:
            limit_1 = 1.0 / math.sqrt(900)  # initialization of Xavier Glorot
            limit_2 = 1.0 / math.sqrt(2500)
            limit_3 = 1.0 / math.sqrt(1500)
            limit_4 = 1.0 / math.sqrt(692)

            self.slow_weights_1 = np.random.uniform(-limit_1, limit_1, (900, 2500))
            self.slow_weights_2 = np.random.uniform(-limit_2, limit_2, (2500, 1500))
            self.slow_weights_3 = np.random.uniform(-limit_3, limit_3, (1500, 692))
            self.slow_weights_4 = np.random.uniform(-limit_4, limit_4, (692, 10))

        self.fast_weights_1 = np.zeros((900, 2500))
        self.fast_weights_2 = np.zeros((2500, 1500))
        self.fast_weights_3 = np.zeros((1500, 692))
        self.fast_weights_4 = np.zeros((692, 10))

        self.weights_1 = self.slow_weights_1 + self.fast_weights_1
        self.weights_2 = self.slow_weights_2 + self.fast_weights_2
        self.weights_3 = self.slow_weights_3 + self.fast_weights_3
        self.weights_4 = self.slow_weights_4 + self.fast_weights_4

        self.slow_lr_multiplier = 0.1  # speed of forgetting of slow memory
        self.fast_decay = 0.999  # speed of forgetting of fast memory

        # ----- learning -----
        self.inference_speed = 0.01
        self.learning_rate = 0.01
        self.precision_1 = 1.0
        self.precision_2 = 1.0
        self.precision_3 = 1.0
        self.precision_4 = 1.0
        self.precision_prior = 0.02  # confidence of prior
        self.base_learning_rate = 0.0005
        self.weight_lr_multiplier = 50.0
        # make optimal parameters

        # ----- Energy -----
        self.previous_energy = 0.0
        self.Total_energy = 0.0

        # ----- time -----
        self.frame_decay = 0.9

        # ---- efference copy -----
        limit_ef = 1.0 / math.sqrt(10)
        self.weights_efference = np.random.uniform(-limit_ef, limit_ef,(10, 900))
        self.simulation_error = np.zeros(900)

        # ----- actions -----
        self.focus_x = 200.0
        self.focus_y = 300.0
        self.eye_speed = 15.0

    def update_oscillators(self):
        self.frame_count += 1

        # 8 pairs for 16 neurons
        i = np.arange(8)

        # denomitor
        div_term = 10000 ** (2 * i / 16)

        # 3. sunchronization of oscillators: sin and cos
        self.time_neurons[0::2] = np.sin(self.frame_count / div_term)
        self.time_neurons[1::2] = np.cos(self.frame_count / div_term)

    def mental_simulation(self, steps = 3):
        real_neurons_0 = self.neurons_0.copy()
        real_neurons_1 = self.neurons_1.copy()
        real_neurons_2 = self.neurons_2.copy()
        real_neurons_3 = self.neurons_3.copy()
        real_neurons_4 = self.neurons_4.copy()

        panic_triggered = False

        for _ in range(steps):
            self.neurons_0 = np.copy(self.predicted_receptors)

            for _ in range(15):
                self.neuron_1_learning()
                self.neuron_2_learning()
                self.neuron_3_learning()
                self.neuron_4_learning()

                top_down_prediction = np.dot(self.weights_1, self.neurons_1)
                efference_prediction = np.dot(self.neurons_4, self.weights_efference)
                self.predicted_receptors = top_down_prediction + efference_prediction

                self.neuron_0_error()
                self.neuron_1_error()
                self.neuron_2_error()
                self.neuron_3_error()

            # if panic is triggered, we can break the loop early
            if self.errors_energy_0 > 5.0:
                panic_triggered = True

        # returning to real neurons after mental simulation
        self.neurons_0 = real_neurons_0
        self.neurons_1 = real_neurons_1
        self.neurons_2 = real_neurons_2
        self.neurons_3 = real_neurons_3
        self.neurons_4 = real_neurons_4

        if panic_triggered:
            self.neurons_4 *= 0.5

    def predictive_programming(self, retina_signal):
        self.update_oscillators()

        # ----- input -----
        self.neurons_0 = retina_signal
        self.neurons_1 *= self.frame_decay
        self.neurons_2 *= self.frame_decay
        self.neurons_3 *= self.frame_decay

        self.neuron_0_error()
        self.neuron_1_error()
        self.neuron_2_error()
        self.neuron_3_error()

        self.Total_energy = self.errors_energy_0 + self.errors_energy_1 + self.errors_energy_2 + self.errors_energy_3 + self.errors_energy_prior_1  + self.errors_energy_prior_2 + self.errors_energy_prior_3

        for _ in range(15):
            self.neuron_1_learning()
            self.neuron_2_learning()
            self.neuron_3_learning()
            self.neuron_4_learning()

            self.neuron_0_error()
            self.neuron_1_error()
            self.neuron_2_error()
            self.neuron_3_error()

            self.Total_energy = self.errors_energy_0 + self.errors_energy_1 + self.errors_energy_2 + self.errors_energy_3 + self.errors_energy_prior_1  + self.errors_energy_prior_2 + self.errors_energy_prior_3 #-----change for two hidden layers --------------

            self.simulation_error = self.neurons_0 - self.predicted_receptors

        self.mental_simulation(steps=3)

        self.weights_update()

        self.previous_energy = self.Total_energy

    # ----- OUTPUT -----
    def get_actions(self):
        hands_logits = self.neurons_4[0:5]
        eyes_logits = self.neurons_4[5:10]

        hands_exp = np.exp(hands_logits - np.max(hands_logits))
        hands_probs = hands_exp / np.sum(hands_exp)

        eyes_exp = np.exp(eyes_logits - np.max(eyes_logits))
        eyes_probs = eyes_exp / np.sum(eyes_exp)

        return hands_probs, eyes_probs

    def neuron_0_error(self):
        top_down_prediction = np.dot(self.weights_1, self.neurons_1)

        efference_prediction = np.dot(self.neurons_4, self.weights_efference)

        self.predicted_receptors = top_down_prediction + efference_prediction

        error = self.neurons_0 - self.predicted_receptors
        self.neuron_errors_0 = error * self.precision_1

        self.errors_energy_0 = np.sum(0.5 * (self.neuron_errors_0 ** 2))

    def neuron_1_error(self):
        predicted_neurons_1 = np.dot(self.weights_2, self.neurons_2)

        error = self.neurons_1 - predicted_neurons_1
        self.neuron_errors_1 = error * self.precision_2

        self.errors_energy_1 = np.sum(0.5 * (self.neuron_errors_1 ** 2))
        self.errors_energy_prior_1 = np.sum(0.5 * self.precision_prior * (self.neurons_1 ** 2))


    def neuron_2_error(self):
        predicted_from_above = np.dot(self.weights_3, self.neurons_3)
        time_context = np.dot(self.time_neurons, self.weights_time_to_2)
        predicted_neurons_2 = predicted_from_above + time_context
        error = self.neurons_2 - predicted_neurons_2
        self.neuron_errors_2 = error * self.precision_3
        self.errors_energy_2 = np.sum(0.5 * (self.neuron_errors_2 ** 2))
        self.errors_energy_prior_2 = np.sum(0.5 * self.precision_prior * (self.neurons_2 ** 2))

    def neuron_3_error(self):
        predicted_from_above = np.dot(self.weights_4, self.neurons_4)
        time_context = np.dot(self.time_neurons, self.weights_time_to_3)
        predicted_neurons_3 = predicted_from_above + time_context
        error = self.neurons_3 - predicted_neurons_3
        self.neuron_errors_3 = error * self.precision_4
        self.errors_energy_3 = np.sum(0.5 * (self.neuron_errors_3 ** 2))
        self.errors_energy_prior_3 = np.sum(0.5 * self.precision_prior * (self.neurons_3 ** 2))

    # ----- LEARNING 1 -----
    def neuron_1_learning(self):
        influence_from_below = np.dot(self.neuron_errors_0, self.weights_1)
        prior_leak = self.precision_prior * self.neurons_1
        self.neurons_1 -= self.inference_speed * (self.neuron_errors_1 - influence_from_below + prior_leak)

    # ----- LEARNING 2 -----
    def neuron_2_learning(self):
        influence_from_below = np.dot(self.neuron_errors_1, self.weights_2)
        prior_leak = self.precision_prior * self.neurons_2
        self.neurons_2 -= self.inference_speed * (self.neuron_errors_2 - influence_from_below + prior_leak)

    # ----- LEARNING 3 -----
    def neuron_3_learning(self):
        influence_from_below = np.dot(self.neuron_errors_2, self.weights_3)
        prior_leak = self.precision_prior * self.neurons_3
        self.neurons_3 -= self.inference_speed * (self.neuron_errors_3 - influence_from_below + prior_leak)

    def neuron_4_learning(self):
        grads = np.dot(self.neuron_errors_3, self.weights_4)

        # lateral inhibition
        inhibition_strength = 0.05
        mean_activity = np.mean(self.neurons_4)
        lateral = inhibition_strength * (self.neurons_4 - mean_activity)

        self.neurons_4 += self.inference_speed * (grads - lateral)

    # ----- WEIGHTS -----
    def weights_update(self):
        current_lr = self.learning_rate
        fast_lr = current_lr
        slow_lr = current_lr * self.slow_lr_multiplier

        grad = np.outer(self.neuron_errors_0, self.neurons_1)
        self.fast_weights_1 += fast_lr * grad
        self.fast_weights_1 *= self.fast_decay

        self.slow_weights_1 += slow_lr * grad

        self.weights_1 = self.slow_weights_1 + self.fast_weights_1

        grad = np.outer(self.neuron_errors_1, self.neurons_2)
        self.fast_weights_2 += fast_lr * grad
        self.fast_weights_2 *= self.fast_decay

        self.slow_weights_2 += slow_lr * grad

        self.weights_2 = self.slow_weights_2 + self.fast_weights_2

        grad = np.outer(self.neuron_errors_2, self.neurons_3)
        self.fast_weights_3 += fast_lr * grad
        self.fast_weights_3 *= self.fast_decay
        self.slow_weights_3 += slow_lr * grad
        self.weights_3 = self.slow_weights_3 + self.fast_weights_3

        grad = np.outer(self.neuron_errors_3, self.neurons_4)
        self.fast_weights_4 += fast_lr * grad
        self.fast_weights_4 *= self.fast_decay
        self.slow_weights_4 += slow_lr * grad
        self.weights_4 = self.slow_weights_4 + self.fast_weights_4

        self.weights_efference += fast_lr * np.outer(self.neurons_4, self.simulation_error)
        self.weights_efference *= self.fast_decay

        self.weights_time_to_2 += fast_lr * np.outer(self.time_neurons, self.neuron_errors_2)
        self.weights_time_to_2 *= self.fast_decay

        self.weights_time_to_3 += fast_lr * np.outer(self.time_neurons, self.neuron_errors_3)
        self.weights_time_to_3 *= self.fast_decay

        self.weights = {
            "weights_1": self.slow_weights_1.tolist(),
            "weights_2": self.slow_weights_2.tolist(),
            "weights_3": self.slow_weights_3.tolist(),
            "weights_4": self.slow_weights_4.tolist(),
            "weights_time_2": self.weights_time_to_2.tolist(),
            "weights_time_3": self.weights_time_to_3.tolist()
        }

    def save_weights(self):
        weights_to_save = self.weights
        try:
            with open("D:/AI/PythonProject/Ailyn/weights.json", "w", encoding="utf-8") as file:
                json.dump(weights_to_save, file, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[SYSTEM ERROR] could not save weights: {e}")

    def move_eyes(self, eyes_probs):
        eye_action = np.argmax(eyes_probs)

        if eye_action == 0:
            self.focus_y -= self.eye_speed
        elif eye_action == 1:
            self.focus_y += self.eye_speed
        elif eye_action == 2:
            self.focus_x -= self.eye_speed
        elif eye_action == 3:
            self.focus_x += self.eye_speed
        # 4 for if she does nothing

        # boundaries of vision
        self.focus_x = max(200.0, min(750.0, self.focus_x))
        self.focus_y = max(200.0, min(500.0, self.focus_y))

    def sleep(self):
        print("[SYSTEM] Ailyn is sleeping...")

        consolidation_rate = 0.5

        self.slow_weights_1 += self.fast_weights_1 * consolidation_rate
        self.slow_weights_2 += self.fast_weights_2 * consolidation_rate
        self.slow_weights_3 += self.fast_weights_3 * consolidation_rate
        self.slow_weights_4 += self.fast_weights_4 * consolidation_rate

        self.fast_weights_1 *= 0.1
        self.fast_weights_2 *= 0.1
        self.fast_weights_3 *= 0.1
        self.fast_weights_4 *= 0.1

        # dont collapse after waking up due to time weigths
        if not hasattr(self, 'slow_weights_time_2'):
            self.slow_weights_time_2 = np.zeros_like(self.weights_time_to_2)
            self.slow_weights_time_3 = np.zeros_like(self.weights_time_to_3)
            self.slow_weights_efference = np.zeros_like(self.weights_efference)

        self.slow_weights_time_2 += self.weights_time_to_2 * consolidation_rate
        self.slow_weights_time_3 += self.weights_time_to_3 * consolidation_rate
        self.slow_weights_efference += self.weights_efference * consolidation_rate

        self.weights_time_to_2 = self.slow_weights_time_2 + self.weights_time_to_2 * 0.1
        self.weights_time_to_3 = self.slow_weights_time_3 + self.weights_time_to_3 * 0.1
        self.weights_efference = self.slow_weights_efference + self.weights_efference * 0.1

        self.weights_1 = self.slow_weights_1.copy()
        self.weights_2 = self.slow_weights_2.copy()
        self.weights_3 = self.slow_weights_3.copy()
        self.weights_4 = self.slow_weights_4.copy()

        self.weights = {
            "weights_1": self.slow_weights_1.tolist(),
            "weights_2": self.slow_weights_2.tolist(),
            "weights_3": self.slow_weights_3.tolist(),
            "weights_4": self.slow_weights_4.tolist(),
            "weights_time_2": self.weights_time_to_2.tolist(),
            "weights_time_3": self.weights_time_to_3.tolist()
        }

        self.neurons_0.fill(0.0)
        self.neurons_1.fill(0.0)
        self.neurons_2.fill(0.0)
        self.neurons_3.fill(0.0)
        self.neurons_4.fill(0.0)

        self.Total_energy = 0.0

        self.save_weights()
        print("[SYSTEM] Ailyn woke up from sleep")

#if __name__ == "__main__":

import math
import numpy as np
import json

class LimbicSystem:
    def __init__(self):
        self.arousal = 0.0
        self.baseline_arousal = 0.15
        self.valence = 0.5
        self.emotional_stability = 0.02
        self.stress_threshold = 0.6
        self.activation = 0.4
        self.previous_energy = 0.0
        self.expected_reward = 0.0
        self.dopamine_clearance_rate = 0.1
        self.dopamine_reuptake_rate = 0.1
        self.adenosine = 0.0
        self.state = "neutral"

    # ----- VALENCE -----
    def valence_calculation(self, Total_energy):
        energy_delta = self.previous_energy - Total_energy

        raw_stimulus = (math.tanh(energy_delta / 5.0) + 1.0) / 2.0
        self.valence = (1 - self.emotional_stability) * self.valence + self.emotional_stability * raw_stimulus

    def arousal_calculation(self, Total_energy):
        dynamic_arousal = math.tanh(Total_energy / 250.0)
        self.arousal = self.baseline_arousal + (1.0 - self.baseline_arousal) * dynamic_arousal
        #nonadrenaline function

    def dopamine_release(self, brain, external_reward):
        RPE = external_reward - self.expected_reward

        self.expected_reward += 0.1 * RPE

        dopamine = math.tanh(RPE / 50.0)

        decay = 0.9999

        brain.slow_weights_1 = (brain.slow_weights_1 * decay) + (dopamine * self.dopamine_clearance_rate * brain.fast_weights_1)
        brain.slow_weights_2 = (brain.slow_weights_2 * decay) + (dopamine * self.dopamine_clearance_rate * brain.fast_weights_2)
        brain.slow_weights_3 = (brain.slow_weights_3 * decay) + (dopamine * self.dopamine_clearance_rate * brain.fast_weights_3)
        brain.slow_weights_4 = (brain.slow_weights_4 * decay) + (dopamine * self.dopamine_clearance_rate * brain.fast_weights_4)

        self.valence = (1 - self.emotional_stability) * self.valence + self.emotional_stability * ((dopamine + 1.0) / 2.0)

    def amygdala(self, brain):
        self.adenosine += 0.00002 + (brain.Total_energy / 5000000.0)
        self.arousal_calculation(brain.Total_energy)
        self.valence_calculation(brain.Total_energy)

        factor = (self.arousal * (2 - self.valence)) / 2
        emotional_modulator = 0.5 + factor
        brain.learning_rate = brain.base_learning_rate * emotional_modulator
        brain.inference_speed = 0.08 * (1.0 + self.arousal)

        brain.precision_1 = math.exp(self.arousal - self.valence * 0.5)
        brain.precision_2 = math.exp(self.valence * 0.5)
        brain.precision_3 = math.exp(self.valence - self.arousal * 1.5)
        brain.precision_4 = math.exp(-self.arousal * 0.5)

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
        with open("D:/AI/PythonProject/Ailyn/emotion_state.json", "w", encoding="utf-8") as file:
            json.dump(state_data, file)

        if self.adenosine > 1.0:
            brain.sleep()

            self.adenosine = 0.0
            self.arousal = self.baseline_arousal

        self.previous_energy = brain.Total_energy


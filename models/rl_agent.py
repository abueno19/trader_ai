# Vamos a crear el modelo de rl para el bot de trading

# pip install gymnasium
# pip install numpy
# pip install pandas
# pip install yfinance

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from gymnasium import spaces

class RLAgent(gym.Env):
    def __init__(self):
        """
        Funcion para iniciar el modelo de RL
        """
        # Primer numero de acciones es:
            # 3 acciones posibles: comprar, vender, mantener
        # Segundo numero de acciones es:
            # 10 acciones que representa el take profit
        # Tercer numero de acciones es:
            # 10 acciones que representa el stop loss
        self.action_space = spaces.MultiDiscrete([3,10,10,10])

        # Create a shape tuple to define the dimensions of the observation space
        self.features = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'ia_prediction_moving_average']
        self.window_size = 30  # Tamaño de la ventana de datos históricos# Establecer el tamaño de la ventana a 30 días
        self.lookback_days = 30
        shape = (self.window_size, len(self.features))

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

# Buffer para almacenar datos históricos
        self.historical_data = []

    def ia_observation(self, data):
        # Agregar nuevos datos al buffer histórico
        self.historical_data.append([data[feature] for feature in self.features])

        # Mantener solo los últimos window_size datos
        if len(self.historical_data) > self.window_size:
            self.historical_data = self.historical_data[-self.window_size:]

        # Crear matriz de observación con datos históricos
        observation = np.array(self.historical_data)

        # Si no hay suficientes datos históricos, rellenar con ceros
        if len(observation) < self.window_size:
            padding = np.zeros((self.window_size - len(observation), len(self.features)))
            observation = np.vstack([padding, observation])

        return observation

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:

import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers import Dense
import numpy as np


class PricePredictionModel(tf.keras.Model): # type: ignore
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(PricePredictionModel, self).__init__()
        self.lstm = Sequential([
            LSTM(hidden_dim, return_sequences=True, input_shape=(None, input_dim)),
            *[LSTM(hidden_dim, return_sequences=True) for _ in range(num_layers - 1)],
            LSTM(hidden_dim),
            Dense(1, activation='sigmoid')
        ])
        self.lstm.summary()

    def call(self, x):
        return self.lstm(x)

class PredictiveModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.features = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal_Line']
        self.model = None
        self.criterion = BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # type: ignore

    def load_data(self):
        data = pd.read_csv(self.data_path, parse_dates=['Datetime'])
        data['target'] = (data['close_price'] > data['open_price']).astype(int)
        data = data.dropna()
        return data

    def preprocess_data(self, data, sequence_length=10):
        X = data[self.features].values
        y = data['target'].values

        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("Los datos de entrada contienen valores NaN o infinitos antes de la normalización.")

        sequences = []
        targets = []
        for i in range(len(X) - sequence_length):
            sequences.append(X[i:i+sequence_length])
            targets.append(y[i+sequence_length])

        X_sequences = tf.convert_to_tensor(sequences, dtype=tf.float32)
        y_sequences = tf.convert_to_tensor(targets, dtype=tf.float32)

# División manual del conjunto de datos
        dataset_size = len(X_sequences)
        test_size = int(0.2 * dataset_size)
        train_size = dataset_size - test_size

        indices = tf.range(dataset_size)
        train_indices = indices[:train_size]
        test_indices = indices[test_size:]

        X_train = tf.gather(X_sequences, train_indices)
        X_test = tf.gather(X_sequences, test_indices)
        y_train = tf.gather(y_sequences, train_indices)
        y_test = tf.gather(y_sequences, test_indices)

        return X_train, X_test, y_train, y_test

    def initialize_model(self):
        input_dim = len(self.features)
        hidden_dim = 64
        num_layers = 10
        self.model = PricePredictionModel(input_dim, hidden_dim, num_layers)
        self.model.compile(optimizer=self.optimizer, loss=self.criterion, metrics=['mae'])

    def train_model(self, X_train_tensor, y_train_tensor, epochs=100):
        if self.model is None:
            raise ValueError("El modelo no ha sido inicializado.")
        self.model.fit(X_train_tensor, y_train_tensor, epochs=epochs, batch_size=32)

    def evaluate_model(self, X_test_tensor, y_test_tensor):
        if self.model is None:
            raise ValueError("El modelo no ha sido inicializado.")
        loss, mae = self.model.evaluate(X_test_tensor, y_test_tensor)
        print(f'MAE: {mae:.4f} - Loss: {loss:.4f}')

    def run(self):
        data = self.load_data()
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = self.preprocess_data(data)
        self.initialize_model()
        self.train_model(X_train_tensor, y_train_tensor)
        self.evaluate_model(X_test_tensor, y_test_tensor)

if __name__ == "__main__":
    predictive_model = PredictiveModel('/workspaces/trader_ai/data_diario.csv')
    predictive_model.run()

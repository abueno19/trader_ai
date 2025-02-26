import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

class PricePredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(PricePredictionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

class PredictiveModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.features = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal_Line']
        self.model = None
        self.criterion = nn.BCELoss()
        self.optimizer = None

    def load_data(self):
        data = pd.read_csv(self.data_path, parse_dates=['Datetime'])
        data['price_diff'] = data['close_price'].diff()
        data['target'] = (data['price_diff'] > 0).astype(int)
        data = data.dropna()
        return data

    def preprocess_data(self, data, sequence_length=10):
        X = data[self.features].values
        y = data['target'].values

        # Normalización manual
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_normalized = (X - X_mean) / X_std

        # Crear secuencias
        sequences = []
        targets = []
        for i in range(len(X_normalized) - sequence_length):
            sequences.append(X_normalized[i:i+sequence_length])
            targets.append(y[i+sequence_length])

        X_sequences = torch.tensor(sequences, dtype=torch.float32)
        y_sequences = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

        # División manual del conjunto de datos
        dataset_size = len(X_sequences)
        test_size = int(0.2 * dataset_size)
        train_size = dataset_size - test_size

        indices = torch.randperm(dataset_size)
        train_indices = indices[:train_size]
        test_indices = indices[test_size:]

        X_train = X_sequences[train_indices]
        X_test = X_sequences[test_indices]
        y_train = y_sequences[train_indices]
        y_test = y_sequences[test_indices]

        return X_train, X_test, y_train, y_test

    def initialize_model(self):
        input_dim = len(self.features)
        hidden_dim = 64
        num_layers = 2
        self.model = PricePredictionModel(input_dim, hidden_dim, num_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_model(self, X_train_tensor, y_train_tensor, epochs=100):
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()

            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def evaluate_model(self, X_test_tensor, y_test_tensor):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test_tensor)
            predicted_classes = (predictions > 0.5).float()
            accuracy = (predicted_classes.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
            print(f'Accuracy: {accuracy:.4f}')

    def run(self):
        data = self.load_data()
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = self.preprocess_data(data)
        self.initialize_model()
        self.train_model(X_train_tensor, y_train_tensor)
        self.evaluate_model(X_test_tensor, y_test_tensor)

if __name__ == "__main__":
    predictive_model = PredictiveModel('/workspaces/trader_ai/data_diario.csv')
    predictive_model.run()
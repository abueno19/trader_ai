import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class HistoricalDataFetcher:
    def __init__(self, symbol='EURUSD=X'):
        self.symbol = symbol

    def get_historical_data(self, start_date='2010-01-01', end_date=None, interval='1d'):
        """
        Obtiene datos históricos
        intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        """
        if end_date is None:
            end_date = datetime.now()

        ticker = yf.Ticker(self.symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)

        # Limpiar y preparar datos
        df = df.dropna()
        df = df.rename(columns={
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price',
            'Close': 'close_price',
            'Volume': 'volume'
        })

        return df

    def add_technical_indicators(self, df):
        """
        Añade indicadores técnicos al DataFrame
        """
        # Medias móviles
        df['SMA_20'] = df['close_price'].rolling(window=20).mean()
        df['SMA_50'] = df['close_price'].rolling(window=50).mean()
        df['SMA_200'] = df['close_price'].rolling(window=200).mean()

        # RSI
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close_price'].ewm(span=12, adjust=False).mean()
        exp2 = df['close_price'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        return df

    def prepare_training_data(self, window_size=30):
        """
        Prepara los datos para el entrenamiento
        """
        # Obtener datos históricos
        df = self.get_historical_data()
        df = self.add_technical_indicators(df)

        # Crear secuencias para entrenamiento
        sequences = []
        targets = []

        for i in range(len(df) - window_size):
            sequence = df.iloc[i:i+window_size]
            target = df.iloc[i+window_size]['close_price']
            sequences.append(sequence)
            targets.append(target)

        return sequences, targets

# Ejemplo de uso
def main():
    fetcher = HistoricalDataFetcher('EURUSD=X')

    # Obtener datos de diferentes períodos
    #data_diario = fetcher.get_historical_data(interval='1d')
    data_diario = fetcher.get_historical_data(
        start_date=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
        interval='1m'
    )

    # Añadir indicadores técnicos
    data_diario = fetcher.add_technical_indicators(data_diario)

    # Preparar datos para entrenamiento
    sequences, targets = fetcher.prepare_training_data()

    print("Forma de los datos diarios:", data_diario.shape)
    print("Número de secuencias de entrenamiento:", len(sequences))
    print("\nPrimeras filas de los datos procesados:")
    print(data_diario.head(-1))


    return data_diario, sequences, targets

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.callbacks import EarlyStopping
import os

class MLPModel:
    def __init__(self, file_path, target_column='PTS', look_back=10, train_size=0.8, epochs=50, batch_size=32, model_save_path='mlp_model.h5'):
        self.file_path = file_path
        self.target_column = target_column
        self.look_back = look_back
        self.train_size = train_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_save_path = model_save_path
        
        self.df = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = None
        self.train, self.test = None, None
        self.trainX, self.trainY = None, None
        self.testX, self.testY = None, None
        self.model = None
        self.trainPredict, self.testPredict = None, None
        self.trainY_orig, self.testY_orig = None, None

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        self.df = pd.read_csv(self.file_path)
        if self.target_column not in self.df.columns:
            raise ValueError(f"Column '{self.target_column}' not found in the dataset.")
        print("Data loaded successfully.")

    def preprocess_data(self):
        self.dataset = self.scaler.fit_transform(self.df[[self.target_column]].values)
        print("Data preprocessing completed.")

    def split_data(self):
        train_size = int(len(self.dataset) * self.train_size)
        self.train, self.test = self.dataset[:train_size], self.dataset[train_size:]
        print(f"Data split into train ({len(self.train)}) and test ({len(self.test)}) sets.")

    def create_sequences(self):
        def _create_sequences(data, look_back):
            X, Y = [], []
            for i in range(len(data) - look_back):
                X.append(data[i:i+look_back])
                Y.append(data[i+look_back])
            return np.array(X), np.array(Y)
        
        self.trainX, self.trainY = _create_sequences(self.train, self.look_back)
        self.testX, self.testY = _create_sequences(self.test, self.look_back)
        print("Sequences created for training and testing.")

    def build_model(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(self.look_back, 1)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        print("Model built and compiled.")

    def train_model(self):
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        history = self.model.fit(
            self.trainX, self.trainY,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            callbacks=[early_stop],
            validation_data=(self.testX, self.testY)
        )
        print("Model training completed.")
        self.plot_training_history(history)

    def plot_training_history(self, history):
        plt.figure(figsize=(10,6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def make_predictions(self):
        self.trainPredict = self.scaler.inverse_transform(self.model.predict(self.trainX))
        self.testPredict = self.scaler.inverse_transform(self.model.predict(self.testX))
        self.trainY_orig = self.scaler.inverse_transform(self.trainY)
        self.testY_orig = self.scaler.inverse_transform(self.testY)
        print("Predictions made on training and testing data.")

    def evaluate_model(self):
        rmse_train = np.sqrt(mean_squared_error(self.trainY_orig, self.trainPredict))
        rmse_test = np.sqrt(mean_squared_error(self.testY_orig, self.testPredict))
        mae = mean_absolute_error(self.testY_orig, self.testPredict)
        
        print(f'Train RMSE: {rmse_train:.2f}')
        print(f'Test RMSE: {rmse_test:.2f}')
        print(f'MAE: {mae:.2f}')
        
        return {'Train RMSE': rmse_train, 'Test RMSE': rmse_test, 'MAE': mae}

    def save_model(self):
        self.model.save(self.model_save_path)
        print(f"Model saved successfully at {self.model_save_path}.")

    def plot_predictions(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.testY_orig, label='Actual')
        plt.plot(self.testPredict, label='Predicted')
        plt.title('Actual vs Predicted PTS')
        plt.xlabel('Time')
        plt.ylabel('PTS')
        plt.legend()
        plt.show()

    def run_pipeline(self):
        self.load_data()
        self.preprocess_data()
        self.split_data()
        self.create_sequences()
        self.build_model()
        self.train_model()
        self.make_predictions()
        metrics = self.evaluate_model()
        self.save_model()
        self.plot_predictions()
        return metrics

if __name__ == "__main__":
    PLAYER = 'giannis-antetokounmpo'
    file_path = '/Users/mauro/Documents/MIA/IA_PROJECT2/ia_project_2/data/giannis-antetokounmpo.csv' # f'../data/{PLAYER}.csv' 
    mlp_model = MLPModel(
        file_path=file_path,
        target_column='PTS',
        look_back=5,
        train_size=0.80,
        epochs=100,
        batch_size=32,
        model_save_path=f'{PLAYER}-mlp.h5'
    )
    metrics = mlp_model.run_pipeline()

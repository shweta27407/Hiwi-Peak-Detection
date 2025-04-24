import numpy as np
import pandas as pd
import tensorflow

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam

class LSTMModel:
    def __init__(self):
        pass

    def normalize(self, X, y):
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaled_X = scaler_X.fit_transform(X)
        scaled_y = scaler_y.fit_transform(y)

        return scaler_X, scaler_y, scaled_X, scaled_y

    def denormalize(self, train_predict, y_train, test_predict, y_test, scaler_y):
        train_predict_inv = scaler_y.inverse_transform(train_predict)
        y_train_inv = scaler_y.inverse_transform(y_train)
        test_predict_inv = scaler_y.inverse_transform(test_predict)
        y_test_inv = scaler_y.inverse_transform(y_test)
        
        return train_predict_inv, y_train_inv, test_predict_inv, y_test_inv
    
    def split_and_scale(self, X, y, peak_indices, split):
        train_size = int(len(X) * split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))

        # Fit the scalers on the entire training data
        scaler_X.fit(X)
        scaler_y.fit(y)
        scaled_X_train = scaler_X.transform(X_train)
        scaled_X_test = scaler_X.transform(X_test)
        scaled_y_train = scaler_y.transform(y_train)
        scaled_y_test = scaler_y.transform(y_test)

        scaled_X_train_df = pd.DataFrame(scaled_X_train, index=X_train.index, columns=X_train.columns)
        scaled_X_test_df = pd.DataFrame(scaled_X_test, index=X_test.index, columns=X_test.columns)
        scaled_y_train_df = pd.DataFrame(scaled_y_train, index=y_train.index, columns=y_train.columns)
        scaled_y_test_df = pd.DataFrame(scaled_y_test, index=y_test.index, columns=y_test.columns)


        train_peak_indices = [i for i in peak_indices if i < len(X_train)]
        test_peak_indices = [i for i in peak_indices if i >= len(X_train)]

        X_train_peak = scaled_X_train_df.loc[train_peak_indices]
        y_train_peak = scaled_y_train_df.loc[train_peak_indices]
        X_train_non_peak = scaled_X_train_df.drop(train_peak_indices)
        y_train_non_peak = scaled_y_train_df.drop(train_peak_indices)
        X_test_peak = scaled_X_test_df.loc[test_peak_indices]
        y_test_peak = scaled_y_test_df.loc[test_peak_indices]
        X_test_non_peak = scaled_X_test_df.drop(test_peak_indices)
        y_test_non_peak = scaled_y_test_df.drop(test_peak_indices)
    
        return X_train_peak, y_train_peak, X_train_non_peak, y_train_non_peak, X_test_peak, y_test_peak, X_test_non_peak, y_test_non_peak, scaler_X, scaler_y

    def create_dataset(X, y, time_step=60):
        Xs, ys = [], []
        for i in range(len(X) - time_step):
            Xs.append(X[i:(i + time_step)])
            ys.append(y[i + time_step])
        return np.array(Xs), np.array(ys)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    
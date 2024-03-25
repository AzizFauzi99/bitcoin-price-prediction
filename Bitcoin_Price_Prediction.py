import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('dataset/BTC-Daily.csv')

# membalik df agar data terurut dari tanggal terlama ke terbaru
df = df.iloc[::-1]

# hapus kolom symbol, karena hanya ada 1 nilai
# hapus kolom unix, karena tidak digunakan
df = df.drop(columns=['symbol', 'unix'])

# ubah date menjadi datetime
df['date'] = pd.to_datetime(df['date'])

# cut data dari 2017 ke atas
df = df[df['date'] >= '2017-01-01']

# Split data menjadi training dan validation set
train_data, val_data = train_test_split(df, test_size=0.2, shuffle=False)

# MinMaxScaler
scaler = MinMaxScaler()
train_data = scaler.fit_transform(df[['close']])
val_data = scaler.transform(val_data[['close']])

# Fungsi untuk membuat dataset
def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 30

X_train, y_train = create_dataset(train_data, time_steps)
X_val, y_val = create_dataset(val_data, time_steps)

# Membuat model
model = tf.keras.models.Sequential([
  tf.keras.layers.LSTM(30, return_sequences=True),
  tf.keras.layers.LSTM(60),
  tf.keras.layers.Dense(30),
  tf.keras.layers.Dense(1),
])

optimizer = tf.keras.optimizers.Adam()
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))
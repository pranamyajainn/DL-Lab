# To develop a GRU-based RNN model for sentiment analysis on the IMDB movie reviews dataset.

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load offline IMDB data
with np.load("imdb.npz", allow_pickle=True) as data:
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

# Set maximum length and dynamically fix vocab size
max_len = 500
vocab_size = np.max([np.max(x) for x in np.concatenate((x_train, x_test))]) + 1

# Pad sequences
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Build GRU model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128),
    tf.keras.layers.GRU(128),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64)

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print("\nAccuracy:", accuracy)

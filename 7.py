import tensorflow as tf

# 1. Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocess: scale & reshape for CNN
x_train = x_train[..., None] / 255.0
x_test = x_test[..., None] / 255.0

# 3. Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)  # 10 digit classes
])

# 4. Compile & train
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

# 5. Test accuracy
model.evaluate(x_test, y_test)

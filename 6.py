#Q6: Implement in python SVM/Softmax classifier for CIFAR-10 dataset.


import tensorflow as tf

# 1. Load and normalize data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Build simple model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(10)  # no activation → logits
])

# 3. Compile with Softmax loss
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 4. Train
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

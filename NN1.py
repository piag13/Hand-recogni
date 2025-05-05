import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape, X_test.shape)
X_train = X_train.reshape(-1, 28*28).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28*28).astype("float32") / 255.0

#sequential API

model = keras.Sequential(
    [
        keras.Input(shape=(28*28,)),  # Corrected shape
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'), 
        layers.Dense(10),
    ]
)

model = keras.Sequential()
model.add(keras.Input(shape=(784,)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10))

#functional API

inputs = keras.Input(shape = (28*28,))
x = layers.Dense(512, activation='relu', name = "first_layer")(inputs)
x = layers.Dense(256, activation='relu', name = "second_layer")(x)
outputs = layers.Dense(10, activation = 'softmax')(x)
model = keras.Model(inputs = inputs, outputs = outputs)

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    metrics=["accuracy"]
)
model.fit(X_train, y_train, batch_size = 32, epochs = 5, verbose = 2)
model.evaluate(X_test, y_test, batch_size = 32, verbose = 2)


predictions = model.predict(X_test[19])  # Predict the first 5 test samples
predicted_classes = np.argmax(predictions, axis=1)  # Get the class with the highest probability

print("Predicted classes:", predicted_classes)
print("Actual classes:", y_test[19])

import matplotlib.pyplot as plt

# Show the 20th image in the training set
plt.imshow(X_train[19].reshape(28, 28), cmap="gray")  # Reshape back to 28x28 for visualization
plt.title(f"Label: {y_train[19]}")
plt.axis("off")
plt.show()
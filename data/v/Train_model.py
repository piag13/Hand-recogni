import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint

# Define dataset path
DATASET_PATH = 'data/process_combine_asl_dataset'

# Parameters
IMG_SIZE = 400
BATCH_SIZE = 32
EPOCHS = 20


# Function to load dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    actions = os.listdir(dataset_path)
    num_classes = len(actions)

    for action in actions:
        action_path = os.path.join(dataset_path, action)
        for image_name in os.listdir(action_path):
            image_path = os.path.join(action_path, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            images.append(image)
            labels.append(actions.index(action))

    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels, dtype='int')

    return images, labels

# Load dataset
X, y = load_dataset(DATASET_PATH)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display some training images
unique_labels = np.unique(y_train)
plt.figure(figsize=(10, 5))
for label in unique_labels:
    index = np.where(y_train == label)[0][0]
    plt.subplot(1, len(unique_labels), label + 1)
    plt.imshow(X_train[index])
    plt.title(f'Label: {label}')
    plt.axis('off')
plt.tight_layout()
plt.show()

#Focal loss
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        # Convert labels to one-hot encoding
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[1])
        
        # Clip the prediction value to prevent log(0) error
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma) * y_true
        
        return tf.reduce_mean(alpha * weight * cross_entropy)
    return focal_loss_fixed


# CNN Model
def cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(actions), activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate = 0.0001), loss=focal_loss(gamma=2., alpha=.25), metrics=['accuracy'])
    return model

callbacks = [
    ModelCheckpoint(
        filepath='palm_detector_cnn.h5',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
]

# Initialize and train the model
input_shape = (IMG_SIZE, IMG_SIZE, 3)
model = cnn_model(input_shape)

history = model.fit(X_train, y_train,epoch = EPOCHS, batch_size = BATCH_SIZE, validation_split = 0.2, callbacks=callbacks)

# Save model
model.save('palm_detector_cnn.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
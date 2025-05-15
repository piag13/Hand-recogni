import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.regularizers import l2

tf.get_logger().setLevel('ERROR')

# # Rest of your code remains the same
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define dataset path
DATA_PATH = os.path.join('data/process_combine_asl_dataset')

actions = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
}

# Parameters
BATCH_SIZE = 32
EPOCHS = 100

# Function to load dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    IMG_SIZE = 224  # Fixed size for all images

    # Check if dataset path exists
    actions = os.listdir(dataset_path)
    num_classes = len(actions)

    for action in actions:
        action_path = os.path.join(dataset_path, action)
        if not os.path.isdir(action_path):
            continue

        image_files = [f for f in os.listdir(action_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"Loading {len(image_files)} images for class '{action}'")

        for image_name in image_files:
            image_path = os.path.join(action_path, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                # Resize image to fixed dimensions
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                images.append(image)
                labels.append(actions.index(action))

    if not images:
        raise ValueError("No valid images found in the dataset")

    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels, dtype='int')

    print(f"Dataset loaded: {len(images)} images, {num_classes} classes")
    return images, labels

# Load dataset
X, y = load_dataset(DATA_PATH)

if len(X) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    raise ValueError("Dataset is empty")

# Display some training images
labels, counts = np.unique(y_train, return_counts=True)
labels_test, counts_test = np.unique(y_test, return_counts=True)

plt.figure(figsize=(10, 5))

# plot training data
plt.subplot(1, 2, 1)
plt.bar(labels, counts, color='blue', alpha=0.7)
plt.xticks(labels, actions, rotation=45)
plt.xlabel("Hành động")
plt.ylabel("Số mẫu")
plt.title("Số lượng mẫu trong tập Train")

# plot test data
plt.subplot(1, 2, 1)
plt.bar(labels_test, counts_test, color='red', alpha=0.7)
plt.xticks(labels_test, actions, rotation=45)
plt.xlabel("Hành động")
plt.ylabel("Số mẫu")
plt.title("Số lượng mẫu trong tập Test")

# CNN Model
def cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu',kernel_regularizer=l2(0.01),  input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D((2, 2)),
        Dropout(0.5),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(actions), activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

callbacks = [
    ModelCheckpoint(
        filepath='model.h5',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    ),
    EarlyStopping(monitor='val_loss', verbose=1, patience=5, restore_best_weights=True)

]
# Initialize and train the model
input_shape = (224, 224, 3)
model = cnn_model(input_shape)

print(model.summary())

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Save model
model.save('model.h5')
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
np.save('y_train.npy', y_train)

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.show()

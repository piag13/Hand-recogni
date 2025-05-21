import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Set seaborn style
sns.set_theme(style="whitegrid")

# Load the saved model and test data
model = tf.keras.models.load_model('model.h5')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate and print accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot some example predictions
def plot_example_predictions(X_test, y_test, y_pred_classes, num_examples=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_examples):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(X_test[i])
        plt.axis('off')
        plt.title(f'True: {y_test[i]}\nPred: {y_pred_classes[i]}')
    plt.tight_layout()
    plt.show()

# Show some example predictions
plot_example_predictions(X_test, y_test, y_pred_classes)

# Calculate per-class accuracy
per_class_accuracy = {}
for i in range(10):
    mask = y_test == i
    if np.any(mask):
        accuracy = np.mean(y_pred_classes[mask] == y_test[mask])
        per_class_accuracy[str(i)] = accuracy

# Plot per-class accuracy
plt.figure(figsize=(10, 6))
classes = list(per_class_accuracy.keys())
accuracies = list(per_class_accuracy.values())
sns.barplot(x=classes, y=accuracies)
plt.title('Per-Class Accuracy')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
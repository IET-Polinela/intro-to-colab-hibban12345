import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load dataset MNIST
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisasi data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Buat model CNN
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training model & simpan history
history = model.fit(x_train.reshape(-1,28,28,1), y_train, epochs=5, validation_data=(x_test.reshape(-1,28,28,1), y_test))

# Evaluasi model
test_loss, test_acc = model.evaluate(x_test.reshape(-1,28,28,1), y_test)
print(f"Test accuracy: {test_acc:.2f}")

# Simpan hasil evaluasi ke file teks
with open("evaluation.txt", "w") as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")

print("Evaluation results saved to evaluation.txt.")

# Visualisasi Loss & Akurasi
history_dict = history.history
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot Loss
ax[0].plot(history_dict['loss'], label='Train Loss', marker='o')
ax[0].plot(history_dict['val_loss'], label='Validation Loss', marker='o')
ax[0].set_title("Model Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()

# Plot Accuracy
ax[1].plot(history_dict['accuracy'], label='Train Accuracy', marker='o')
ax[1].plot(history_dict['val_accuracy'], label='Validation Accuracy', marker='o')
ax[1].set_title("Model Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()

# Simpan grafik ke dalam file
plt.savefig("training_performance.png")
plt.show()

print("Training performance plot saved to training_performance.png.")

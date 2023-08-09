import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

# Laden und aufteilen des MNIST-Datensatzes
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Daten vorverarbeiten
train_images = train_images / 255.0
test_images = test_images / 255.0

# Modell erstellen
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Modell kompilieren
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modell trainieren
model.fit(train_images, train_labels, epochs=5)

# Modell evaluieren
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Genauigkeit:', test_acc)

# Vorhersagen machen
predictions = model.predict(test_images)

# Einige Vorhersagen visualisieren
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = predictions[i].argmax()
    true_label = test_labels[i]
    plt.xlabel(f"Pred: {predicted_label}, True: {true_label}")
plt.show()

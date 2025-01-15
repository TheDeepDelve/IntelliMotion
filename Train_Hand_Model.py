import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
from tensorflow.python.data import Dataset
import matplotlib.pyplot as plt

#Rubrics - DATASET LOAD AND PREPARATION
dataset_path = "D:\Projects\Deep Learning Project\ComputerVisionProject\Hand_Dataset"
folders = ["1", "2", "3","4","5","6","7","8","9","10"]

IMG_HEIGHT = 64
IMG_WIDTH = 64

def load_data():
    data = []
    labels = []                               
    for folder in folders:
        path = os.path.join(dataset_path, folder)
        class_num = folders.index(folder)
        for img in os.listdir(path):
            try:
                img_array = tf.keras.preprocessing.image.load_img(os.path.join(path, img), target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_array = tf.keras.preprocessing.image.img_to_array(img_array) / 255.0
                data.append(img_array)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img}: {e}")
    return data, labels

data, labels = load_data()
data = tf.convert_to_tensor(data)
labels = tf.convert_to_tensor(labels)

dataset = Dataset.from_tensor_slices((data, labels))

# split sizes calc..
total_samples = len(data)
val_size = int(0.2 * total_samples)
train_size = total_samples - val_size

# shuffle nd split dataset
dataset = dataset.shuffle(buffer_size=total_samples)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

batch_size = 32
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

# Rubrics - MODEL DESIGN AND ARCHITECTURE
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(folders), activation='softmax')
])

#Rubrics - EXECUTION AND IMPLEMENTATION
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model train
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

model.save("hand_landmark_model.h5")
print("Model training complete and saved as 'hand_landmark_model.h5'")

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_acc) + 1)

#Rubrics - RESULT AND ANALYSIS (VISUALISATIONS)
# plotting training nd valid. acc.
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Training Accuracy', marker='o')
plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
plt.title('Hand Model - Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# plotting training nd valid. loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Training Loss', marker='o', color='orange')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o', color='red')
plt.title('Hand Model - Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

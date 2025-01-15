import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#Rubrics - DATASET LOAD AND PREPARATION
train_dir = "D:\Projects\Deep Learning Project\ComputerVisionProject\Pose_Dataset\Train Data"
val_dir = "D:\Projects\Deep Learning Project\ComputerVisionProject\Pose_Dataset\Validation Data"

img_size = (64, 64)
batch_size = 32
epochs = 5

#data augmentation nd preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

#Rubrics - MODEL DESIGN AND ARCHITECTURE
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(52, activation="relu"),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation="softmax")
])

#Rubrics - EXECUTION AND IMPLEMENTATION
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#model train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

model.save("pose_model.h5")
print("Model saved as pose_model.h5")

#Rubrics - RESULT AND ANALYSIS (VISUALISATIONS)
# plotting training nd valid. acc.
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Pose Model - Accuracy")

# plotting training nd valid. loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Pose Model - Loss")
plt.show()

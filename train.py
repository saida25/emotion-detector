import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset paths
train_dir = "data/train"
val_dir = "data/test"

# Data preprocessing
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_dir, target_size=(48,48), color_mode="grayscale",
    batch_size=64, class_mode="categorical"
)

val_generator = datagen.flow_from_directory(
    val_dir, target_size=(48,48), color_mode="grayscale",
    batch_size=64, class_mode="categorical"
)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")  # 7 emotions
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_generator, epochs=25, validation_data=val_generator)

# Save model
model.save("models/emotion_model.h5")


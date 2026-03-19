import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.callbacks import ModelCheckpoint

IMG_SIZE = 224
BATCH_SIZE = 32

# Paths
BASE_PATH = "dataset"
folders = ["color", "grayscale", "segmented"]

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generators = []
val_generators = []

# Load all folders
for folder in folders:
    path = os.path.join(BASE_PATH, folder)
    
    train_gen = datagen.flow_from_directory(
        path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    val_gen = datagen.flow_from_directory(
        path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    train_generators.append(train_gen)
    val_generators.append(val_gen)

# Combine generators
train_data = tf.data.Dataset.from_generator(
    lambda: (item for gen in train_generators for item in gen),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, train_generators[0].num_classes), dtype=tf.float32)
    )
)

val_data = tf.data.Dataset.from_generator(
    lambda: (item for gen in val_generators for item in gen),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, val_generators[0].num_classes), dtype=tf.float32)
    )
)

import json

class_labels = train_data.class_indices

import os
os.makedirs("model", exist_ok=True)

with open("model/class_labels.json", "w") as f:
    json.dump(class_labels, f)
# Model
import tensorflow as tf

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False
# 👇 Add this BEFORE model
NUM_CLASSES = len(train_generators[0].class_indices)

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
checkpoint = ModelCheckpoint(
    "model/crop_disease_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
# Train
model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[early_stop, checkpoint]
)

# Save
os.makedirs("model", exist_ok=True)
model.save("model/crop_disease_model.h5")

print("✅ Model trained using ALL datasets!")
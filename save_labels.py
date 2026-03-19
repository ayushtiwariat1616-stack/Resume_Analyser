from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

# Use ANY ONE folder (color recommended)
DATASET_PATH = "dataset/grayscale"

datagen = ImageDataGenerator()

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

class_labels = train_data.class_indices

os.makedirs("model", exist_ok=True)

with open("model/class_labels.json", "w") as f:
    json.dump(class_labels, f)

print("✅ class_labels.json saved correctly!")
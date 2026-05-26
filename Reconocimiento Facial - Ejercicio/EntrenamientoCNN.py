import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Config
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

# 1. Load Data with Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(r'C:\Users\MSI\Documents\Universidad\IA\Reconocimiento Facial - Ejercicio\data\train', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse')
val_gen = val_datagen.flow_from_directory(r'C:\Users\MSI\Documents\Universidad\IA\Reconocimiento Facial - Ejercicio\data\val', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse')

# 2. Build Model (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Train and Save
model.fit(train_gen, validation_data=val_gen, epochs=25)
model.save(r'C:\Users\MSI\Documents\Universidad\IA\Reconocimiento Facial - Ejercicio\celebrity_model.h5')

# Store class names for prediction
class_names = list(train_gen.class_indices.keys())
print("Classes identified:", class_names)
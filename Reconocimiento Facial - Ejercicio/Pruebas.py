import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

def predict_celebrity(img_path, class_names, model_path=r'C:C:\Users\MSI\Documents\Universidad\IAl\Reconocimiento Facial - Ejercicio\celebrity_model.h5'):
    # Load model and image
    model = tf.keras.models.load_model(model_path)
    img = image.load_img(img_path, target_size=(160, 160))
    
    # Pre-process
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Create batch axis

    # Predict
    predictions = model.predict(img_array)
    score = np.max(predictions)
    class_idx = np.argmax(predictions)
    
    # Display
    plt.imshow(img)
    plt.title(f"Prediction: {class_names[class_idx]} ({100 * score:.2f}%)")
    plt.axis('off')
    plt.show()

# Usage:
mis_clases = ['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling']
foto_prueba = r"C:\Users\MSI\Documents\Universidad\IA\Reconocimiento Facial - Ejercicio\data\val\ben_afflek\1.jpg"

predict_celebrity(foto_prueba, mis_clases)
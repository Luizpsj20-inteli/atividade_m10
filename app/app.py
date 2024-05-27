from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Carregar o modelo salvo sem compilação
model = load_model('cifar10_model_no_compile.h5')

# Compilar o modelo após carregar
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print("Model compiled successfully")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ler e converter a imagem para RGB
        image = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB').resize((32, 32))
        print(f"Image mode after conversion: {image.mode}")
        
        # Transformar a imagem em um array numpy
        image = np.array(image)
        print(f"Image shape after conversion: {image.shape}")
        
        # Expandir as dimensões da imagem para corresponder ao formato esperado pelo modelo
        image = np.expand_dims(image / 255.0, axis=0)
        print(f"Image shape after expand_dims: {image.shape}")
        
        # Fazer a predição
        prediction = model.predict(image)
        class_id = np.argmax(prediction)
        print(f"Predicted class_id (before conversion): {class_id} of type {type(class_id)}")
        class_id = int(class_id)  # Convertendo class_id para int
        print(f"Predicted class_id (after conversion): {class_id} of type {type(class_id)}")
        return jsonify({'class_id': class_id})
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

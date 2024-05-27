# Image Classification API

## Descrição
Este projeto implementa um modelo de classificação de imagens utilizando o dataset CIFAR-10 e uma API Flask que recebe uma imagem e retorna a categoria inferida pelo modelo. O objetivo é criar um serviço de inferência que possa ser acessado via HTTP para classificar imagens conforme o tutorial fornecido.

## Implementação do Modelo
O modelo foi treinado usando o dataset CIFAR-10, que consiste em 60.000 imagens coloridas de 32x32 pixels em 10 classes, com 6.000 imagens por classe. O conjunto de dados foi dividido em 50.000 imagens de treinamento e 10.000 imagens de teste.

### Código para Treinamento do Modelo
```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Carregar e preparar o dataset CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Construir o modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Salvar o modelo
model.save('app/model/cifar10_model_no_compile.h5')
```

### Resultados do Treinamento
O modelo foi treinado por 10 épocas com os seguintes resultados de perda (loss) e acurácia (accuracy):

```plaintext
Epoch 1/10
1563/1563 [==============================] - 67s 42ms/step - loss: 1.5196 - accuracy: 0.4478 - val_loss: 1.2927 - val_accuracy: 0.5417
Epoch 2/10
1563/1563 [==============================] - 65s 42ms/step - loss: 1.1376 - accuracy: 0.5979 - val_loss: 1.0680 - val_accuracy: 0.6253
Epoch 3/10
1563/1563 [==============================] - 64s 41ms/step - loss: 0.9829 - accuracy: 0.6578 - val_loss: 0.9862 - val_accuracy: 0.6561
Epoch 4/10
1563/1563 [==============================] - 63s 41ms/step - loss: 0.8787 - accuracy: 0.6923 - val_loss: 0.9032 - val_accuracy: 0.6876
Epoch 5/10
1563/1563 [==============================] - 63s 40ms/step - loss: 0.8079 - accuracy: 0.7182 - val_loss: 0.8789 - val_accuracy: 0.6979
Epoch 6/10
1563/1563 [==============================] - 62s 40ms/step - loss: 0.7587 - accuracy: 0.7353 - val_loss: 0.8938 - val_accuracy: 0.6955
Epoch 7/10
1563/1563 [==============================] - 61s 39ms/step - loss: 0.7087 - accuracy: 0.7523 - val_loss: 0.8854 - val_accuracy: 0.6969
Epoch 8/10
1563/1563 [==============================] - 61s 39ms/step - loss: 0.6667 - accuracy: 0.7664 - val_loss: 0.8865 - val_accuracy: 0.6991
Epoch 9/10
1563/1563 [==============================] - 61s 39ms/step - loss: 0.6256 - accuracy: 0.7802 - val_loss: 0.8929 - val_accuracy: 0.7078
Epoch 10/10
1563/1563 [==============================] - 62s 40ms/step - loss: 0.5879 - accuracy: 0.7922 - val_loss: 0.8703 - val_accuracy: 0.7135
```

## Implementação da API Flask
A API Flask foi implementada para receber uma imagem via POST, processá-la e retornar a categoria inferida pelo modelo.

### Código da API
```python
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Carregar o modelo salvo sem compilação
model = load_model('model/cifar10_model_no_compile.h5')

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
```

## Testando a API
Três imagens de teste foram utilizadas para verificar a funcionalidade da API. As imagens foram salvas a partir do conjunto de dados CIFAR-10.

### Código para Salvar Imagens de Teste
```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Carregar o dataset CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Função para salvar uma imagem
def save_image(image, index):
    plt.imsave(f'test/test_image_{index}.png', image)
    print(f"Imagem salva como 'test/test_image_{index}.png'")

# Salvar 3 imagens de teste
for i in range(3):
    save_image(test_images[i], i)
```

### Código para Testar a API
```python
import requests

url = 'http://127.0.0.1:5000/predict'

# Lista de arquivos de imagens de teste
test_images = [
    'test/test_image_0.png',
    'test/test_image_1.png',
    'test/test_image_2.png'
]

for image_path in test_images:
    with open(image_path, 'rb') as img:
        files = {'image': img}
        response = requests.post(url, files=files)
        
        try:
            response_data = response.json()
            print(f"Response for {image_path}: {response_data}")
        except requests.exceptions.JSONDecodeError:
            print(f"Failed to decode JSON response for {image_path}. Raw response:")
            print(response.text)
```

## Resultados dos Testes
Os testes foram realizados com três imagens diferentes, e os resultados foram os seguintes:

### Teste 1
- **Imagem**: `test_image_0.png`
- **Resultado**: `{'class_id': 3}`

### Teste 2
- **Imagem**:

 `test_image_1.png`
- **Resultado**: `{'class_id': 8}`

### Teste 3
- **Imagem**: `test_image_2.png`
- **Resultado**: `{'class_id': 9}`

## Conclusão
A implementação foi bem-sucedida, com o modelo sendo capaz de classificar corretamente as imagens de teste através da API Flask. A API está funcionando conforme o esperado, retornando as categorias previstas em formato JSON.

## Contribuições
Código desenvolvido por Luiz Carlos da Silva - Ciência da Computação - Inteli


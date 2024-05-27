import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Carregar o dataset CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Função para salvar uma imagem
def save_image(image, index):
    plt.imsave(f'C:/Users/Inteli\Documents/CC_inteli/ponderada/atividade_m10/app/save_image{index}.png', image)
    print(f"Imagem salva como 'app/test/test_image_{index}.png'")

# Salvar 3 imagens de teste
for i in range(3):
    save_image(test_images[i], i)

import requests

url = 'http://127.0.0.1:5000/predict'

# Lista de arquivos de imagens de teste
test_images = [
    'save_image0.png',
    'save_image1.png',
    'save_image2.png'
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

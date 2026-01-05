from flask import Flask, render_template, jsonify
import torch
from torchvision import datasets, transforms
import random
import base64
from io import BytesIO
from matplotlib import pyplot as plt

# On réutilise ta classe (obligatoire pour charger le modèle)
from main import NeuronalNetwork  # Remplace par le nom de ton fichier ou copie la classe ici

app = Flask(__name__)

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ["T-shirt/top", "Pantalon", "Pullover", "Robe", "Blouson", "Sandale", "Chemise", "Basket", "Sac à main",
           "Bottine"]

# Chargement du modèle
model = NeuronalNetwork().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Chargement des données de test
test_data = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())


@app.route('/')
def index():
    return render_template('index.html')  # Affiche la page web


@app.route('/prediction-vetement')
def test_random():
    # 1. Choisir image au hasard
    idx = random.randint(0, len(test_data) - 1)
    img_tensor, label = test_data[idx]

    # 2. Prédiction
    with torch.no_grad():
        output = model(img_tensor.to(device).unsqueeze(0))
        prediction = classes[output[0].argmax(0)]
        actual = classes[label]

    # 3. Convertir l'image en format Web (Base64) pour l'afficher en HTML
    plt.figure(figsize=(2, 2))
    plt.imshow(img_tensor.squeeze(), cmap="gray")
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # 4. Envoyer les résultats au format JSON
    return jsonify({
        "prediction": prediction,
        "actual": actual,
        "image": img_base64,
        "success": (prediction == actual)
    })


if __name__ == '__main__':
    app.run(debug=True)
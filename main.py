import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# ==========================================
# 1. PRÉPARATION DES DONNÉES
# ==========================================

# Téléchargement du jeu de données d'entraînement
training_data = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)

# Téléchargement du jeu de données de test
test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Création des DataLoaders : ils gèrent le passage des données par "lots" (batches)
# Cela évite de saturer la mémoire vive du PC.
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# ==========================================
# 2. CONFIGURATION DU MATÉRIEL (CPU vs GPU)
# ==========================================

# On vérifie si une carte graphique (CUDA ou MPS) est disponible, sinon on utilise le processeur (CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation du matériel : {device}")


# ==========================================
# 3. DÉFINITION DE L'ARCHITECTURE DU MODÈLE
# ==========================================

class NeuronalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # On aplatit l'image 28x28 en un vecteur de 784 pixels
        self.flatten = nn.Flatten()
        # Définition des couches du réseau (Linaire -> Activation ReLU -> ...)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),  # 10 sorties car il y a 10 types de vêtements
        )

    # La fonction forward définit comment les données passent dans le réseau
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# On crée l'instance du modèle et on l'envoie sur le processeur choisi (CPU ou GPU)
model = NeuronalNetwork().to(device)
# On charge dernière sauvegarde du model pour améliorer de + en + sa précision
model.load_state_dict(torch.load("model.pth", weights_only=True))

# ==========================================
# 4. FONCTIONS D'ENTRAÎNEMENT ET DE TEST
# ==========================================

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()  # On passe le modèle en mode entraînement
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 1. Calcul de l'erreur (prediction vs réalité)
        pred = model(X)
        loss = loss_fn(pred, y)

        # 2. Backpropagation (le coeur de l'apprentissage)
        optimizer.zero_grad()  # On remet les gradients à zéro
        loss.backward()  # On calcule les erreurs pour chaque neurone
        optimizer.step()  # On ajuste les poids du réseau

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Perte : {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # On passe le modèle en mode évaluation (désactive le Dropout, etc.)
    test_loss, correct = 0, 0

    # On désactive le calcul des gradients pour aller plus vite (inutile en test)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # On compte le nombre de bonnes réponses
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Résultats du test : \n Précision: {(100 * correct):>0.1f}%, Perte moyenne: {test_loss:>8f} \n")


# ==========================================
# 5. LANCEMENT DE L'APPRENTISSAGE
# ==========================================

# On définit la fonction de perte (CrossEntropy est idéal pour la classification)
# et l'optimiseur (SGD = Descente de gradient stochastique)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# La boucle finale qui fait tourner les époques
epoques = 5
for t in range(epoques):
    print(f"Époque {t + 1}\n---------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Entraînement terminé !")

# ==========================================
# 6. SAUVEGRADER LE MODELE
# ==========================================
torch.save(model.state_dict(), "model.pth")
print("Le modele PyTorch a été sauvegarder dans le fichier model.pth")

# ==========================================
# 7. LE MODELE ENTRAINER TOUVE LE VETEMENT
# ==========================================
#Tableau de vetement disponible = 10 (catégories)
classes = [
    "T-shirt/top",
    "Pentalon",
    "Pullover",
    "Robe",
    "Blouson",
    "Sandale",
    "Chemise",
    "Basquette",
    "Sac à main",
    "Chaussure femme"
]

model.eval()
x,y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actuel = classes[pred[0].argmax(0)], classes[y]
    print(f"Dernière prédiction : {predicted}, actuel {actuel}")
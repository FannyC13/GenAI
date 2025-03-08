import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from datasets import load_dataset
import matplotlib.pyplot as plt

# Charger le modèle et le processeur fine-tunés
model = ViTForImageClassification.from_pretrained("vit-cifar10-finetuned")
processor = ViTFeatureExtractor.from_pretrained("vit-cifar10-finetuned")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Charger le dataset CIFAR-10
raw_ds = load_dataset("cifar10", split="test")

# Classes CIFAR-10
class_names = ['plane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Fonction de prédiction

def predict_image(image):
    inputs = processor(images=image.resize((224, 224)), return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx

# Visualisation des 10 premières images avec prédictions
plt.figure(figsize=(15, 6))

for i in range(10):
    image = raw_ds[i]["img"]
    true_label = raw_ds[i]["label"]
    predicted_class = predict_image(image=image)

    plt.subplot(2, 5, i+1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Pred: {predicted_class} ({class_names[predicted_class]})\nVrai: {raw_ds[i]['label']} ({class_names[raw_ds[i]['label']]})", fontsize=10)

plt.tight_layout()
plt.savefig("cifar_test.png")

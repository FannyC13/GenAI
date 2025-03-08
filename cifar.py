import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import (ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments, DefaultDataCollator)
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image

# Chargement de cifar
dataset = load_dataset("cifar10")

# Charger le processor ViTFeatureExtractor
processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

def preprocess(batch):
    """
    Transforme les images PIL en tenseurs normalisés et ajoute la clé 'pixel_values'.
    """
    images = [img.convert("RGB") for img in batch["img"]]
    features = processor(images, return_tensors="np")
    batch["pixel_values"] = features["pixel_values"]
    return batch

# Preprocess
dataset["train"] = dataset["train"].map(preprocess, batched=True, remove_columns=["img"])
dataset["test"] = dataset["test"].map(preprocess, batched=True, remove_columns=["img"])

# ViTForImageClassification + adaptation pour CIFAR-10 (10 classes)
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",num_labels=10,ignore_mismatched_sizes=True)
model.gradient_checkpointing_enable()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",         # Évaluer à chaque époque
    save_strategy="epoch",               # Sauvegarder à chaque époque
    learning_rate=2e-5,                  # Learning rate
    per_device_train_batch_size=32,      # Batch Training
    per_device_eval_batch_size=32,       # Betch Eval
    num_train_epochs=5,                  # Epoch
    weight_decay=0.01,                   # Weight Decay
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,                          
    dataloader_num_workers=2,           
    remove_unused_columns=True,
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=DefaultDataCollator(),
    tokenizer=processor, 
    compute_metrics=compute_metrics
)

# Entraînement
trainer.train()

# Sauvegarde du modèle et du processor
model.save_pretrained("vit-cifar10-finetuned")
processor.save_pretrained("vit-cifar10-finetuned")

def predict_image(image):
    """
    Prédit la classe d'une image en utilisant le modèle fine-tuné.
    """
    image = image.resize((224, 224))
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx

# Test
raw_ds = load_dataset("cifar10")["test"]
sample_image = raw_ds[0]["img"]
predicted_class = predict_image(sample_image)
print(f"Classe prédite pour la première image de test CIFAR-10 : {predicted_class}")

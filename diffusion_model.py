import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Chargement des données d'entraînement et de test
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalisation des images
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# On définit une fonction pour ajouter du bruit gaussien aux images à différents niveaux
def add_gaussian_noise(images, noise_factor=0.5):
    """
    Ajoute du bruit gaussien aux images et s'assure que les valeurs restent dans [0, 1].
    
    Paramètres :
      - images      : ensemble d'images (numpy array)
      - noise_factor: facteur de bruit (float)
      
    Sortie :
      - images bruitées
    """
    noise = np.random.normal(0, noise_factor, images.shape)
    return np.clip(images + noise, 0, 1)

# Préparation des images d'entraînement et de test
train_images = X_train.astype("float32")
test_images = X_test.astype("float32")

# Ajout de bruit aux images
train_noisy = add_gaussian_noise(train_images)
test_noisy = add_gaussian_noise(test_images)

# 2.2. Modèle U-Net
def build_unet(input_shape=(32, 32, 3)):
    """
    Construit un U-Net plus complexe avec deux blocs d'encodeur et deux blocs de décodeur.
    
    Architecture :
      - Encoder Block 1 : deux convolutions puis max pooling.
      - Encoder Block 2 : deux convolutions avec un nombre de filtres augmenté puis max pooling.
      - Bottleneck : deux convolutions.
      - Decoder Block 1 : convolution transposée, concaténation avec le bloc d'encodeur correspondant, deux convolutions.
      - Decoder Block 2 : idem.
      - Couche de sortie : convolution 1x1 pour obtenir 3 canaux.
    """
    inputs = keras.Input(shape=input_shape)
    
    # Encoder Block 1
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    # Encoder Block 2
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    
    # Decoder Block 1
    u1 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    
    # Decoder Block 2
    u2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c5)
    
    # Couche de sortie avec sigmoid activation
    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid', padding='same')(c5)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Modèle U-Net
model = build_unet()
model.compile(optimizer="adam", loss="mse")

# Entraînement
history = model.fit(train_noisy[:5000], train_images[:5000],epochs=100, batch_size=32,validation_data=(test_noisy[:1000], test_images[:1000]))

# Prédictions
predictions = model.predict(test_noisy[:10])

# Visualisation des résultats pour le U-Net
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
for i in range(10):
    axes[0, i].imshow(test_noisy[i])
    axes[0, i].axis("off")
    axes[1, i].imshow(predictions[i])
    axes[1, i].axis("off")
    axes[2, i].imshow(test_images[i])
    axes[2, i].axis("off")

axes[0, 0].set_ylabel("Bruité", fontsize=10)
axes[1, 0].set_ylabel("Débruité", fontsize=10)
axes[2, 0].set_ylabel("Original", fontsize=10)

plt.savefig("unet_results.png")
plt.close()

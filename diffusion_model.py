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

# Définition du planning de bruitage (noise schedule)
num_timesteps = 1000 
beta_start = 0.0001
beta_end = 0.02
beta = np.linspace(beta_start, beta_end, num_timesteps)

def add_noise(image, t):
    """
    Ajoute du bruit à une image donnée en fonction de l'étape t.
    Paramètres :
      - image : image originale (numpy array)
      - t     : étape de bruitage (un entier entre 0 et num_timesteps-1)
      
    Sortie :
      - noisy_image : image bruitée
    """
    # Générer un bruit aléatoire de même forme que l'image
    noise = np.random.randn(*image.shape)
    # On combine l'image et le bruit en utilisant beta[t] pour définir l'intensité du bruit
    noisy_image = np.sqrt(1 - beta[t]) * image + np.sqrt(beta[t]) * noise
    return noisy_image

# Sélection aléatoire d'une image dans le jeu d'entraînement
sample_image = X_train[np.random.randint(0, X_train.shape[0])]
t = int(num_timesteps - 1)
noisy_sample = add_noise(sample_image, t)

# Visualisation de l'image originale et de plusieurs versions bruitées à différents niveaux
plt.figure(figsize=(8, 4))
plt.subplot(3, 5, 1)
plt.imshow(sample_image)
plt.title("Image originale")
plt.axis('off')

# Affichage d'images bruitées avec différents niveaux (en variant t)
plt.subplot(3, 5, 2)
plt.imshow(add_noise(sample_image, t // 5))
plt.title(f"Image bruitée (t={t // 5})")
plt.axis('off')

plt.subplot(3, 5, 3)
plt.imshow(add_noise(sample_image, t // 2))
plt.title(f"Image bruitée (t={t // 2})")
plt.axis('off')

plt.subplot(3, 5, 4)
plt.imshow(add_noise(sample_image, t // 2 + t // 3))
plt.title(f"Image bruitée (t={t // 2 + t // 3})")
plt.axis('off')

plt.subplot(3, 5, 5)
plt.imshow(noisy_sample)
plt.title(f"Image bruitée (t={t})")
plt.axis('off')

plt.savefig("diffusion_demo.png")
plt.close()

# On définit une fonction pour ajouter du bruit gaussien aux images
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

# 2.1. Modèle U-Net avec 1 seule couche

def build_simple_unet(input_shape=(32, 32, 3)):
    """
    Construit un U-Net
    
    Architecture :
      - Encoder : une couche convolutionnelle suivie d'un max pooling.
      - Decoder : une couche de convolution transposée pour upsampling.
      - Couche de sortie : convolution avec activation sigmoïde pour restituer 3 canaux.
    """
    inputs = keras.Input(shape=input_shape)
    # Encoder : Convolution + MaxPooling
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    # Decoder : Upsampling avec Conv2DTranspose
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    outputs = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)
    model = keras.Model(inputs, outputs)
    return model

# Modèle U-Net 
model = build_simple_unet()
model.compile(optimizer="adam", loss="mse")

# Entraînement sur les données
history = model.fit(train_noisy[:5000], train_images[:5000],
                    epochs=50, batch_size=32,
                    validation_data=(test_noisy[:1000], test_images[:1000]))

# Prédictions
predictions = model.predict(test_noisy[:10])

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
plt.savefig("simple_unet_results.png")
plt.close()


# 2.2. Modèle U-Net complexe
def build_complex_unet(input_shape=(32, 32, 3)):
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

# Modèle U-Net complexe
model = build_complex_unet()
model.compile(optimizer="adam", loss="mse")

# Entraînement
history = model.fit(train_noisy[:5000], train_images[:5000],
                    epochs=50, batch_size=32,
                    validation_data=(test_noisy[:1000], test_images[:1000]))

# Prédictions
predictions = model.predict(test_noisy[:10])

# Visualisation des résultats pour le U-Net complexe
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

plt.savefig("complex_unet_results.png")
plt.close()

from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from vae.decoder import Decoder

# Charger le décodeur VAE comme générateur
latent_dim = 8  # La dimension doit correspondre à celle utilisé dans le VAE
generator = Decoder(latent_dim).get_model()
generator.load_weights("vae_decoder.weights.h5")  # Charger les poids du décodeur VAE

# Échantillonner des images générées par le générateur
def generate_images(generator, n_images=5):
    """
    Génère et affiche des images à partir de points aléatoires de l'espace latent.
    """
    z_samples = np.random.normal(0, 1, size=(n_images, latent_dim))  # Points aléatoires à partir d'une distribution normale
    generated_images = generator.predict(z_samples)  # Générer des images grâce au générateur
    plt.figure(figsize=(20, 4))
    for i in range(n_images):
        plt.subplot(1, n_images, i + 1)
        plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()

generate_images(generator)

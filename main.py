import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from vae.encoder import Encoder
from vae.decoder import Decoder
from vae.utils import plot_images, save_models
import tensorflow as tf

def load_data():
    """
    Charge les données MNIST et les normalise entre 0 et 1.
    - Les réseaux neuronaux fonctionnent mieux avec des données normalisées.
    - MNIST contient des images 28x28 de chiffres manuscrits (0-9).

    Outputs :
    - x_train : Données d'entraînement normalisées.
    - x_test : Données de test normalisées.
    """
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.  # Normalisation entre 0 et 1
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # Ajout d'une dimension pour le canal (gris)
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    return x_train, x_test

def main(input_shape, latent_dim, epochs, batch_size):
    """
    Entraîne le VAE avec les données MNIST et reconstruit les images.

    Paramètres :
    - input_shape : Dimensions des données d'entrée (28, 28, 1).
    - latent_dim : Taille de l'espace latent.
    - epochs : Nombre d'itérations d'entraînement sur l'ensemble des données.
    - batch_size : Taille des lots pour l'entraînement.
    """
    x_train, x_test = load_data()

    # Initialisation des modèles encodeur et décodeur
    encoder = Encoder(input_shape, latent_dim).get_model()
    decoder = Decoder(latent_dim).get_model()

    optimizer = Adam(learning_rate=0.001)  # Optimiseur Adam
    beta = 0.1  # Coefficient de la KL Divergence

    @tf.function
    def train_step(x):
        """
        Effectue une étape d'entraînement avec GradientTape.

        Paramètre :
        - x : Batch d'entrée.

        Outputs :
        - loss : Perte totale pour le lot.

        Le GradientTape est une méthode permettant d'enregistrer toutes les opérations mathématiques faites pendant le passage des données (encodeur, reparamétrisation, décodeur, loss). Il
        enregistre automatiquement comment chaque paramètre du modèle influence la loss. Une fois la loss calculé, il utilise ces informations pour calculer les gradients et mettre à jour les poids.
        Il offre un meilleur controle sur le calcul des loss et gradients.
        """
        with tf.GradientTape() as tape:
            # Passe les données dans l'encodeur pour obtenir les sorties latentes (distribution latente)
            z_mean, z_log_var, z = encoder(x)
            x_decoded = decoder(z)  # Génère les images reconstruites

            # Calcul de la Reconstruction Loss
            # Différence entre les images d'origine et les images reconstruites
            reconstruction_loss = binary_crossentropy(tf.keras.backend.flatten(x),  tf.keras.backend.flatten(x_decoded) ) * 28 * 28 

            # Calcul de la KL Divergence
            # axis=-1 : Car on somme sur la dernière dimension, ici les dimensions latentes (vecteur z).
            kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),axis=-1 )

            # Combine Reconstruction Loss et KL Divergence
            loss = tf.reduce_mean(reconstruction_loss + beta * kl_loss)

        # Calcul des gradients et mise à jour des poids
        gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
        return loss

    """Il permet d'entrainer le modèle sur plusieurs epoch, il mélange les données d'entrainement à chaque ecpoch pour éviter que le modèle apprenne des pattern lié à l'ordre. On divise en batch pour avoir
    un entrainement plus efficace en terme de mémoire. Pour chaque batch on calcule la loss pour mettre à jour les poids"""
    for epoch in range(epochs):
        np.random.shuffle(x_train)
        for i in range(0, len(x_train), batch_size):
            batch = x_train[i:i + batch_size]
            loss = train_step(batch)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy():.4f}")

    # Reconstruction des images pour visualisation
    z_mean, z_log_var, z = encoder.predict(x_test)  # Obtenir les vecteurs latents
    decoded_imgs = decoder.predict(z)  # Reconstruire les images à partir des vecteurs latents
    plot_images(x_test, decoded_imgs)  # Afficher les images originales et reconstruites

    # Sauvegarde des modèles
    save_models(encoder, decoder)

if __name__ == "__main__":
    main(input_shape=(28, 28, 1), latent_dim=8, epochs=30, batch_size=64)

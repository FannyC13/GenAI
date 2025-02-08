import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def train_gan(generator, discriminator, gan, x_train, epochs=100, batch_size=128, latent_dim=100):  # Augmentation du nombre d'époques
    '''
    Entraînement du GAN :
    - À chaque epoch, nous générons des images fausses et nous les opposons aux vraies images.
    - Le discriminateur est entraîné à reconnaître les vraies images et détecter les fausses.
    - Le générateur est entraîné à produire des images qui trompent le discriminateur.
    '''
    for epoch in range(epochs):
        for _ in range(batch_size):
             
            # Entraînement du discriminateur

            # Génération d'un bruit aléatoire pour générer des images
            ''' 
            On utilise tf.random.normal pour générer un bruit aléatoire suivant une distribution normale (gaussienne). 
            Cette distribution est utilisée car elle facilite l'apprentissage et permet de couvrir un large espace latent.
            Cela aide le générateur à explorer différentes variations et à produire des images diversifiées.'''
            noise = tf.random.normal([batch_size, latent_dim])
            # Génération de fausses images
            fake_images = generator.predict(noise)
            # Sélection d'un lot/batch d'images réelles provenant des données d'entraînement
            real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
            real_images += tf.random.normal(tf.shape(real_images), mean=0.0, stddev=0.05)  # Add small noise
            real_images = tf.clip_by_value(real_images, -1.0, 1.0)  # Keep values in range [-1, 1]

            
            # Attribution des labels : 1 pour les images réelles, 0 pour les images générées
            # Le discriminateur était trop fort
            real_labels = tf.ones((batch_size, 1)) * 0.9  # Label smoothing
            fake_labels = tf.zeros((batch_size, 1))

            '''
            On entraine le discriminateur sur les vraies et fausses images pour qu'il apprenne à bien classer les vraies images 
            comme étant vraies (label = 1) et celles générées comme étant fausses (label = 0).
            '''
            #Entrainement adversarial du discriminateur/générateur
            discriminator.trainable = True
            # Entraînement du discriminateur sur les vraies images
            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            # Entraînement du discriminateur sur les fausses images
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
            discriminator.trainable = False

            # Entraînement du générateur
            
            # Il tente de tromper le discriminateur en générant des images réaliste
            misleading_labels = tf.ones((batch_size, 1)) # Fait croire au discriminateur que les images générées sont réelles
            # Train generator twice to compensate for strong discriminator
            for _ in range(2):
                g_loss = gan.train_on_batch(noise, misleading_labels)
            # Mise à jour des poids du générateur

            ''' train_on_batch est une fonction qui entraîne le modèle sur un seul batch de données au lieu d'une epoch entière
                ce qui permet une mise à jour plus fréquente des poids et donc une convergence plus rapide.
            '''
        print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss_real[0] + d_loss_fake[0]}, G Loss: {g_loss}")
        


# Génération d'images après l'entraînement
def generate_images(generator, n_images, latent_dim=100):
    '''
    - On crée un vecteur de bruit aléatoire pour produire de nouvelles images.
    - On utilise le générateur qui a été entrainé pour transformer ce bruit en images réalistes.
    '''
    noise = tf.random.normal([n_images, latent_dim])
    generated_images = generator.predict(noise)  # Génération des images
    fig, axes = plt.subplots(1, n_images, figsize=(20, 4))
    for i, img in enumerate(generated_images):
        '''
        Les images générées ont une forme (28, 28, 1), ce qui signifie qu'elles ont un canal supplémentaire utilisé pour représenter le gris.
        `squeeze()` permet d'éliminer cette dimension inutile pour avoir une dimension 2D (28,28) attendues pour afficher correctement l'image.
        '''
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].axis("off")
    plt.show()


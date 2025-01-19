# Fichier : vae/loss_layer.py
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class VAELossLayer(Layer):
    """
    Classe VAELossLayer :
    Cette couche calcule la perte combinée du VAE.

    - Le VAE utilise deux types de pertes :
      1. Reconstruction Loss : Qui compare l'image d'entrée et l'image reconstruite. Son objectif est de s'assurer que le décodeeur peut recréer l'image d'origine à partir du vecteur latent.
         Elle permet de vérifier que l'espace latent contient assez d'informations pour reconstruire l'image.
      2. KL Divergence : Qui contraint la distribution latente à suivre une distribution normale standard (N(0, 1)). Son objectif est de régulariser l'espace latent pour qu'il soit continu.
         On peut ainsi échantilloner des points dans cet espaces pour générer de nouvelles images, même si ces points n'étaient pas vu à l'entrainement car les points proche dans cet espace correspondront
         à des données similaires.

    Paramètres :
    - loss_function : Fonction de perte pour la reconstruction. Par exemple, la binary_crossentropy qui est bien adaptée car les valeurs de sortie ici sont normalisée étant entre 0 et 1. 
                      Cette dernière traite chaque pixel comme une probabilité qu'il appartienne à une classe (0 ou 1).
    - beta : Il permet de contrôler l'importance relative de la KL Divergence par rapport à la reconstruction Loss. Se concentrer soit sur la précision des données ou la générativité.

    """

    def __init__(self, loss_function, beta=1.0, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.loss_function = loss_function  # Fonction de perte pour la reconstruction.
        self.beta = beta # Coefficient divergence KL

    def compute_loss(self, x, x_decoded, z_mean, z_log_var):
        # Reconstruction Loss (importance de chaque pixel dans l'image)
        # On aplatit les images pour traiter chaque pixel indépendamment et on multiplie par 28*28 pour ramener à l'échelle du nombre total de pixel de l'image
        reconstruction_loss = self.loss_function(K.flatten(x), K.flatten(x_decoded)) * 28 * 28

        """ KL Divergence : Contraint l'espace latent à suivre une distribution normale standard
        La formule consiste à mesurer la différence entre deux distributions, ici celle apprise par l'encodeur et une distribution normale.
        La formule générale consiste en une intégrale (car distribution continue) sur la distribution latente de la différence entre le logarithme de la distribution latente 
        et la distribution normale (représentant la divergence entre les deux), multiplié par la distribution latente pour donner plus d'importance aux zones où elle est élevée et donc signigicative.

        Mathématiquement, en considérant deux gaussiennes N(mu, sigma) et N(0, 1) : on arrive bien à -0.5 * Somme(1 + z_log_var - (z_mean)^2 - var))
        """
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1) * -0.5

        # Retourner la somme des deux pertes
        return K.mean(reconstruction_loss + self.beta * kl_loss)

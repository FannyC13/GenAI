# Fichier : vae/encoder.py
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class Encoder:
    """
    L'encodeur a pour but de compresser les données d'entrée (ici ce sont des images de dimensions 28x28)
    en une représentation compacte dans l'espace latent. Un espace latent est un espace dans lesquels les données d'entrée sont compressées 
    et représentées sous forme de vecteur numérique. Cette compression permettra de capturer les informations essentielles de l'image afin 
    d'apprendre une distribution latente (relationn entre les données dans l'espace latent).

    Paramètres :
    - input_shape : Dimensions de l'image d'entrée (28, 28, 1).
    - latent_dim : Entier, taille de l'espace latent.

    Ce qu'on va faire ici c'est donc :
    - Compresser les données d'entrée 
    - Définir une distribution probabilistique (représenter chaque donnée par une distribution gaussienne) par deux paramètres z_mean et z_log_var
    - Echantillonner un vecteur latent z pour représenter la donnée dans l'espace latent. Cependant, tirer directement z rendrait le modèle non différentiable car l'opération d'échantillonnage est intrinsèquement sochastique
      donc aléatoire. Une opération aléatoire ne permet pas de calculer de dérivées parce qu'elle ne dépend pas de manière continue des paramètres z_mean et z_log_var. On utilise
      ainsi la fonction de reparamétrisation qui est une technique permettant de rendre le modèle différentiable. En effet, au lieu de tirer un vecteur z directement de la distribution
      on décompose z en deux partie une partie apprise qui définit la distribution et une partie aléatoire. La partie apprise (z_mean et z_log_var) est déterminée par le modèle et différentiable, 
      ce qui permet au modèle d'apprendre efficacement. La partie aléatoire, indépendante, permet d'échantillonner z pour capturer les variations naturelles des données.
    """

    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape  # Dimensions des données d'entrée, ici 28x28 pixels avec 1 canal (niveaux de gris).
        self.latent_dim = latent_dim  # Nombre de dimensions dans l'espace latent.
        self.model = self.build_encoder()  # Construction du modèle encodeur.

    def build_encoder(self):
        """
        Construction de l'encodeur :
        - Les images sont d'abord aplaties en vecteurs linéaires (Flatten).
        - Une couche Dense extrait les caractéristiques importantes.
        - Deux sorties (z_mean et z_log_var) définissent la distribution latente gaussienne.
        - Une fonction de reparamétrisation permet l'échantillonnage différentiable.
        """
        inputs = Input(shape=self.input_shape, name='encoder_input')
        
        # On aplatit les images 28x28 en un vecteur de 784 éléments
        x = Flatten()(inputs)
        
        # On extrait des caractéristiques avec une couche dense (256 unités, valeur courante pour capturer les caractéristiques complexes)
        
        x = Dense(256, activation='relu')(x)
        
        # On calcule ensuite la moyenne (z_mean) et le logarithme de la variance (z_log_var) de la distribution latente
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # Cecei est la fonction de reparamétrisation pour permettre l'échantillonnage différentiable
        def sampling(args):
            """
            Formule : z = z_mean + exp(0.5 * z_log_var) * epsilon
            Où epsilon est la partie aléatoire. Le log permet une représentation plus stable pour mieux gérer les gradients. 
            On applique l'exponentielle pour obtenir la variance (0.5 c'est pour obtenir la racine carrée de la variance)
            """
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # Créer le modèle de l'encodeur
        return Model(inputs, [z_mean, z_log_var, z], name='encoder')

    def get_model(self):
        return self.model

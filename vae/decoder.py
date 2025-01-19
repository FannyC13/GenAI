# Fichier : vae/decoder.py
from tensorflow.keras.layers import Dense, Reshape, Input
from tensorflow.keras.models import Model

class Decoder:
    """
    Classe Decoder :
    Le décodeur est responsable de reconstruire une image à partir de l'espace latent. C'est l'opération inverse de l'encodeur en prenant des points de l'espace latent 
    et en les transformant en une image de sortie.
    

    Paramètres :
    - latent_dim : taille de l'espace latent (doit correspondre à celle utilisée par l'encodeur).

    Rôle principal :
    - Reconstruire les images originales en utilisant les points de l'espace latent.
    """

    def __init__(self, latent_dim):
        self.latent_dim = latent_dim  # Taille de l'espace latent utilisée pour générer les images.
        self.model = self.build_decoder()  # Construction du modèle décodeur.

    def build_decoder(self):
        """
        Ce qu'on fait : 
        - Prendre un vecteur latent (de dimension latent_dim) comme entrée.
        - Ce vecteur va passer par des couches Dense pour recréer les caractéristiques perdues. 
          Ces couches permettent de recréer progressivement les détails de l'image originale en apprenant les relations complexes entre les données
        - Reconstruire une image (28x28 pixels) avec une activation sigmoid pour limiter les valeurs des pixels entre 0 et 1, ce qui correspond au niveau de gris dans l'image.
        """
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        
        # Expansion du vecteur latent via une couche dense (256 unités, ce qui est cohérent avec l'encodeur pour capturer les relations complexes).
        x = Dense(256, activation='relu')(latent_inputs)
        
        # Reconstruction de l'image aplatie (28 * 28 = 784).
        x = Dense(28 * 28, activation='sigmoid')(x)
        
        # Reshape du vecteur aplati en une image 28x28x1 (1 pour le niveau de gris).
        outputs = Reshape((28, 28, 1))(x)

        # Créer le modèle du décodeur
        return Model(latent_inputs, outputs, name='decoder')

    def get_model(self):
        return self.model
import tensorflow as tf
from tensorflow.keras import layers


def build_generator(latent_dim=100):
    '''
    Generator : 
    - Prend un vecteur de bruit aléatoire de dimension `latent_dim` (100 par défaut)
    - Passe par des couches de transformation pour générer une image de 28x28 pixels ce qui correspond à la taille standard des images dans le dataset MNIST.

    Dimension : 
    On choisit ici 7x7 pour pouvoir atteindre à la fin 28x28 après plusieurs convolutions transposées (taille adaptée pour les images MNIST)
    De plus, 256 représente le nombre de filtres qui sont utilisés dans les couches de convolution transposée qui suivent la couche Dense. 
    Chaque filtre apprend à détecter un motif spécifique dans les données. Ces filtres sont appliqués pour transformer progressivement le vecteur de taille 7 * 7 * 256 en une image de taille 28 * 28.
    '''
    model = tf.keras.Sequential([
         
        # Couche Dense : transforme le bruit aléatoire en un vecteur de taille 7*7*256
        layers.Dense(7 * 7 * 256, input_dim=latent_dim),

        # Reshape pour transformer le vecteur en une matrice 3D de taille (7,7,256)
        # Cela prépare les données pour les couches de convolution transposée suivantes, qui s'attendent à des entrées en 3D.
        layers.Reshape((7, 7, 256)),
       
        # Convolution transposée pour augmenter la taille de l'image progressivement
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"),  # Passage à 14x14x128
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"),  # Passage à 28x28x64
        
        
        # Dernière couche de sortie : image normalisée entre -1 et 1 (activation tanh)
        layers.Conv2DTranspose(1, kernel_size=7, activation="tanh", padding="same")  
    ])
    return model

    '''
    Conv2DTranspose permet la déconvolution et les activations relu et tanh sont utilisées pour mettre de la non-linéarité dans le réseau
    (cf Readme pour plus d'explication).
    '''

def build_discriminator():
    '''
    Discriminateur :
    - Prend une image 28x28 comme entrée
    - Passe par plusieurs couches de convolutions pour extraire des caractéristiques
    - Produit une probabilité indiquant si l'image est réelle ou fausse
    '''
  
    '''
    Conv2D est une couche de convolution (cf ReadME).

    La couche Conv2D renvoie une nouvelle image transformée où chaque pixel est une combinaison linéaire des pixels de l'image d'entrée, pondérée par les filtres.
    Après cette opération de convolution, on applique une fonction d'activation pour introduire de la non-linéarité.
    '''
    
    model = tf.keras.Sequential([
        
        # Première couche de convolution qui réduit l'image de 28x28 à 14x14
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=(28, 28, 1)),
        layers.LeakyReLU(alpha=0.2),  # Activation LeakyReLU pour la non linéarité et éviter la suppression des gradients
      
        # Deuxième couche de convolution qui réduit l'image de 14x14 à 7x7
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        
        
        #On utilise 2 couches Conv2D car chaque couche extrait des caractéristiques de plus en plus complexes
        #Donc plus la profondeur du réseau est grande, plus les caractéristiques apprises sont précises
        
        # Aplatissement pour convertir la matrice 3D en un vecteur 1D
        layers.Flatten(),
        
        
        #On doit ensuite convertit la matrice 3D en un vecteur 1D utilisable par une couche Dense
        #Cela permet de relier les informations extraites par les convolutions à la décision finale du modèle.
        # Couche Dense finale avec activation sigmoid pour classifier (0 = faux, 1 = réel)
        layers.Dense(1, activation="sigmoid")  
    ])
    return model

  
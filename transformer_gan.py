import tensorflow as tf  # Importation de la bibliothèque TensorFlow pour le deep learning
import numpy as np  # Importation de NumPy pour la gestion des matrices et calculs numériques
import matplotlib.pyplot as plt  # Importation de Matplotlib pour la visualisation des images générées
from tensorflow.keras import layers


def transformer_generator(latent_dim=100):
    '''
    Générateur basé sur un Transformer :
    - Prend un vecteur de bruit aléatoire en entrée.
    - Passe par des couches de self-attention pour capturer les relations entre pixels.
    - Génère une image réaliste de 28x28 pixels.
    '''
    inputs = layers.Input(shape=(latent_dim,))
    
    # Utilisation de Dense pour une projection initiale du bruit aléatoire dans un espace plus structuré
    x = layers.Dense(7 * 7 * 256, activation="relu")(inputs)
    x = layers.Reshape((49, 256))(x) # Reshape en (7*7=49, 128)
    
    '''
    La couche Dense projette le vecteur latent dans un espace plus structuré (cf Readme.MD pour explication)
    L'activation ReLu est utilisée pour introduire de la non-linéarité dans le modèle, il est simple et rapide à calculer (cf Readme.MD pour explication)
    Le Reshape transforme ce vecteur en une matrice 2D (49,128) qui servira aux couches d'attention.
    '''
    
    # Ajout de l'encodage positionnel pour conserver la structure spatiale
    position_indices = tf.range(start=0, limit=49, delta=1) #Creation d'un vecteur d'indices de position allant de 0 à 48 (49 positions au total).
    position_embedding = layers.Embedding(input_dim=49, output_dim=256)(position_indices) #Utilise une couche d'embedding pour transformer les indices de position en vecteurs d'encodage de dimension 128. input_dim est le nombre total de positions possible et output_dim la dimension des vecteurs. 
    position_embedding = tf.expand_dims(position_embedding, axis=0)  # Ajout de l'encodage positionnel aux données d'entrée
    x = x + position_embedding 

    '''
    Les transformers n'ont pas de notion d'ordre, l'encodage positionnel est une technique qui permet donc de lui faire 
    comprendre l'odre des élements dans une séquence en ajoutant des informations sur la position des pixels.
    '''
    
    # Appliquer plusieurs mécanismes de self-attention
    for _ in range(4):  # Ajouter plusieurs couches de self-attention
        attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x) #Définir 4 têtes d'attention pour explorer en parallèle
        x = layers.LayerNormalization()(x + attention_output)  # Normalisation après l'attention
        dense_output = layers.Dense(256, activation="relu")(x)   # Ajout d'une couche Dense avec activation relu pour la non-linéarité
        dense_output = layers.Dropout(0.2)(dense_output) # Dropout pour améliorer la diversité
        x = layers.LayerNormalization()(x + dense_output)  # Normalisation après la couche Dense
    
    '''
    Ici on divise en 4 têtes pour explorer en parallèle. 
    On normalise ensuite pour stabiliser l'entraînement et améliorer la convergence. Cela permet d'éviter des gradients trop grands ou trop petits.
    '''
    x = layers.Reshape((7, 7, 256))(x)
    
    # Upsampling to 28x28
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)  # Aide à réguler l'entraînement
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(1, kernel_size=4, strides=1, padding="same", activation="tanh")(x)

    
    '''
    Après le MultiHeadAttention, on obtient une représentation enrichie des données d'entrée, où les relations complexes entre les différentes parties de l'image ont été capturées.
    On applique ensuite une couche Dense pour introduire de la non-linéarité et transformer les données pour capturer des caractéristiques supplémentaires et plus riches. 
    La combinaison des deux permet de capturer à la fois des caractéristiques locales (Dense) et globales (MultiHeadAttention) de l'image.

    On normalise à nouveau pour stabiliser l'entraînement.
    Enfin on reshape pour revenir à une structure d'image (7,7,128) car le tenseur était en (49;128)
    '''  
    '''
    On utilise ensuite Conv2DTranspose pour agrandir progressivement l'image jusqu'à atteindre (28,28,1).
    tanh est utilisé pour normaliser les valeurs entre [-1,1], ce qui correspond à la normalisation des données MNIST.
    '''
    
    model = tf.keras.Model(inputs, x, name="Transformer_Generator")

    '''
    On crée ensuite le modèle en spécifiant les entrées et les sorties.
    '''
    return model


def transformer_discriminator(): #(cf cnn_gan.py)
    '''
    Discriminateur  :
    - Passe par plusieurs couches de convolutions pour extraire des caractéristiques
    - Produit une probabilité indiquant si l'image est réelle ou fausse
    '''
    inputs = layers.Input(shape=(28, 28, 1))
    
    # Première convolution 
    x = layers.Conv2D(32, kernel_size=4, strides=2, padding="same")(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x) 
    
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.Model(inputs, x, name="Transformer_Discriminator")
    return model

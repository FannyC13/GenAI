import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

class SelfAttention(tf.keras.layers.Layer):
    """
    Implémentation d'un mécanisme de self attention
    Le self-attention permet à un modèle de déterminer quelles parties d'une phrase sont les plus importantes pour comprendre le contexte global. Il est utilisé dans des architectures transformers (cf branche Lab2 MultiHead)
    """
    def __init__(self, embedding_dim):
        """
        Initialise les poids des matrices de requête (Q), clé (K) et valeur (V)
        - Q représente ce que chaque mot cherche à comprendre dans les autres mots
        - K représente ce que chaque mot signifie et peut fournir comme information
        - V représente l'information réelle contenue dans chaque mot

        Ces matrices permettent de calculer l'attention entre les mots d'une phrase
        
        :param embedding_dim: Taille des vecteurs d'embedding (dimension de chaque mot après encodage)
        """
        super(SelfAttention, self).__init__()
        self.W_q = tf.keras.layers.Dense(embedding_dim)  # Matrice Q
        self.W_k = tf.keras.layers.Dense(embedding_dim)  # Matrice K
        self.W_v = tf.keras.layers.Dense(embedding_dim)  # Matrice V
        self.scale = tf.math.sqrt(tf.cast(embedding_dim, tf.float32))  # Normalisation des scores d'attention, on utilise scale pour éviter que les scores deviennent trop grands et donc pour stabiliser l'apprentissage
    
    def call(self, inputs):
        """
        Applique le mécanisme de self attention.
        
        :param inputs: Tensor représentant la séquence d'entrée (batch_size x séquence_length x embedding_dim)
        :return: Tensor après application du mécanisme de self-attention et scores d'attention.
        """
        # Création des matrices de requêtes, clés et valeurs à partir de l'entrée
        Q = self.W_q(inputs)  
        K = self.W_k(inputs)  
        V = self.W_v(inputs)
        
        # Calcul des scores d'attention en faisant un produit scalaire entre Q et K
        # Cela sert à mesurer la similarité entre chaque paire de mots dans la séquence
        attention_scores = tf.matmul(Q, K, transpose_b=True) / self.scale
        
        # Application de la fonction softmax pour obtenir des probabilités entre 0 et 1 ce qui permet de donner plus d'importance aux relations fortes
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Multiplication des poids d'attention par les valeurs (V) pour obtenir la nouvelle représentation
        output = tf.matmul(attention_weights, V)
        
        return output, attention_weights


def tokenize_sequence(sequence, vocab, word_to_id):
    """
    Convertit une phrase en indices en se basant sur un vocabulaire.
    
    :param sequence: La phrase d'entrée 
    :param vocab: La liste des mots connus
    :param word_to_id: Dictionnaire associant chaque mot à un indice unique
    :return: Liste d'entiers représentant la phrase sous forme de tokens
    """
    return [word_to_id[word] for word in sequence.split() if word in vocab]


def plot_attention(attention_weights, sequence):
    """
    Affiche une heatmap représentant les scores d'attention entre les mots.
    
    :param attention_weights: Matrice des scores d'attention.
    :param sequence: Liste des mots de la phrase analysée.
    """
    attention_weights = attention_weights.numpy()[0]  # Convertit les valeurs en tableau numpy
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_weights, xticklabels=sequence, yticklabels=sequence, cmap='viridis', annot=True)
    plt.xlabel('Clé (K) - Contexte des mots')
    plt.ylabel('Requête (Q) - Mots cherchant du contexte')
    plt.title('Matrice des Scores d Attention')
    plt.show()
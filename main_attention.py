import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from self_attention import SelfAttention, tokenize_sequence, plot_attention

# Définition du vocabulaire et de la séquence
test_sentences = [
    "je fais un exercice sur les mécanismes d'attention"
]

# Création du vocabulaire à partir des mots de la séquence
vocab = list(set(" ".join(test_sentences).split())) # Set pour éviter les doublons
vocab_size = len(vocab) # Nombre de mots
word_to_id = {word: i for i, word in enumerate(vocab)} # Association mots -> Indice
id_to_word = {i: word for word, i in word_to_id.items()}

# Paramètres d'hyperparamétrisation
embed_dim = 64
num_heads_list = [2, 20, 50, 100]  # Différents nombres de têtes d'attention à tester

# Initialisation de la couche d'embedding pour convertir les indeces de mots en vecteurs 
embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_dim)

# Traitement des phrases pour tester l'attention
for num_heads in num_heads_list:
    print(f"\nTest avec {num_heads} tête(s) d'attention")
    for sentence in test_sentences:
        print(f"\nPhrase : {sentence}")

        # Conversion de la phrase en une liste d'indices
        tokenized_sequence = tokenize_sequence(sentence, vocab,word_to_id)
        print("Séquence tokenisée :", tokenized_sequence)

        # Conversion de la séquence en tenseur
        input_sequence_ids = tf.constant([tokenized_sequence])
        embedded_input = embedding_layer(input_sequence_ids)
        
        # Instancier le SelfAttention pour tester plusieurs têtes
        attention_layer = SelfAttention(embed_dim // num_heads)
        attention_output, attention_weights = attention_layer(embedded_input)
        
        print("Scores d'attention:\n", attention_weights.numpy())
        plot_attention(attention_weights, sentence.split())

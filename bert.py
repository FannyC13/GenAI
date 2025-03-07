import tensorflow as tf
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, TFBertForSequenceClassification


# Load le dataset IMDb
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# IMDb est un dataset de critiques de films :
# - x_train contient une liste de phrases, mais ces phrases sont représentées par des séquences d'indices
# - y_train contient des étiquettes : 0 pour négatif, 1 pour positif.

# On récupère le dictionnaire des mots pour décoder les indices en texte
word_index = tf.keras.datasets.imdb.get_word_index()

# TensorFlow garde les indices 0 à 3 pour des tokens spéciaux, donc on doit décaler nos indices de 3
index_to_word = {i + 3: word for word, i in word_index.items()}
index_to_word[0] = "<PAD>"  # Token spécial pour les phrases courtes
index_to_word[1] = "<START>"  # Pour le début d'une phrase
index_to_word[2] = "<UNK>"  # Pour les mots inconnu unknown
index_to_word[3] = "<UNUSED>"  # mots non utilisé

def decode_review(integers):
    """
    Ce code est utilisé pour convertir la séquence d'indices en mots
    """
    return " ".join([index_to_word.get(i, "?") for i in integers])

# Utilisation de decode
train_texts = [decode_review(review) for review in x_train]
test_texts  = [decode_review(review) for review in x_test]

train_df = pd.DataFrame({'text': train_texts, 'label': y_train})
test_df  = pd.DataFrame({'text': test_texts, 'label': y_test})

# On transforme les phrases en une séquence de tokens pour BERT.

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(texts, labels, max_length=256):
    """
    Prépare les données pour BERT en les transformant en tokens et en masquant les parties inutiles.
    - add_special_tokens=True : On ajoute les tokens [CLS] et [SEP] (Cf Readme pour explication)
    - padding=max_length: Ajoute des <PAD> pour que toutes les phrases aient la même longueur.
    - truncation=True : Coupe les phrases trop longues.
    - return_attention_mask=True : Génère un masque indiquant quels mots sont importants.
    - return_tensors=tf : Convertit le résultat en format TensorFlow.
    """
    encoded = tokenizer.batch_encode_plus(texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    return {
        'input_ids': encoded['input_ids'],  
        'attention_mask': encoded['attention_mask'],  
        'labels': tf.convert_to_tensor(labels, dtype=tf.int32)
    }

# Preprocess data
train_encodings = preprocess_data(train_df['text'].tolist(), train_df['label'].tolist())
test_encodings  = preprocess_data(test_df['text'].tolist(), test_df['label'].tolist())

# Trainet Test
train_dataset = tf.data.Dataset.from_tensor_slices(train_encodings).shuffle(10000).batch(32)
test_dataset  = tf.data.Dataset.from_tensor_slices(test_encodings).batch(32)

# Modèle Bert based Uncased
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(
    train_dataset, 
    epochs=3, 
    validation_data=test_dataset
)

# -----------------------------------------------
# 4. ÉVALUATION DU MODÈLE
# -----------------------------------------------

# Prédictions sur le dataset de test
predictions = model.predict(test_dataset)
pred_labels = np.argmax(predictions.logits, axis=1)

# Affichage des performances du modèle
print("Accuracy:", accuracy_score(test_encodings["labels"].numpy(), pred_labels))
print(classification_report(test_encodings["labels"].numpy(), pred_labels))

# -----------------------------------------------
# 5. UTILISATION DU MODÈLE POUR DE NOUVELLES PRÉDICTIONS
# -----------------------------------------------

def predict_sentiment(text):
    """
    Fonction pour prédire le sentiment d'une critique de film.
    Utilise le modèle BERT fine-tuné.
    """
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    logits = model(encoding)[0]
    pred = np.argmax(logits, axis=1)[0]
    return "Positif" if pred == 1 else "Négatif"

# Test de la fonction
print(predict_sentiment("I love this movie!"))  
print(predict_sentiment("This movie was terrible."))  

# Sauvegarde du modèle et du tokenizer
model.save_pretrained("final_saved_model")
tokenizer.save_pretrained("final_saved_model")

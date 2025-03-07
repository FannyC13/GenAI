import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# On échantillonne 1000 valeurs de x dans l'intervalle [-1, 1]
n_samples = 1000
x = np.linspace(-1, 1, n_samples)
y = np.sin(x)

# On ajoute un léger bruit sur y pour simuler des données réelles
noise = 0.1 * np.random.randn(n_samples)
y_noised = y + noise
data = np.column_stack((y_noised, x))

# On mélange les indices pour créer les ensembles de train et de test
indices = np.random.permutation(n_samples)
train_size = int(0.8 * n_samples)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

# Train et Test
X_train = data[train_idx, 0].reshape(-1, 1)
y_train = data[train_idx, 1].reshape(-1, 1)
X_test = data[test_idx, 0].reshape(-1, 1)
y_test = data[test_idx, 1].reshape(-1, 1)

# Visualisation
plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color='r', alpha=0.6, label='Données d\'entraînement')
plt.xlabel("y = sin(x) (avec bruit)")
plt.ylabel("x")
plt.title("Visualisation des données d'entraînement")
plt.legend()
plt.show()
plt.savefig("MLP_Train_Dataset.png")

# On souhaite approximer la fonction inverse de sin -> arcsin(y).
# On va construire un MLP avec :
# 1 neurone en entrée (y),1 couches cachée de 3 neurones chacune avec activation ReLU et 1 neurone en sortie (x).
model = keras.Sequential([
    layers.Dense(3, activation=lambda x: tf.math.atan(x), input_shape=(1,)),
    layers.Dense(1)
])

# On compile le modèle avec l'optimiseur Adam et la loss MSE
model.compile(optimizer='adam', loss='mse')

# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

x_new = np.linspace(-1,1, 200)
y_new = np.sin(x_new)

X_eval = y_new.reshape(-1, 1) 
y_eval = x_new.reshape(-1, 1)

# Évaluation du modèle
test_loss = model.evaluate(X_eval, y_eval, verbose=0)
print("Test Loss :", test_loss)

# Prédiction des valeurs de x par le modèle à partir des y de test
predictions = model.predict(X_eval)

# Vrai valeurs 
y_line = np.linspace(-1, 1, 200)
x_true_line = np.arcsin(y_line)

# Comparaison entre x prédit par le modèle et la fonction x = arcsin(y)
plt.figure(figsize=(8,6))
plt.scatter(y_eval, predictions, alpha=0.5, label="x prédit")
plt.plot( y_line, x_true_line, color='red', linewidth=2, label="Fonction: x = arcsin(y)")
plt.xlabel("y")
plt.ylabel("x")
plt.title("x prédit vs. x vrai")
plt.legend()
plt.show()
plt.savefig("MLP_Arcsin.png")

# Visualisation de la loss pendant l'entrainement
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label="Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Evolution de la loss")
plt.legend()
plt.savefig("MLP_Loss.png")
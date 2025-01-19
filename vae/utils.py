import matplotlib.pyplot as plt

def plot_images(original, decoded):
    """
    Affiche les images originales et reconstruites pour comparaison.

    Paramètres :
    - original : Images originales de test.
    - decoded : Images reconstruites par le modèle.
    """
    n = 10  # Nombre d'images à afficher.
    fig, axes = plt.subplots(2, n, figsize=(20, 4))
    for i in range(n):
        # Ligne 1 : Images originales
        axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')

        # Ligne 2 : Images reconstruites
        axes[1, i].imshow(decoded[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
    plt.show()

def save_models(encoder, decoder):
    """
    Sauvegarde les poids des modèles encodeur et décodeur.
    
    Paramètres :
    - encoder : Modèle encodeur à sauvegarder.
    - decoder : Modèle décodeur à sauvegarder.
    """
    encoder.save_weights('vae_encoder.weights.h5')
    decoder.save_weights('vae_decoder.weights.h5')

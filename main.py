import tensorflow as tf
from tensorflow.keras import layers
from data_load import load_mnist
from cnn_gan import build_generator as build_cnn_generator, build_discriminator as build_cnn_discriminator
from transformer_gan import transformer_generator, transformer_discriminator
from train_gan import train_gan, generate_images

def main(model_type='cnn', load_model=False):
    latent_dim = 100 # Dimension du vecteur latent, utilisé comme entrée pour le générateur
    x_train = load_mnist() # Chargement des données MNIST

    # Sélection du type de modèle GAN
    if model_type == 'cnn':
        print("Using CNN-based GAN")
        generator = build_cnn_generator(latent_dim)
        discriminator = build_cnn_discriminator()
    elif model_type == 'transformer':
        print("Using Transformer-based GAN")
        generator = transformer_generator(latent_dim)
        discriminator = transformer_discriminator()
    else:
        raise ValueError("Invalid model_type. Choose 'cnn' or 'transformer'.")
    
    # Si modèle mis en input, utilisé ce dernier
    if load_model:
        generator.load_weights("generator_epoch_50.h5")
        discriminator.load_weights("discriminator_epoch_50.h5")
    
    #Structure des modèles
    generator.summary()
    discriminator.summary()
    
    # Compilation du discriminateur
    '''Compile permet de configurer le modèle pour l'entrainement avec l'optimiseur, la fonction de loss (binary_crossentropy car classification binaire) et les métriques (accuracy)'''
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002), loss="binary_crossentropy", metrics=["accuracy"])
    discriminator.trainable = False  # On s'assure que le discriminateur n'est pas entraîné lorsque le modèle GAN est compilé.
    
    # Création du modèle GAN (générateur + discriminateur)
    gan_input = layers.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))  # Passage de l'entrée dans le générateur puis dans le discriminateur
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002), loss="binary_crossentropy")
    
    # Le discriminateur et le générateur sont entraînés alternativement (adversarial) dans la fonction train_gan
    if not load_model:
        train_gan(generator, discriminator, gan, x_train, epochs=50) 
    generate_images(generator, 10)
    
    # Sauvegarde des modèles
    if model_type == 'cnn':
        generator.save("cnn_generator.h5")
        discriminator.save("cnn_discriminator.h5")
    elif model_type == 'transformer':
        generator.save("transformer_generator.h5")
        discriminator.save("transformer_discriminator.h5")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["cnn", "transformer"], default="cnn", help="Choose GAN model type")
    parser.add_argument("--load_model", action="store_true", help="Load pre-trained models")
    args = parser.parse_args()
    
    main(model_type=args.model, load_model=args.load_model)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ===========================
# 1. Chargement de CIFAR-10
# ===========================
(x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("x_train shape:", x_train.shape)  # (50000, 32, 32, 3)
print("x_test shape:", x_test.shape)    # (10000, 32, 32, 3)
img_shape = (32, 32, 3)

# ===========================
# 2. Noise Schedule (cosine)
# ===========================
# On définit T=200 pour avoir un débruitage progressif.
T = 200

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Schedule inspiré de "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal).
    timesteps : nombre d'étapes (T).
    s : petit offset pour éviter les valeurs extrêmes de cos().
    Retourne un tableau betas de taille timesteps.
    """
    steps = timesteps
    x = np.linspace(0, timesteps, timesteps+1)
    alphas_cum = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cum = alphas_cum / alphas_cum[0]
    betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
    return np.clip(betas, 0, 0.999)

betas = cosine_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)  # alpha_bar
alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

# On prépare des utilitaires
def alpha_bar(t):
    return alphas_cumprod[t]

def alpha_(t):
    return alphas[t]

def get_index_from_list(vals, t, x_shape):
    """
    Sélectionne vals[t] et reshape en (batch, 1, 1, 1) pour être broadcasté.
    """
    batch_size = tf.shape(t)[0]
    out = tf.gather(tf.convert_to_tensor(vals, dtype=tf.float32), t)
    return tf.reshape(out, (batch_size, 1, 1, 1))

# ===========================
# 3. Forward Diffusion (q_sample)
# ===========================
@tf.function
def q_sample(x0, t, noise):
    """
    x_t = sqrt(alpha_bar[t])*x0 + sqrt(1-alpha_bar[t])*noise
    t : (batch,) indices
    x0, noise : (batch,32,32,3)
    """
    sqrt_alpha_bar_t = get_index_from_list(alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alpha_bar_t = tf.sqrt(1.0 - sqrt_alpha_bar_t**2)  # pas tout à fait correct
    # Correction : sqrt_alpha_bar_t est déjà la racine de alpha_bar ? Non, on veut
    #   sqrt_alpha_bar_t = sqrt(alpha_bar[t]) -> on peut faire:
    sqrt_alpha_bar_t = tf.sqrt(get_index_from_list(alphas_cumprod, t, x0.shape))
    sqrt_one_minus_alpha_bar_t = tf.sqrt(1 - get_index_from_list(alphas_cumprod, t, x0.shape))

    return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

# ===========================
# 4. U-Net plus grand
# ===========================
# On ajoute GroupNorm, plus de filtres, etc.
# Embedding temporel sinusoïdal (simplifié).
# ===========================
class TimeEmbedding(layers.Layer):
    """
    Embedding sinusoïdal minimal pour le temps.
    """
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim

    def call(self, t):
        """
        t : (batch,) indices entiers
        On génère un embedding sin/cos.
        """
        half_dim = self.dim // 2
        freqs = tf.exp(tf.linspace(tf.math.log(1.0), tf.math.log(10000.0), half_dim))
        freqs = tf.reshape(freqs, (1, -1))
        t = tf.cast(t, tf.float32)
        t = tf.reshape(t, (-1,1))
        sinus = t * freqs
        sin = tf.math.sin(sinus)
        cos = tf.math.cos(sinus)
        emb = tf.concat([sin, cos], axis=-1)  # (batch, dim)
        return emb

def GroupNorm(x, groups=32):
    return layers.GroupNormalization(groups=groups, axis=-1)(x)

def conv_block(x, filters, kernel_size=3):
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = GroupNorm(x)
    x = tf.nn.silu(x)
    return x

def upsample_block(x, filters):
    x = layers.UpSampling2D((2,2))(x)
    x = conv_block(x, filters)
    return x

def downsample_block(x, filters):
    x = layers.Conv2D(filters, kernel_size=4, strides=2, padding='same')(x)
    x = GroupNorm(x)
    x = tf.nn.silu(x)
    return x

def build_unet(img_shape=(32,32,3), time_embed_dim=256):
    """
    U-Net plus large avec groupnorm et embedding temporel sinusoidal.
    """
    image_input = keras.Input(shape=img_shape, name="image_input")
    t_input = keras.Input(shape=(), dtype=tf.int32, name="t_input")  # un entier

    # Embedding temporel sin/cos
    t_emb_layer = TimeEmbedding(dim=time_embed_dim)
    t_emb = t_emb_layer(t_input)
    # On passe par quelques Dense
    temb = layers.Dense(time_embed_dim, activation=tf.nn.silu)(t_emb)
    temb = layers.Dense(time_embed_dim, activation=tf.nn.silu)(temb)

    # --- Encoder
    x = conv_block(image_input, 64)
    x = conv_block(x, 64)
    skip1 = x
    x = downsample_block(x, 128)

    x = conv_block(x, 128)
    x = conv_block(x, 128)
    skip2 = x
    x = downsample_block(x, 256)

    x = conv_block(x, 256)
    x = conv_block(x, 256)
    skip3 = x
    x = downsample_block(x, 256)

    # Bottleneck
    x = conv_block(x, 256)
    # On ajoute le time embedding (en le transformant en (batch,1,1,256) et en l'ajoutant)
    temb_broadcast = layers.Dense(256)(temb)
    temb_broadcast = tf.reshape(temb_broadcast, [-1,1,1,256])
    x = x + temb_broadcast
    x = conv_block(x, 256)

    # --- Decoder
    x = upsample_block(x, 256)
    x = layers.Concatenate()([x, skip3])
    x = conv_block(x, 256)
    x = conv_block(x, 256)

    x = upsample_block(x, 128)
    x = layers.Concatenate()([x, skip2])
    x = conv_block(x, 128)
    x = conv_block(x, 128)

    x = upsample_block(x, 64)
    x = layers.Concatenate()([x, skip1])
    x = conv_block(x, 64)
    x = conv_block(x, 64)

    out = layers.Conv2D(3, 1, padding='same')(x)  # bruit prédit

    model = keras.Model(inputs=[image_input, t_input], outputs=out)
    return model

model = build_unet()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
model.summary()

# ===========================
# 5. Entraînement
# ===========================
@tf.function
def train_step(x0):
    # x0 : (batch,32,32,3)
    batch_size = tf.shape(x0)[0]
    # Tirage d'un t aléatoire
    t = tf.random.uniform(minval=0, maxval=T, shape=(batch_size,), dtype=tf.int32)
    # Bruit aléatoire
    noise = tf.random.normal(shape=tf.shape(x0))
    # On calcule x_t
    x_t = q_sample(x0, t, noise)

    with tf.GradientTape() as tape:
        eps_pred = model([x_t, t], training=True)
        loss_val = tf.reduce_mean(tf.square(noise - eps_pred))

    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_val

def train_model(x_train, epochs=100, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(50000).batch(batch_size)
    steps_per_epoch = len(x_train)//batch_size

    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch_x in enumerate(dataset):
            loss_val = train_step(batch_x)
            total_loss += loss_val
        total_loss /= float(steps_per_epoch)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss.numpy():.4f}")

optimizer = keras.optimizers.Adam(learning_rate=1e-4)
train_model(x_train, epochs=100, batch_size=64)

# ===========================
# 6. Inference (Reverse)
# ===========================
# Formule standard : x_{t-1} = ...
# On ajoute un petit terme stochastique sigma_t si on veut,
# sinon on le met à 0 pour simplifier.
import math

def get_posterior_variance(t):
    # Variation du posterior dans DDPM
    # = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    # On va le calculer dynamiquement.
    betas_tf = tf.convert_to_tensor(betas, dtype=tf.float32)
    alpha_bar_tf = tf.convert_to_tensor(alphas_cumprod, dtype=tf.float32)
    alpha_bar_tf_prev = tf.concat([[1.0], alpha_bar_tf[:-1]], axis=0)
    batch_size = tf.shape(t)[0]
    # gather betas[t]
    beta_t = tf.gather(betas_tf, t)
    alpha_bar_t = tf.gather(alpha_bar_tf, t)
    alpha_bar_t_prev = tf.gather(alpha_bar_tf_prev, t)
    var = beta_t * (1.0 - alpha_bar_t_prev)/(1.0 - alpha_bar_t)
    return tf.reshape(var, (batch_size,1,1,1))

@tf.function
def p_sample(x_t, t):
    """
    x_{t-1} = 1/sqrt(alpha_t) * ( x_t - (1-alpha_t)/sqrt(1-alpha_bar_t} * eps_pred ) + sigma_t * z
    """
    betas_tf = tf.convert_to_tensor(betas, dtype=tf.float32)
    alphas_tf = tf.convert_to_tensor(alphas, dtype=tf.float32)
    alpha_bar_tf = tf.convert_to_tensor(alphas_cumprod, dtype=tf.float32)

    a_t = tf.gather(alphas_tf, t)
    ab_t = tf.gather(alpha_bar_tf, t)
    a_t = tf.reshape(a_t, (-1,1,1,1))
    ab_t = tf.reshape(ab_t, (-1,1,1,1))

    sqrt_recip_a_t = 1.0 / tf.sqrt(a_t)
    sqrt_one_minus_ab_t = tf.sqrt(1.0 - ab_t)

    eps_pred = model([x_t, t], training=False)
    x0_hat = (x_t - (1.0 - a_t)/ sqrt_one_minus_ab_t * eps_pred) * sqrt_recip_a_t

    # Posterior variance
    var = get_posterior_variance(t)
    # On échantillonne un z
    z = tf.random.normal(tf.shape(x_t))
    # On met un masque si t=0, pas de bruit
    mask = 1 - tf.cast(tf.equal(t, 0), tf.float32)
    x_tm1 = x0_hat + mask * tf.sqrt(var)*z
    return x_tm1

def p_sample_loop(batch_size=1):
    """
    On part de x_T ~ N(0,1) et on descend jusqu'à x_0.
    """
    x = tf.random.normal((batch_size, 32,32,3))
    for i in reversed(range(T)):
        t = tf.ones((batch_size,), dtype=tf.int32)*i
        x = p_sample(x, t)
    return x

# ===========================
# 7. Visualisation
# ===========================
num_images = 5
x_gen = p_sample_loop(batch_size=num_images)
x_gen_np = x_gen.numpy().clip(0,1)

# On prend 5 images du test pour comparer
real_imgs = x_test[:num_images]

# On crée 5 bruits purs pour affichage
noise_imgs = np.random.normal(0,1,(num_images,32,32,3)).astype(np.float32)
noise_imgs = (noise_imgs - noise_imgs.min())/(noise_imgs.max()-noise_imgs.min())

fig, axes = plt.subplots(3, num_images, figsize=(15,5))
for i in range(num_images):
    axes[0,i].imshow(noise_imgs[i])
    axes[0,i].axis('off')
    axes[1,i].imshow(x_gen_np[i])
    axes[1,i].axis('off')
    axes[2,i].imshow(real_imgs[i])
    axes[2,i].axis('off')

axes[0,0].set_ylabel("Bruité (départ)", fontsize=10)
axes[1,0].set_ylabel("Généré (débruité)", fontsize=10)
axes[2,0].set_ylabel("Original", fontsize=10)
plt.tight_layout()
plt.savefig("test_unet_results.png")
# Lab 1: Variational Autoencoders and Generative Adversarial Networks

## Configuration de l'environnement

```bash
python -m venv gen-ai
.\gen-ai\Scripts\activate
pip install -r requirements.txt
```

---

## Part 1: Implementing a Variational Autoencoder (VAE)


### 1. Pourquoi utilisons-nous l'astuce de reparamétrisation dans les VAE ?

Cette astuce est utilisé pour rendre l'apprentissage des VAE par rétropropagation. En effet, dans un VAE le but est de pouvoir échantillonner un vecteur latent z pour représenter la donnée dans l'espace latent. Les points de cet espace sont définis par une distribution probabilistique. Le problème est que tirer directement z rendrait le modèle non différentiable car l'opération d'échantillonnage est intrinsèquement stochastique et donc aléatoire. Il est cependant nécessaire de garder cette différentiabilité pour pouvoir entrainer le modèle. En effet, on doit pouvoir calculer des dérivées et l'opération d'échantillonnage ne permet pas de les calculer car elle ne dépend pas de manière continue des paramètres z_mean et z_log_var. C'est donc pour cette raison que l'on utilise la fonction de reparamétrisation qui permet de rendre le modèle différentiable mais aussi de pouvoir effectuer cette opération d'échantillage. Pour ce faire, au lieu de tirer un vecteur z directement de la distribution, ce dernier sera décomposé en deux parties, une apprise et une aléatoire. La partie apprise (z_mean et z_log_var) définit la distribution et est déterminée par le modèle et différentiable ce qui permet un apprentissage efficace. La partie aléatoire quant à elle permet d'échantilloner z pour capturer les variations du modèle.

--- 

###  2. Comment la perte de divergence KL influence-t-elle l'espace latent ?

La perte de divergence KL permet au VAE d'organiser l'espace latent. Elle oblige l'espace à suivre une distribution normale standard. Cela permet d'éviter que l'espace latent soit trop dispersé et donc mal structuré.S'il l'était, les points ne se regrouperaient pas bien et donc des points proches pourraient représenter des données différentes, ce qui empêcherait de bien échantilloner un point et donc que la donnée soit mal généré. Il est important donc de trouver un équilibre pour pas que la perte soit trop grande à risque d'ignorer les détails spécifiques ou trop faible à risque que le modèle mémorise chaque donnée comme un point isolé ce qui rendrait l'espace dispersé. 

---
### 3. Comment la taille de l’espace latent (latent_dim) impacte-t-elle la reconstruction ?

L'espace latent est un espace comprimé dans lequel sont stockés les informations importantes. Si cet espace est trop petit il manquera donc de capacité pour encoder toutes les informations nécessaires. De ce fait, les reconstructions seront floues ou incomplètes car des données seront perdues. En revanche, si l'espace est trop grand, le modèle mémorisera trop de détails et peut-être même du bruit. Cela pourrait désorganiser l'espace latent et rendre le modèle moins efficace pour générer de la donnée.

Il est donc important de trouver un équilibre pour retenir les informations importantes mais sans trop de détails.

---

## Part 2: From VAE to GAN

### 1. Comment le décodeur d’un VAE peut être utilisé comme générateur dans un GAN ?

Le décodeur d'un VAE est un réseau qui apprend à reconstruire des données à partir de l'espace latent qui capture des caractéristiques essentielles des données d'origine. Dans un GAN, le générateur va prendre des vecteurs aléatoire et générer des données réalistes. Le décodeur peut être utilisé comme générateur car ce dernier à été entrainé pour produire des données semblables à celles d'entrainement et donc de générer des exemples à partir de l'espace latent. De plus l'espace latent d'un VAE est forcé de suivre une distribution normale, ce qui est nécessaire pour le GAN pour les échantillons utilisés. 

---

### 2. Différences entre l’encodeur d’un VAE et le discriminateur d’un GAN

L'encodeur d'un VAE et le discriminateur d'un GAN ont pour but d'apprendre à extraire des informations utile. Cependant l'encodeur du VAE apprend à projeter des données dans un espace latent pour en capturer les caractéristiques essentielles. De cela, il produit une distribution probabilistique pour chaque données afin que le décodeur puisse reconstruire les données d'origine. Le discriminateur quant à lui apprend à différencier les données réelles des données générées par le générateur. Ainsi, en sortie le discrimateur rend une probabilité qui indique si l'entrée est réelle ou non.
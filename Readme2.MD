# Lab : Introduction to Stable Diffusion Models

## Configuration de l'environnement

```bash
python -m venv gen-ai
.\gen-ai\Scripts\activate 
pip install -r requirements.txt
```

## Part 1: Preliminary Activity - Neural Network for Function Inversion

Les images de visualisations sont dans MLP_Arcsin/
### What happens for values outside the range [-1,1]?

sin(x) est une fonction monotone sur cet intervalle et de ce fait l'inverse est bien défini. En effet, elle est bijective entre [-1, 1].
Les valeurs en dehors de cet interval sortent du domaine de définition d'arcsin, de ce fait les résultats seraient imprécis ou totalement faux car il n'aura pas appris ces valeurs. 
L'inversion va donc échouer en dehors car la fonction n'est pas bijective et donc l'opération d'inversion sera impossible 

### What are the implications of approximating inverses in more complex functions?

Si on travaille avec des fonctions plus complexes que sin on peut avoir des fonctions non bijectives sur l'ensemble de leur domaine ce qui complique la tâche car l'inverse ne sera donc pas unique. De plus certaines fonctions ne seront pas continus ou auront des variations rapide et de ce fait on devra repérer les variations et irrégularités. Ainsi il faut faire fortement attention aux erreurs qui peuvent entrainer de grandes variations étant donné que ce sont des inverses ce qui est beaucoup plus délicat. 

### Discussion:
####  What happens if we change the noise schedule?

En modifiant le noise schedule on change la manière dont le bruit est ajouté aux images à chaque étape. On peut donc avoir un processus plus brutal ou progressif ce qui peut impacter la capacité du modèle à apprendre à inverser. 

#### How do diffusion models compare to GANs?

Les modèles de diffusions et les Gans génèrent tous les deux des images mais de manière différente. Les Gans sont adversarial entre un générateur et un discriminateur alors qu'un modèle de diffusion génère du bruit et l'enlève progressivement grâce à un processus itératif. Les diffusions model sont généralement plus stable mais plus lent.

## Part 2: Diffusion Models on Images

### How does function inversion relate to diffusion models?


### How does iterative noise removal help generate realistic images?

###  Potential applications of diffusion models (e.g., text-to-image generation like Stable Diffusion).



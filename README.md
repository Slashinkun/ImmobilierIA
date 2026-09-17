# ImmobilierIA

## Description

Projet de L3 Informatique en binôme à **l'UPEC (Université Paris-Est Créteil).**

**ImmobilierIA** est un projet d'intelligence artificielle permettant de prédire le prix d'un appartement à partir de différentes caractéristiques telles que sa surface, son nombre de pièces, etc.

Le projet est entièrement développé en **Python**.

> **Attention** : les données récupérées sont limitées à la région Île-de-France, ne sont plus à jour et reflètent plus le prix actuels de l’immobilier.

## En quoi consiste le projet ?

Le projet se déroule en plusieurs étapes :


- **Scraping des données** : récupération de données immobilières à partir d'un site immobilier à l'aide de **BeautifulSoup** et **requests**
- **Nettoyage des données** : traitement et préparation des données avec **Pandas**
- **Apprentissage** : entraînement et comparaison de différents modèles de machine learning avec **Scikit-learn**
  - Régression linéaire
  - Arbre de décision
  - K plus proches voisins (KNN)
- **Prédiction** : utilisation des modèles entraînés pour estimer le prix d'un appartement à partir de ses caractéristiques


## Mode d'emploi

### Installation

Installer les dépendances du projet à partir du fichier `requirements.txt`

```pip install -r requirements.txt```


### Modifier les caractéristiques de l'appartement

Pour modifier les caractéristiques utilisé pour la prédiction, ouvrez le fichier `ImmoIA.py` et modifier les valeurs passés à la fonction `.predict()`.

Les caractéristiques doivent être renseignées dans un tableau, dans l'ordre attendu par le modèle.


### Lancer une prédiction  

Pour effectuer une prédiction, lancer le fichier Main.py :

``python Main.py``

Vous pouvez également lancer le fichier directement depuis un IDE.

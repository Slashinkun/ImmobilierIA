from DataScraper import *
from DataCleaning import *
from ApprentissageIA import *

from sklearn.model_selection import train_test_split

#
# Partie 1: Data Scraping
#
# scrapLink("https://www.immo-entre-particuliers.com/annonces/france-ile-de-france/vente/ta-offer")

#
# Partie 2: Data Cleaning
#
annonces = read_csv('data.csv', encoding='latin1')
villes = read_csv('cities.csv')

annonces = preparer_donnees_ia(annonces, villes)

#
# Partie 3: Apprentissage et tests
#
X = annonces.drop("Prix",axis="columns")
y = annonces["Prix"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=49, train_size=0.75,test_size=0.25)

scores, y_pred = apprentissage(X_train, X_test, y_train, y_test) # Apprentissage sur 9 méthodes et obtiens leur scores et les prédictions réalisés

scores_lr = [scores['LR'], scores['LR Normal'], scores['LR Standard']]
scores_ad = [scores['DTR'], scores['DTR Normal'], scores['DTR Standard']]
scores_knn = [scores['KNN'], scores['KNN Normal'], scores['KNN Standard']]
meilleur_scores = [max(scores_lr), max(scores_ad), max(scores_knn)]

afficherTableau(['LR', 'Normalisation + LR', 'Standardisation + LR'], scores_lr)
print()
afficherTableau(['AD', 'Normalisation + AD', 'Standardisation + AD'], scores_ad)
print()
afficherTableau(['KNN', 'Normalisation + KNN', 'Standardisation + KNN'], scores_knn)
print()
afficherTableau(['LR', 'AD', 'KNN'], meilleur_scores)


# Meilleur méthode
Meilleur_methode = max(scores, key=scores.get)
print(f"\nMeilleur méthode: {Meilleur_methode} \navec le score: {scores[Meilleur_methode]}\n")


# Meilleur méthode avec PCA = 2
X_train_pca, X_test_pca = convertir_en_pca(X_train, X_test, 2)
score_pca, y_pred_pca = apprentissage(X_train_pca, X_test_pca, y_train, y_test, Meilleur_methode)


# Meilleur méthode avec matrice de corrélation
corr_df = annonces.corr(method='pearson').filter(items=["Prix"])
corr_df = corr_df.drop("Prix", axis=0)
cinq_meilleur_caracteristiques = corr_df.sort_values(by="Prix", ascending=False).head(5).index.tolist() # Tri dans l'ordre décroissant, récupère l'index des 5 meilleur et le change en liste

new_X = X.filter(items=cinq_meilleur_caracteristiques)
new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(new_X, y, random_state=49, train_size=0.75,test_size=0.25)
score_corr, y_pred_corr = apprentissage(new_X_train, new_X_test, new_y_train, new_y_test, Meilleur_methode)


# Affichages
print(f"Score de la meilleur méthode: {scores[Meilleur_methode]}")
print(f"Score de la meilleur méthode avec PCA=2: {score_pca[Meilleur_methode]}")
print(f"Score de la meilleur méthode avec corrélation: {score_corr[Meilleur_methode]}")
print(f"Les cinqs meilleures caractéristiques de la matrice corrélation: {cinq_meilleur_caracteristiques}")

# print(corr_df)

afficher_graphique(y_pred, y_test, Meilleur_methode)
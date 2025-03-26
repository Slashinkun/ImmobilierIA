from sklearn.model_selection import train_test_split
from DataCleaning import *

import pandas as pd 
from sklearn.metrics import r2_score
from sklearn import linear_model

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

#Récupération des données

annonces = pd.read_csv('data.csv', encoding='latin1')
annonces['DPE'] = annonces['DPE'].replace('-', 'Vierge')
villes = read_csv('cities.csv')

clean_all(annonces)
annonces = splitMergeAll(annonces)
annonces = mergeVille(annonces, villes)
annonces = annonces.dropna()

#Préparation des données

X = annonces.drop("Prix",axis="columns")
y = annonces["Prix"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=49, train_size=0.75,test_size=0.25)


def apprentissage(X_train, X_test, methode):

    scores = {}
    y_pred = {}

    #Regression Linéaire :
    if methode in ["TOUT", "LR"]:
        lin_reg = linear_model.LinearRegression()
        lin_reg.fit(X_train,y_train)
        pred_lr = lin_reg.predict(X_test)
        score_lr = lin_reg.score(X_test,y_test)
        scores["LR"] = score_lr
        y_pred["LR"] = pred_lr


    #StandardScaler + LinearRegression
    if methode in ["TOUT", "Standardisation + LR"]:
        scale_lr = make_pipeline(StandardScaler(),linear_model.LinearRegression())
        scale_lr.fit(X_train,y_train)
        scaled_pred = scale_lr.predict(X_test)
        score_std = scale_lr.score(X_test,y_test)
        scores["Standardisation + LR"] = score_std
        y_pred["Standardisation + LR"] = scaled_pred


    #MinMaxScaler + LinearRegression
    if methode in ["TOUT", "Normalisation + LR"]:
        minmax_lr = make_pipeline(MinMaxScaler(),linear_model.LinearRegression())
        minmax_lr.fit(X_train,y_train)
        minmax_lr_pred = minmax_lr.predict(X_test)
        score_minmax = minmax_lr.score(X_test,y_test)
        y_pred["Normalisation + LR"] = minmax_lr_pred
        scores["Normalisation + LR"] = score_minmax

    #Arbre de décision :

    #DecisionTreeRegressor
    if methode in ["TOUT", "AD"]:
        decision_tr = DecisionTreeRegressor(max_depth=4)
        decision_tr.fit(X_train,y_train)
        pred_dtr = decision_tr.predict(X_test)
        score_dtr = decision_tr.score(X_test,y_test)
        scores["AD"] = score_dtr
        y_pred["AD"] = pred_dtr


    #StandardScaler + DecisionTreeRegressor
    if methode in ["TOUT", "Standardisation + AD"]:
        std_dtr = make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=4))
        std_dtr.fit(X_train,y_train)
        std_dtr_pred = std_dtr.predict(X_test)
        score_std_dtr = std_dtr.score(X_test,y_test)
        scores["Standardisation + AD"] = score_std_dtr
    
        y_pred["Standardisation + AD"] = std_dtr_pred

    #MinMaxScaler + DecisionTreeRegressor
    if methode in ["TOUT", "Normalisation + AD"]:
        minmax_dtr = make_pipeline(MinMaxScaler(), DecisionTreeRegressor(max_depth=4))
        minmax_dtr.fit(X_train,y_train)
        minmax_dtr_pred = minmax_dtr.predict(X_test)
        score_minmax_dtr = minmax_dtr.score(X_test,y_test)
        scores["Normalisation + AD"] = score_minmax_dtr
        y_pred["Normalisation + AD"] = minmax_dtr_pred

    

    #N plus proches voisins :
    if methode in ["TOUT", "KNN"]:
        k_neigbors = KNeighborsRegressor(n_neighbors=5)
        k_neigbors.fit(X_train,y_train)
        k_neigbors_pred = k_neigbors.predict(X_test)
        score_kn = k_neigbors.score(X_test,y_test)
        scores["KNN"] = score_kn
        y_pred["KNN"] = k_neigbors_pred


    if methode in ["TOUT", "Standardisation + KNN"]:
        k_neigbors_std = make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=5))
        k_neigbors_std.fit(X_train,y_train)
        k_neigbors_std_pred = k_neigbors_std.predict(X_test)
        k_neigbors_std_score = k_neigbors_std.score(X_test,y_test)
        scores["Standardisation + KNN"] = k_neigbors_std_score
        y_pred["Standardisation + KNN"] = k_neigbors_std_pred

    if methode in ["TOUT", "Normalisation + KNN"]:
        k_neigbors_minmax = make_pipeline(MinMaxScaler(),KNeighborsRegressor(n_neighbors=5))
        k_neigbors_minmax.fit(X_train,y_train)
        k_neigbors_minmax_pred = k_neigbors_minmax.predict(X_test)
        k_neigbors_minmax_score = k_neigbors_minmax.score(X_test,y_test)
        scores["Normalisation + KNN"] = k_neigbors_minmax_score
        y_pred["Normalisation + KNN"] = k_neigbors_minmax_pred

    return scores, y_pred


scores, y_pred = apprentissage(X_train, X_test, "TOUT")

def printtableau(methodes, scores):
    methodes.insert(0, 'Méthode')
    scores.insert(0, 'r²')
    for i in range(len(methodes)):
        print(f"{methodes[i]}{" "*(30-len(methodes[i]))} | {scores[i]}")


scores_lr = [scores['LR'], scores['Normalisation + LR'], scores['Standardisation + LR']]
scores_ad = [scores['AD'], scores['Normalisation + AD'], scores['Standardisation + AD']]
scores_knn = [scores['KNN'], scores['Normalisation + KNN'], scores['Standardisation + KNN']]
meilleur_scores = [max(scores_lr), max(scores_ad), max(scores_knn)]

printtableau(['LR', 'Normalisation + LR', 'Standardisation + LR'], scores_lr)
print()
printtableau(['AD', 'Normalisation + AD', 'Standardisation + AD'], scores_ad)
print()
printtableau(['KNN', 'Normalisation + KNN', 'Standardisation + KNN'], scores_knn)
print()


printtableau(['LR', 'AD', 'KNN'], meilleur_scores)

M = max(scores.items())[0]




#A faire : Visualisation des résultats

plt.scatter(np.array(y_pred[M]), np.array(y_test), marker='*', c='green')
min_val = min(y_test.min(), y_pred[M].min())
max_val = max(y_test.max(), y_pred[M].max())
plt.plot([min_val, max_val], [min_val, max_val], c='red')
plt.xlabel('y_test')
plt.ylabel('estimation')
plt.show()

#Reduction des composants 
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)



score_pca, y_pred_pca = apprentissage(X_train_pca, X_test_pca, M)
print(score_pca[M])
print(scores[M])

corr_df = annonces.corr(method='pearson').filter(items=["Prix"])
print(corr_df)

new_X = X.filter(items=["Surface", "Nbr de salle de bains", "Nbr de chambres", "D", "Nbr de pieces"])

new_X_train, new_X_test, y_train, y_test = train_test_split(new_X, y, random_state=49, train_size=0.75,test_size=0.25)

score_corr, y_pred_corr = apprentissage(new_X_train, new_X_test, M)

print(score_corr[M])
print(score_pca[M])
print(scores[M])
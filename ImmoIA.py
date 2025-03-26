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


def apprentissage(X_train,X_test):

    scores = {}
    y_pred = {}

    #Regression Linéaire :

    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_train,y_train)
    pred_lr = lin_reg.predict(X_test)
    score_lr = lin_reg.score(X_test,y_test)
    print("Sans pretraitement + LR :", score_lr)
    scores["LR"] = score_lr
    y_pred["LR"] = pred_lr


    #StandardScaler + LinearRegression
    scale_lr = make_pipeline(StandardScaler(),linear_model.LinearRegression())
    scale_lr.fit(X_train,y_train)
    scaled_pred = scale_lr.predict(X_test)
    score_std = scale_lr.score(X_test,y_test)
    print("StandardScaler + LR :", score_std)
    scores['Standardisation + LR'] = score_std
    y_pred["Standardisation + LR"] = scaled_pred


    #MinMaxScaler + LinearRegression
    minmax_lr = make_pipeline(MinMaxScaler(),linear_model.LinearRegression())
    minmax_lr.fit(X_train,y_train)
    minmax_lr_pred = minmax_lr.predict(X_test)
    score_minmax = minmax_lr.score(X_test,y_test)
    print("MinMax + LR",score_minmax)
    y_pred["MinMaxScaler + LinearRegression"] = minmax_lr_pred
    scores["Normalisation + LR"] = score_minmax
    print("\n")

    #Arbre de décision :

    #DecisionTreeRegressor
    
    decision_tr = DecisionTreeRegressor(max_depth=4)
    decision_tr.fit(X_train,y_train)
    pred_dtr = decision_tr.predict(X_test)
    score_dtr = decision_tr.score(X_test,y_test)
    scores[f"AD"] = score_dtr
    print(f"AD : ",score_dtr)
    y_pred["AD"] = score_dtr


    #StandardScaler + DecisionTreeRegressor
    std_dtr = make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=4))
    std_dtr.fit(X_train,y_train)
    std_dtr_pred = std_dtr.predict(X_test)
    score_std_dtr = std_dtr.score(X_test,y_test)
    scores[f"Standardisation + AD"] = score_std_dtr
    print(f"StandardScaler + AD  :", score_std_dtr)
    y_pred["Standardisation + AD"] = std_dtr

    #MinMaxScaler + DecisionTreeRegressor
    minmax_dtr = make_pipeline(MinMaxScaler(), DecisionTreeRegressor(max_depth=4))
    minmax_dtr.fit(X_train,y_train)
    minmax_dtr_pred = minmax_dtr.predict(X_test)
    score_minmax_dtr = minmax_dtr.score(X_test,y_test)
    scores[f"Normalisation + AD"] = score_minmax_dtr
    print(f"MinMaxScaler + AD  :", score_minmax_dtr)
    y_pred["Normalisation + AD"] = minmax_dtr_pred

    print("\n")

    #N plus proches voisins :

    k_neigbors = KNeighborsRegressor(n_neighbors=5)
    k_neigbors.fit(X_train,y_train)
    k_neigbors_pred = k_neigbors.predict(X_test)
    score_kn = k_neigbors.score(X_test,y_test)
    scores[f"KNN"] = score_kn
    print(f"KNeighborsR :", score_kn)
    y_pred["KNN"] = k_neigbors_pred


    k_neigbors_std = make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=5))
    k_neigbors_std.fit(X_train,y_train)
    k_neigbors_std_pred = k_neigbors_std.predict(X_test)
    k_neigbors_std_score = k_neigbors_std.score(X_test,y_test)
    scores[f"Standardisation + KNN"] = k_neigbors_std_score
    print(f"StandardScaler + KNeighborsR :", k_neigbors_std_score)
    y_pred["Standardisation + KNN"] = k_neigbors_std_pred

    k_neigbors_minmax = make_pipeline(MinMaxScaler(),KNeighborsRegressor(n_neighbors=5))
    k_neigbors_minmax.fit(X_train,y_train)
    k_neigbors_minmax_pred = k_neigbors_minmax.predict(X_test)
    k_neigbors_minmax_score = k_neigbors_minmax.score(X_test,y_test)
    scores[f"Normalisation + KNN"] = k_neigbors_minmax_score
    print(f"MinMaxScaler + KNeighborsR :", k_neigbors_minmax_score)
    print("\n")
    y_pred["Normalisation + KNN"] = k_neigbors_minmax_pred

    return scores, y_pred


scores, y_pred = apprentissage(X_train, X_test)





#A faire : Visualisation des résultats

# plt.plot(np.array(y_pred),np.array(y_test))
# plt.xlabel('y_test')
# plt.ylabel('prediction')
# plt.show()

#Reduction des composants 
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
   

#print("Avec PCA à 2 composants:\n")

# apprentissage(X_train_pca,X_test_pca)




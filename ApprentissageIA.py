import pandas as pd 
from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

def apprentissage_methode(methode, X_train, X_test, y_train, y_test):
    modele = methode
    modele.fit(X_train, y_train)

    prediction_modele = modele.predict(X_test)
    score = modele.score(X_test, y_test)

    return score, prediction_modele

def linear_regression(X_train, X_test, y_train, y_test):
    lr = linear_model.LinearRegression()
    return apprentissage_methode(lr, X_train, X_test, y_train, y_test)

def linear_regression_standard(X_train, X_test, y_train, y_test):
    lr_standard = make_pipeline(StandardScaler(),linear_model.LinearRegression())
    return apprentissage_methode(lr_standard, X_train, X_test, y_train, y_test)

def linear_regression_minmax(X_train, X_test, y_train, y_test):
    lr_minmax = make_pipeline(MinMaxScaler(),linear_model.LinearRegression())
    return apprentissage_methode(lr_minmax, X_train, X_test, y_train, y_test)

def decision_tree_regressor(X_train, X_test, y_train, y_test):
    dtr = DecisionTreeRegressor(max_depth=4)
    return apprentissage_methode(dtr, X_train, X_test, y_train, y_test)

def decision_tree_regressor_standard(X_train, X_test, y_train, y_test):
    dtr_standard = make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=4))
    return apprentissage_methode(dtr_standard, X_train, X_test, y_train, y_test)

def decision_tree_regressor_minmax(X_train, X_test, y_train, y_test):
    dtr_minmax = make_pipeline(MinMaxScaler(), DecisionTreeRegressor(max_depth=4))
    return apprentissage_methode(dtr_minmax, X_train, X_test, y_train, y_test)

def kneighbors_regressor(X_train, X_test, y_train, y_test):
    knn = KNeighborsRegressor(n_neighbors=5)
    return apprentissage_methode(knn, X_train, X_test, y_train, y_test)

def kneighbors_regressor_standard(X_train, X_test, y_train, y_test):
    knn_standard = make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=5))
    return apprentissage_methode(knn_standard, X_train, X_test, y_train, y_test)

def kneighbors_regressor_minmax(X_train, X_test, y_train, y_test):
    knn_minmax = make_pipeline(MinMaxScaler(),KNeighborsRegressor(n_neighbors=5))
    return apprentissage_methode(knn_minmax, X_train, X_test, y_train, y_test)


def apprentissage(X_train, X_test, y_train, y_test, methode="TOUT"):

    scores = {}
    y_pred = {}

    # Regression linéaire
    if methode in ["TOUT", "LR"]:
        scores["LR"], y_pred["LR"] = linear_regression(X_train, X_test, y_train, y_test)

    if methode in ["TOUT", "LR Normal"]:
        scores["LR Normal"], y_pred["LR Normal"] = linear_regression_minmax(X_train, X_test, y_train, y_test)

    if methode in ["TOUT", "LR Standard"]:
        scores["LR Standard"], y_pred["LR Standard"] = linear_regression_standard(X_train, X_test, y_train, y_test)

    # Arbre de décision (Regression)
    if methode in ["TOUT", "DTR"]:
        scores["DTR"], y_pred["DTR"] = decision_tree_regressor(X_train, X_test, y_train, y_test)

    if methode in ["TOUT", "DTR Normal"]:
        scores["DTR Normal"], y_pred["DTR Normal"] = decision_tree_regressor_minmax(X_train, X_test, y_train, y_test)

    if methode in ["TOUT", "DTR Standard"]:
        scores["DTR Standard"], y_pred["DTR Standard"] = decision_tree_regressor_standard(X_train, X_test, y_train, y_test)

    # N Plus proches voisins (Regression)
    if methode in ["TOUT", "KNN"]:
        scores["KNN"], y_pred["KNN"] = kneighbors_regressor(X_train, X_test, y_train, y_test)

    if methode in ["TOUT", "KNN Normal"]:
        scores["KNN Normal"], y_pred["KNN Normal"] = kneighbors_regressor_minmax(X_train, X_test, y_train, y_test)

    if methode in ["TOUT", "KNN Standard"]:
        scores["KNN Standard"], y_pred["KNN Standard"] = kneighbors_regressor_standard(X_train, X_test, y_train, y_test)

    return scores, y_pred


def afficherTableau(methodes, scores):
    methodes.insert(0, 'Méthode')
    scores.insert(0, 'r²')
    for i in range(len(methodes)):
        print(f"{methodes[i]}{" "*(30-len(methodes[i]))} | {scores[i]}")

def afficher_graphique(y_pred, y_test, methode):
    plt.scatter(np.array(y_pred[methode]), np.array(y_test), marker='*', c='green')

    min_val = min(y_test.min(), y_pred[methode].min())
    max_val = max(y_test.max(), y_pred[methode].max())
    plt.plot([min_val, max_val], [min_val, max_val], c='red') # Diagonale

    plt.title(f"Méthode: {methode}")
    plt.xlabel('y_test')
    plt.ylabel('Estimation')
    plt.show()


# Reduction des composants PCA
def convertir_en_pca(X_train, X_test, nombre_composants):
    pca = PCA(n_components=nombre_composants)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca


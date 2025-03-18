from sklearn.model_selection import train_test_split
from DataCleaning import *
import pandas as pd 
from sklearn.metrics import r2_score
from sklearn import linear_model

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier

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

y_pred = []


 

#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=49, train_size=0.75,test_size=0.25)


#Reduction des composants 
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#Regression Linéaire :

lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_train_pca,y_train)
prediction=lin_reg.predict(X_test_pca)
score = lin_reg.score(X_test_pca,y_test)
#print(y_test.shape)
#print(prediction.shape)
print("Sans pretraitement + LR : ",score)
y_pred.append(prediction)


#StandardScaler + LinearRegression
scale_lr = make_pipeline(StandardScaler(),linear_model.LinearRegression())
scale_lr.fit(X_train,y_train)
scaled_pred = scale_lr.predict(X_test)
score_scaled = scale_lr.score(X_test,y_test)
print("StandardScaler + LR : ",score_scaled)
y_pred.append(scaled_pred)


#MinMaxScaler + LinearRegression
minmax_lr = make_pipeline(MinMaxScaler(),linear_model.LinearRegression())
minmax_lr.fit(X_train,y_train)
minmax_lr_pred = minmax_lr.predict(X_test)
score_minmax = minmax_lr.score(X_test,y_test)
print("MinMax + LR : ",score_minmax)
y_pred.append(minmax_lr_pred)
print("\n")

#Arbre de décision :

for i in range(4,6):

    #DecisionTreeRegressor
    decision_tr = DecisionTreeRegressor(max_depth=i)
    decision_tr.fit(X_train,y_train)
    pred_dtr = decision_tr.predict(X_test)
    score_dtr = decision_tr.score(X_test,y_test)
    print(f"AD {i} : ",score_dtr)


    #StandardScaler + DecisionTreeRegressor
    std_dtr = make_pipeline(StandardScaler(),DecisionTreeRegressor(max_depth=i))
    std_dtr.fit(X_train,y_train)
    std_dtr_pred = std_dtr.predict(X_test)
    score_std_dtr = std_dtr.score(X_test,y_test)
    print(f"StandardScaler + AD {i} : ",score_std_dtr)

    #MinMaxScaler + DecisionTreeRegressor
    minmax_dtr = make_pipeline(MinMaxScaler(),DecisionTreeRegressor(max_depth=i))
    minmax_dtr.fit(X_train,y_train)
    minmax_dtr_pred = minmax_dtr.predict(X_test)
    score_minmax_dtr = minmax_dtr.score(X_test,y_test)
    print(f"MinMaxScaler + AD {i} : ",score_minmax_dtr)
print("\n")

#N plus proches voisins :

for i in range(4,6):

    k_neigbors = KNeighborsClassifier(n_neighbors=i)
    k_neigbors.fit(X_train,y_train)
    k_neigbors_pred = k_neigbors.predict(X_test)
    score_kn = k_neigbors.score(X_test,y_test)
    print(f"KNeighbors {i} : " , score_kn)


    k_neigbors_std = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=i))
    k_neigbors_std.fit(X_train,y_train)
    k_neigbors_std_pred = k_neigbors_std.predict(X_test)
    k_neigbors_std_score = k_neigbors_std.score(X_test,y_test)
    print(f"StandardScaler + KNeighbors {i} : ",k_neigbors_std_score)

    k_neigbors_minmax = make_pipeline(MinMaxScaler(),KNeighborsClassifier(n_neighbors=i))
    k_neigbors_minmax.fit(X_train,y_train)
    k_neigbors_minmax_pred = k_neigbors_minmax.predict(X_test)
    k_neigbors_minmax_score = k_neigbors_minmax.score(X_test,y_test)
    print(f"MinMaxScaler + KNeighbors {i} : ",k_neigbors_minmax_score)

#A faire : Visualisation des résultats

# plt.plot(np.array(y_pred),np.array(y_test))
# plt.xlabel('y_test')
# plt.ylabel('prediction')
# plt.show()

   




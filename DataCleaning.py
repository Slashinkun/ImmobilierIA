from pandas import *
import numpy as np


# Remplace les NA d'une colonne par la moyenne
def clean_na_column(annonces, column):
    annonces[column] = annonces[column].replace("-", np.nan)
    annonces[column] = annonces[column].astype('float64')
    moyenne = annonces[column].mean(skipna=True)
    annonces[column] = annonces[column].fillna(moyenne)
    annonces[column] = annonces[column].astype('int64')

# Remplace les NA de toutes les colonnes (spécifié) par leur moyenne
def clean_na_all(annonces):
    columns = ["Nbr de pieces","Surface",'Nbr de chambres','Nbr de salle de bains']

    for col in columns:
        clean_na_column(annonces,col)

# Converti la colonne spécifié en colonne binaire
def splitMerge(annonces, column):
    dummies = get_dummies(annonces[column])
    annonces = annonces.merge(dummies, how="outer", left_index=True, right_index=True)
    return annonces.drop([column], axis=1)

# Converti toute les colonnes spécifiés en colonne binaire
def splitMergeAll(annonces):
    columns = ["DPE", "Type"]
    for col in columns: 
        annonces = splitMerge(annonces, col)
    return annonces

# Supprime/remplace les noms de villes incorrect 
def normalizeCitiesVille(villes):
    villes['city_code'] = villes['city_code'].str.replace("paris 0", "paris ")
    villes['city_code'] = villes['city_code'].drop_duplicates()

# Fait correspondre/Fusionne les villes de "annonces" avec les latitude, longitude des villes de "villes"
def mergeVille(annonces, villes):
    normalizeCitiesVille(villes)
    annonces = annonces.merge(villes[['city_code', 'latitude', 'longitude']], how="inner", left_on="Ville", right_on="city_code")
    annonces = annonces.drop(['Ville', 'city_code'], axis=1)
    return annonces

# Prépare les données scrapés pour être utilisé dans l'apprentissage ia
def preparer_donnees_ia(annonces, villes):
    annonces['DPE'] = annonces['DPE'].replace('-', 'Vierge')

    clean_na_all(annonces)
    annonces = splitMergeAll(annonces)
    annonces = mergeVille(annonces, villes)
    annonces = annonces.dropna()

    return annonces





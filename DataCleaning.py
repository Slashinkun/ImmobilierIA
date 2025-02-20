from pandas import *
import numpy as np


def clean_column(annonces, column):
    annonces[column] = annonces[column].replace("-", np.nan)
    annonces[column] = annonces[column].astype('float64')
    moyenne = annonces[column].mean(skipna=True)
    annonces[column] = annonces[column].fillna(moyenne)
    annonces[column] = annonces[column].astype('int64')


def clean_all(annonces):
    columns = ["Nbr de pieces","Surface",'Nbr de chambres','Nbr de salle de bains']

    for col in columns:
        clean_column(annonces,col)

def splitMerge(annonces, column):
    dummies = get_dummies(annonces[column])
    annonces = annonces.merge(dummies, how="outer", left_index=True, right_index=True)
    return annonces.drop([column], axis=1)

def splitMergeAll(annonces):
    columns = ["DPE", "Type"]
    for col in columns: 
        annonces = splitMerge(annonces, col)
    return annonces

def normalizeCitiesVille(villes):
    villes['city_code'] = villes['city_code'].str.replace("paris 0", "paris ")
    villes['city_code'] = villes['city_code'].drop_duplicates()


def mergeVille(annonces, villes):
    normalizeCitiesVille(villes)
    annonces = annonces.merge(villes[['city_code', 'latitude', 'longitude']], how="inner", left_on="Ville", right_on="city_code")
    # annonces = annonces.drop(['Ville', 'city_code'], axis=1)
    return annonces


annonces = read_csv('data.csv', encoding='latin1')
annonces['DPE'] = annonces['DPE'].replace('-', 'Vierge')

villes = read_csv('cities.csv')

clean_all(annonces)
annonces = splitMergeAll(annonces)

annonces = mergeVille(annonces, villes)
annonces = annonces.dropna()
print(annonces.to_string())



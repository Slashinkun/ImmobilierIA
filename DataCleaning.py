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

def formatVille(annonces):
    

    annonces['Ville'] = annonces['Ville'].str.lower()
    annonces['Ville'] = annonces['Ville'].str.replace('é|è', 'e', regex=True)
    annonces['Ville'] = annonces['Ville'].str.replace('eme|er', '', regex=True)
    annonces['Ville'] = annonces['Ville'].str.replace('î|ï', 'i', regex=True)
    annonces['Ville'] = annonces['Ville'].str.replace('-', ' ')
    annonces['Ville'] = annonces['Ville'].str.replace("'", ' ')
    annonces['Ville'] = annonces['Ville'].str.replace("saint","st")
    

def mergeVille(annonces, villes):
    annonces = annonces.merge(villes[['label', 'latitude', 'longitude']], how="inner", left_on="Ville", right_on="label")
    annonces = annonces.drop(['Ville'], axis=1)
    annonces = annonces.drop(['label'], axis=1)
    return annonces


annonces = read_csv('data.csv', encoding='iso8859_15')
annonces['DPE'] = annonces['DPE'].replace('-', 'Vierge')

villes = read_csv('cities.csv')
   
clean_all(annonces)
annonces = splitMergeAll(annonces)
formatVille(annonces)
annonces = mergeVille(annonces, villes)
annonces = annonces.dropna()
print(annonces.to_string())

# print(villes[['label', 'latitude', 'longitude']].to_string())


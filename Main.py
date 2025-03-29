# https://www.immo-entre-particuliers.com
# https://www.immo-entre-particuliers.com/annonces/france-ile-de-france/vente/ta-offer
# https://www.immo-entre-particuliers.com/annonce-yvelines-mantes-la-jolie/409301-studio-meuble

from DataScraper import *
from DataCleaning import *

# Partie 1: Data Scraping
#scrapLink("https://www.immo-entre-particuliers.com/annonces/france-ile-de-france/vente/ta-offer")

# Partie 2: Data Cleaning
annonces = read_csv('data.csv', encoding='latin1')
annonces['DPE'] = annonces['DPE'].replace('-', 'Vierge')
villes = read_csv('cities.csv')

clean_all(annonces)
annonces = splitMergeAll(annonces)
annonces = mergeVille(annonces, villes)
annonces = annonces.dropna()





#print(annonces.to_string())

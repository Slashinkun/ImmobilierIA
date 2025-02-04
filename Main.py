# https://www.immo-entre-particuliers.com
# https://www.immo-entre-particuliers.com/annonces/france-ile-de-france/vente/ta-offer
# https://www.immo-entre-particuliers.com/annonce-yvelines-mantes-la-jolie/409301-studio-meuble

from DataScraper import *

soup = getsoup("https://www.immo-entre-particuliers.com/annonce-yvelines-mantes-la-jolie/409301-studio-meuble")
print(prix(soup))
print(ville(soup))

print()

try:
    soup2 = getsoup("https://www.immo-entre-particuliers.com/annonce-paris-paris-1er/408780-hotel-a-vendre")
    print(prix(soup2))
except NonValide as e:
    print(e)

print()

soup3 = getsoup("https://www.immo-entre-particuliers.com/annonce-paris-paris-14eme/408922-rare-appartement-lumineux-et-calme-sur-coulee-verte-du-14ieme")
print(prix(soup3))
print(ville(soup3))
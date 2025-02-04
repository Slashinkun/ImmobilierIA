# https://www.immo-entre-particuliers.com
# https://www.immo-entre-particuliers.com/annonce-yvelines-mantes-la-jolie/409301-studio-meuble

from DataScraper import *

soup = getsoup("https://www.immo-entre-particuliers.com/annonce-yvelines-mantes-la-jolie/409301-studio-meuble")
print(prix(soup))

soup2 = getsoup("https://www.immo-entre-particuliers.com/annonce-paris-paris-1er/408780-hotel-a-vendre")
print(prix(soup2))
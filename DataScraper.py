from bs4 import BeautifulSoup
import requests
import csv

class NonValide(Exception):

    def __init__(self, raison):
        self.raison = raison

    def __str__(self):
        return f"Annonce non valide: {self.raison}"

def getsoup(url):
    rhtml = requests.get(url, auth=('user', 'pass')).text
    soup = BeautifulSoup(rhtml, 'html.parser')

    return soup

def prix(soup):

    try:
        prixTexte = soup.find('p', class_='product-price').text
    except AttributeError:
        raise NonValide("Prix non précisé")
    
    prix = prixTexte.replace(' €', '').replace(' ', '')

    if (int(prix) < 10000):
        raise NonValide("Prix < 10000")

    return prix

def ville(soup):

    villeTexte = soup.find('h2', class_='mt-0').text
    indexVille = villeTexte.rfind(', ') + 2
    ville = villeTexte[indexVille::]

    return ville
 
# Récupère le bloc parent où se trouve les caractéristiques
def __caracteristiques__(soup):
    blocCaracteres = soup.find(string="Caractéristiques :").parent.parent
    return blocCaracteres

def type(soup):
    blocCaracteres = __caracteristiques__(soup)

    try:
        type = blocCaracteres.find(string="Type").parent.next_sibling.text
    except AttributeError:
        raise NonValide("Pas de type précisé")

    if (type not in ['Maison', 'Appartement']):
        raise NonValide("Pas une maison ou un appartement")

    return type

def surface(soup):
    try:
        blocCaracteres = __caracteristiques__(soup)
        surface = blocCaracteres.find(string="Surface").parent.next_sibling.text.replace(" m²","")

        return surface
    except AttributeError:
        return "-"


def nbrpieces(soup):
    try:
        blocCaracteres = __caracteristiques__(soup)
        return blocCaracteres.find(string="Nb. de pièces").parent.next_sibling.text
    except AttributeError:
        return "-"

def nbrchambres(soup):
    try:
        blocCaracteres = __caracteristiques__(soup)
        return blocCaracteres.find(string="Nb. de chambres").parent.next_sibling.text
    except AttributeError:
        return "-"

def nbrsdb(soup):
    try:
        blocCaracteres = __caracteristiques__(soup)
        return blocCaracteres.find(string="Nb. de sales de bains").parent.next_sibling.text
    except AttributeError:
        return "-"
    
def dpe(soup):
    try:
        blocCaracteres = __caracteristiques__(soup)
        return blocCaracteres.find(string="Consommation d'énergie (DPE)").parent.next_sibling.text[0]
    except AttributeError:
        return "-"
    
def informations(soup):
    return f"{ville(soup)},{type(soup)},{surface(soup)},{nbrpieces(soup)},{nbrchambres(soup)},{nbrsdb(soup)},{dpe(soup)},{prix(soup)}"

# Récupère les informations de toute les pages d'une page de recherche et l'écrit dans un csv
def scrapLink(link):

    npage = 1    
    flagend = False
    headers = ["Ville","Type","Surface","Nbr de pieces","Nbr de chambres","Nbr de salle de bains","DPE","Prix"]

    with open('data.csv', 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        while (not flagend):
            print()
            print(f"Page: {npage}")
            print()
            soup = getsoup(link+f"/{npage}")
            __getInformationsOfLinks__(writer, soup)

            npage += 1

            if soup.find('li', class_="next disabled") != None:
                flagend = True

# Récupère les informations d'une page contenant une liste d'annonce
# fonction auxiliaire et privé de scrapLink: 
# écrit les informations récupérés dans le csv 
def __getInformationsOfLinks__(writer, soup):

    detailsContainers = soup.find_all('div', class_='product-details-container')

    for container in detailsContainers:
        clink = container.find('a')
        href = clink.get("href")
        if "https://www.immo-entre-particuliers.com" not in href:
            href = "https://www.immo-entre-particuliers.com" + href
        try:
            information = informations(getsoup(href)) 
            writer.writerow(information.split(','))

            print(information)
        except NonValide as e:
            print(e)

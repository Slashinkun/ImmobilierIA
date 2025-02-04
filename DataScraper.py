from bs4 import BeautifulSoup
import requests

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

    prixTexte = soup.find('p', class_='product-price').text
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

    type = blocCaracteres.find(string="Type").parent.next_sibling.text

    if (type not in ['Maison', 'Appartement']):
        raise NonValide("Pas une maison ou un appartement")

    return type

def surface(soup):
    blocCaracteres = __caracteristiques__(soup)

    surface = blocCaracteres.find(string="Surface").parent.next_sibling.text.replace(" m²","")

    return surface


def nbrpieces(soup):
    blocCaracteres = __caracteristiques__(soup)

    return blocCaracteres.find(string="Nb. de pièces").parent.next_sibling.text

def nbrchambres(soup):
    blocCaracteres = __caracteristiques__(soup)

    return blocCaracteres.find(string="Nb. de chambres").parent.next_sibling.text

def nbrsdb(soup):
    blocCaracteres = __caracteristiques__(soup)

    return blocCaracteres.find(string="Nb. de sales de bains").parent.next_sibling.text
    
def dpe(soup):
    blocCaracteres = __caracteristiques__(soup)

    return blocCaracteres.find(string="DPE").parent.next_sibling.text[0]
    
def informations(soup):
    try:
        return f"{ville(soup)},{type(soup)},{surface(soup)},{nbrpieces(soup)},{nbrchambres(soup)},{nbrsdb(soup)},{dpe(soup)},{prix(soup)}"
    except NonValide as e:
        raise e

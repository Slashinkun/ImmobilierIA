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
 
from bs4 import BeautifulSoup
import requests

class NonValide(Exception):

    def __str__():
        return "Annonce non valide"

def getsoup(url):
    rhtml = requests.get(url, auth=('user', 'pass')).text
    soup = BeautifulSoup(rhtml, 'html.parser')

    return soup


from bs4 import BeautifulSoup
import requests

def getsoup(url):
    rhtml = requests.get(url, auth=('user', 'pass')).text
    soup = BeautifulSoup(rhtml, 'html.parser')

    return soup
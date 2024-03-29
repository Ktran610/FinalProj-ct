import requests
from bs4 import BeautifulSoup
import urllib.parse

url = 'https://scs.duytan.edu.vn/'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

DEFAULT_TARGET_LINKS = ['body']
base_url = url
target_links = DEFAULT_TARGET_LINKS
for target in soup.find_all(target_links):
	for link in target.find_all('a'):
		print((urllib.parse.urljoin(base_url, link.get('href'))))

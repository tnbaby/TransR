import requests
from bs4 import BeautifulSoup

res = requests.get("https://www.wikidata.org/w/index.php?search=&search=beijing&title=Special%3ASearch&go=Go")
res.encoding = 'utf-8'
soup = BeautifulSoup(res.text,"html.parser")
#print soup.prettify()
for tag in soup.find_all("span", class_="wb-itemlink-id"):
	print tag.string
	break
res = requests.get("https://www.wikidata.org/wiki/Q956")
res.encoding = "utf-8"
soup = BeautifulSoup(res.text, "html.parser")
for tag in soup.find_all("span", class_="wikibase-descriptionview-text"):
	print tag.string

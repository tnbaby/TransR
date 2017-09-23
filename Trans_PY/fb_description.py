import requests
from bs4 import BeautifulSoup
import pickle
import numpy
import time

def calc_sum(list1, list2):
	s = 0
	for i in xrange(len(list1)):
		s += abs(list1[i] - list2[i])
#	for i in xrange(len(list1)):
#		s += (list1[i] * list2[i])
	return abs(s)
def parseline(line):
	vec = []
	for i in line.split("\t"):
		if i != "\n":
			vec += [float(i)]
        return vec

f = open("../FB15k/entity2id.txt","r")
data = f.readlines()
f.close()
entitylist = []
for i in data:
	str1, num = i.split("\t")
	entitylist += [str1]
f = open("../FB15k/f2w.nt","r")
data = f.readlines()
f.close()
fb2wiki = {}
wiki2fb = {}
for i in data:
	fb, wiki = i.split(" ")
	fb2wiki[fb] = wiki.rstrip()
	wiki2fb[wiki.rstrip()] = fb

#input search string
entity_description_list = {}
t1 = time.time()
for i in xrange(len(entitylist)):
    try:
        index = fb2wiki[entitylist[i]]
        res = requests.get("https://www.wikidata.org/wiki/"+index)
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text,"html.parser")
        for tag in soup.find_all("span", class_="wikibase-title-label"):
            name = tag.string
        for tag in soup.find_all("span", class_="wikibase-descriptionview-text"):
            desc = tag.string
        print name
        entity_description_list[name] = desc
    except KeyError, e:
        print e
        continue
    except Exception, e:
        print e
        time.sleep(2)
t2 = time.time()
print "total time:", (t2 - t1)
f = open("description.txt", "w")
for i in xrange(len(entitylist)):
    if entity_description_list.has_key(id2entity[i]):
        f.write(id2entity[i]+"\t"+entity_description_list[id2entity[i]]+"\n")
    else:
        f.write(id2entity[i]+"\t"+"no found in wikidata\n")
f.close()

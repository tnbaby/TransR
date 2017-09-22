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

f = open("TransH_entity2vec_label.txt","r")
data = f.readlines()
f.close()
label = []
for i in data:
	label += [int(i)]
print "label:",len(label)

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

clusters = {}
t_s = time.time()
#search on wikidata for the meaning of the index
for i in xrange(len(label)):
	start = time.time()
	try:
		index = fb2wiki[entitylist[i]]
		res = requests.get("https://www.wikidata.org/wiki/"+index)
		res.encoding = 'utf-8'
		soup = BeautifulSoup(res.text,"html.parser")
		for tag in soup.find_all("span", class_="wikibase-title-label"):
			name = tag.string
		for tag in soup.find_all("span", class_="wikibase-descriptionview-text"):	
			description = tag.string
		end = time.time()
		print "epoch:", i, "time:", str(end - start)+ "s", name	
		if clusters.has_key(label[i]):
			clusters[label[i]] += [str(name + "-->" + description)]
		else:
			clusters[label[i]] = [str(name + "-->" + description)]
	except KeyError:
		continue
	except Exception, e:
		print e
		time.sleep(2)
s_e = time.time()
print "total time:", str(s_e - s_t)+"s"
f = open("TransH_clusters.txt", "w")
for key, values in clusters.items():
	print key, values[0]
	f.write("%s: \n"%key)
	for i in values:
		f.write("%s\t"%i)
	f.write("\n")	 

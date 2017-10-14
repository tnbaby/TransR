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
s_s = time.time()
f = open('description.txt', 'r')
data = f.readlines()
f.close()
for i in data:
	index = int(i[0 : i.find('\t')])
	s = i[i.find('\t') + 1 : -1]
	if clusters.has_key(label[index]):
		clusters[label[index]] += [s]
	else:
		clusters[label[index]] = [s]
s_e = time.time()
print "total time:", str(s_e - s_s)+"s"
#f = open("clusters.txt", "w")
#pickle.dump(clusters, f)
#f.close()
for key, values in clusters.items():
	f = open(str("label"+str(key)+".txt"), "w")
	print key, values[0]
	for i in values:
		f.write(i+"\n")
	f.close()
#		try:
#			index = fb2wiki[i]
#			print index
#			res = requests.get("https://www.wikidata.org/wiki/" + index)
#			res.encoding = "utf-8"
#			soup = BeautifulSoup(res.text, "html.parser")
#			for tag in soup.find_all("span", class_="wikibase-title-label"):
#				name = tag.string
#			for tag in soup.find_all("span", class_="wikibase-descriptionview-text"):
#				description = tag.string
#			f.write("%s\t%s\n"%(name, description))
#		except KeyError, e:
#			print "exception: ", e
#		except Exception, e:
#			time.sleep(1)
#			print "url exception: ", e
#		finally:
#			print "key: ", key
#	f.close()

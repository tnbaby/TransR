import requests
from bs4 import BeautifulSoup
import pickle
import numpy

def calc_sum(list1, list2):
	s = 0
	for i in range(len(list1)):
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

f = open("entity2vec.bern","r")
data = f.readlines()
f.close()
entityvec = []
for i in data:
	entityvec += [parseline(i)]
print (len(entityvec), len(entityvec[0]))

f = open("relation2vec.bern", 'r')
data = f.readlines()
f.close()
relationvec = []
for i in data:
	relationvec += [parseline(i)]
print (len(relationvec), len(relationvec[0]))

f = open("../../FB15k/entity2id.txt","r")
data = f.readlines()
f.close()
entitylist = []
entity2id = {}
count = 0
for i in data:
	str1, num = i.split("\t")
	entitylist += [str1]
	entity2id[str1] = count
	count = count + 1

f = open('../../FB15k/relation2id.txt', 'r')
data = f.readlines()
f.close()
relationlist = []
relation2id = {}
count = 0
for i in data:
	str1, num = i.split('\t')
	relationlist += [str1]
	relation2id[str1] = count
	count = count + 1

f = open("../../FB15k/f2w.nt","r")
data = f.readlines()
f.close()
fb2wiki = {}
wiki2fb = {}
for i in data:
	fb, wiki = i.split(" ")
	fb2wiki[fb] = wiki.rstrip()
	wiki2fb[wiki.rstrip()] = fb

#input search string
inp_str = input("input a string:")
res = requests.get("https://www.wikidata.org/w/index.php?search=&search="+inp_str+"&title=Special%3ASearch&go=Go")
soup = BeautifulSoup(res.text,"html.parser")
for tag in soup.find_all("span", class_="wb-itemlink-id"):
	inp_str = tag.string
	print ("search first id: ", inp_str)
	break
for tag in soup.find_all("span",  class_="wb-itemlink-label"):
	print ("search first result: ",tag.string)
	break
#for ii in xrange(len(relationvec)):
	#tmp = raw_input("press any key to continue!")
#calcuate the sum and sort
k = entity2id[wiki2fb[inp_str[1:len(inp_str)-1]]]
candidate_list = {}
f = open("notfound","w")
ii = 37
print (relationlist[ii])
for i in range(len(entitylist)):
#	for ii in xrange(len(relationvec)):
	try:
		candidate_list[fb2wiki[entitylist[i]]] = calc_sum(numpy.add(entityvec[k], relationvec[ii]), entityvec[i])
	#	candidate_list[fb2wiki[entitylist[i]]] = calc_sum(entityvec[k], entityvec[i])
	except KeyError:
		f.write(entitylist[i]+"\n")
f.close()
result = {}
result = sorted(candidate_list.items(), key= lambda item:item[1])
count = 0
simlist = []
for i in result:
	simlist += [i[0]]
	print (i[0], i[1])
	count += 1
	if count > 20:
		break

#search on wikidata for the meaning of the index
for i in simlist:
	res = requests.get("https://www.wikidata.org/wiki/"+i)
	res.encoding = 'utf-8'
	soup = BeautifulSoup(res.text,"html.parser")
	for tag in soup.find_all("span", class_="wikibase-title-label"):
		print (tag.string,"-->",)
	for tag in soup.find_all("span", class_="wikibase-descriptionview-text"):	
		print (tag.string)	

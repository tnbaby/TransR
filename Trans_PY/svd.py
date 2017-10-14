from numpy import *
import random
import pickle

#u, sigma, vt = linalg.svd(data)

f = open("TransH_entity2vec.bern", "r")
entity_vec = pickle.load(f)
f.close()

print len(entity_vec), len(entity_vec[0])

f = open("../FB15k/entity2id.txt", 'r')
data = f.readlines()
f.close()
entitylist = []
entity2id = {}
for i in data:
	str1, ind = i.split('\t')
	entitylist += [str1.strip()]
	entity2id[str1] = int(ind)

f = open("../FB15k/relation2id.txt","r")
data = f.readlines()
f.close()
relationlist = []
relation2id = {}
relation_num = 0
for i in data:
	str1, ind = i.split("\t")
	relation2id[str1] = int(ind)
	relationlist += [str1.strip()]
	relation_num += 1

f = open("../FB15k/train.txt", "r")
data = f.readlines()
f.close()
relation_map = {}
for i in data:
	left, right, rel = i.split("\t")
	left = left.strip()
	right = right.strip()
	rel = rel.strip()
	if(entity2id.has_key(left) == False):
		print "miss entity:", left
	if(entity2id.has_key(right) == False):
		print "miss entity:", right
	if(relation2id.has_key(rel) == False):
		relation2id[rel] = relation_num
		relation_num = relation_num + 1
	if relation_map.has_key(rel):
		relation_map[rel] += [left, right]
	else:
		relation_map[rel] = [left, right]

for key, values in relation_map.items():
	if len(values) < 2:
		continue
	data = []
	for entity in values:
		data += [entity_vec[entity2id[entity]]]
	print len(data)
	mean = sum(data, axis=0)/len(data)
	C = []
	for i in xrange(len(data)):
		C += (transpose(mat(data[i]) - mat(mean))*(mat(data[i]) - mat(mean))/len(data)).tolist()
	u, sigma, vt = linalg.svd(C)
	print key
	print sigma[0]/sum(sigma[:]), sigma[1]/sum(sigma[:])
	#print sigma

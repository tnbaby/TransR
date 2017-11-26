import math
import pickle
from numpy import *

datapath = "../FB15k/"

entitylist = []
entity2id = {}
relationlist = []
relation2id = {}

entity_vec = []
relation_vec = []
L1_FLAG = True
HIT = 10

def calc_sum(e1, e2, rel):
	sum1 = (mat(entity_vec[e2]) - mat(entity_vec[e1]) - mat(relation_vec[rel])).getA()[0]
	if L1_FLAG:
		sum1 = sum(abs(sum1))
	else:
		sum1 = sum(sum1**2)
	return sum1

# perpare function
f = open(datapath+'entity2id.txt','r')
data = f.readlines()
f.close()
for i in data:
	str1, ind = i.split('\t')
	entity2id[str1.strip()] = int(ind)
	entitylist += [str1.strip()]

f = open(datapath+'relation2id.txt', 'r')
data = f.readlines()
f.close()
for i in data:
	str1, ind = i.split('\t')
	relation2id[str1.strip()] = int(ind)
	relationlist += [str1.strip()]

f = open("TransH_entity2vec.bern", "r")
entity_vec = pickle.load(f)
f.close()

f = open("TransH_relation2vec.bern", "r")
relation_vec = pickle.load(f)
f.close()

f = open(datapath+'test.txt', 'r')
data = f.readlines()
f.close()
relation_map = {}
for i in data:
	left, right, rel = i.split('\t')
	left = left.strip()
	right = right.strip()
	rel = rel.strip()
	if(entity2id.has_key(left) == False):
		print "miss entity:", left
	if(entity2id.has_key(right) == False):
		print "miss entity:", right
	if(relation2id.has_key(rel) == False):
		print "miss relationship:", rel
	if relation_map.has_key(rel):
		if relation_map[rel].has_key(left):
			relation_map[rel][left] += 1
		else:
		 	relation_map[rel][left] = 1
		if relation_map[rel].has_key(right):
			relation_map[rel][right] += 1
		else:
		  	relation_map[rel][right] = 1
	else:
		relation_map[rel] = {left : 1, right : 1}  	
for rel, values in relation_map.items():
		if len(values) < 10:
			continue
		mean_inner = []
		for left, cnt in values.items():
			right_list = {}
			for en in entitylist:
				right_list[en] = calc_sum(entity2id[left], entity2id[en], relation2id[rel])
			result = sorted(right_list.items(), key = lambda item:item[1])
			count = 0
			for i in xrange(HIT):
				if(relation_map[rel].has_key(result[i][0])):
						count += 1
			mean_inner += [count/10.0]
		print rel,'\n', len(values), mean(mean_inner)

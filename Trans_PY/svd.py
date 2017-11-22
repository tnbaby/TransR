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
#	entitylist += [str1.strip()]
	entity2id[str1] = int(ind)

#f = open("../FB15k/relation2id.txt","r")
#data = f.readlines()
#f.close()
#relationlist = []
#relation2id = {}
#relation_num = 0
#for i in data:
#	str1, ind = i.split("\t")
#	relation2id[str1] = int(ind)
#	relationlist += [str1.strip()]
#	relation_num += 1
#
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
#	if(relation2id.has_key(rel) == False):
#		relation2id[rel] = relation_num
#		relation_num = relation_num + 1
	if relation_map.has_key(rel):
		if relation_map[rel].has_key(left):
			relation_map[rel][left]  += 1
		else:
		 	relation_map[rel][left] = 1
		if relation_map[rel].has_key(right):
			relation_map[rel][right] += 1
		else:
		 	relation_map[rel][right] = 1
	else:
		relation_map[rel] = {left : 1, right : 1}

fb = open("PCA_bigdata.txt", 'w')
fm = open("PCA_meddata.txt", 'w')
fs = open("PCA_smadata.txt", 'w')

for key, values in relation_map.items():
	if len(values) < 10:
		continue
	elif len(values) < 100:
		f = fs
	elif len(values) < 1000:
		f = fm
	else:
		f = fb
	print len(values)
	data = []
	for entity, cnt in values.items():
		data += [entity_vec[entity2id[entity]]]
	mean = sum(data, axis=0)/len(data)
	C = zeros((100, 100))
	for i in xrange(len(data)):
		C = add(C, (transpose(mat(data[i]) - mat(mean))*(mat(data[i]) - mat(mean))/len(data)).tolist())
	u, sigma, vt = linalg.svd(data)
	f.write("\n"+key+"\t"+str(len(values))+"\t"+str(sum(sigma[0:10])/sum(sigma[:]))+"\t")
	for i in xrange(10):
		f.write(str(sigma[i]/sum(sigma[:]))+"\t")
 		print sigma[i]/sum(sigma[:]),
	print sum(sigma[0:10])/sum(sigma[:])
	#print sigma
fb.close()
fm.close()
fs.close()

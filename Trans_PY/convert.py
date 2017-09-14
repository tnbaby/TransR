import pickle

f = open("p_entity2vec.bern", "r")
entity_vec = pickle.load(f)
f.close()
f = open("p_relation2vec.bern", "r")
relation_vec = pickle.load(f)
f.close()
f = open("e.bern", "w")
for i in xrange(len(entity_vec)):
	for ii in xrange(100):
		f.write(str(entity_vec[i][ii])+"\n")
	f.write("\n")
f.close()
f = open("r.bern", "w")
for i in xrange(len(relation_vec)):
	for ii in xrange(100):
		f.write(str(relation_vec[i][ii])+"\n")
	f.write("\n")
f.close()

import random
from math import sqrt
from numpy import mat 
from numpy.linalg import norm
import pickle
import numpy
import datetime

datapath = "../FB15k/"

entity2id = {}
id2entity = {}
relation2id = {}
id2relation = {}
fb_h = []
fb_t = []
fb_r = []
entity_num = 0
relation_num = 0
relation_vec = []
A = []
entity_vec = []
ok = {}
left_candidate_ok = {}
right_candidate_ok = {}


dim = 100
rate = 0.01
margin = 1.0 
method = "bern"

L1_FLAG = True

def norm1(vec):
	k = norm(vec)
	if k > 1:
		vec /= k
	return vec
 
def normalize(miu, sigma, mimi, maxi):
	n = random.gauss(miu, sigma)
	while n < mimi or n > maxi:
		n = (maxi - mimi) * random.random() + random.gauss(miu, sigma)
	return n

def normA(a, A):
	while True:
		norm(A)
		x = sum(numpy.multiply(a, A))
		if x > 0.1:
			a -= rate*A
			A -= rate*a
		else:
			break
	norm(A)
	return a, A
		
def parseline(line):
	vec = []
	for i in line.split("\t"):
		if i != "\n":
			vec += [float(i)]
	return vec

def train_kb(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b):
	global res, count
	sum1 = calc_sum(e1_a, e2_a, rel_a)
	sum2 = calc_sum(e1_b, e2_b, rel_b)
	if sum1+margin>sum2:
		count += 1
		res += margin + sum1 - sum2
	#	s = datetime.datetime.now()
		gradient(e1_a, e2_a, rel_a, -1)
		gradient(e1_b, e2_b, rel_b, 1)
	#	e = datetime.datetime.now()
	#	print "gradient: ", (e - s).microseconds
def calc_sum(e1, e2, rel):
	tmp1 = sum(numpy.multiply(A[rel], entity_vec[e1]))
	tmp2 = sum(numpy.multiply(A[rel], entity_vec[e2]))
	sum1 = mat(entity_vec[e2]) - mat(tmp2*A[rel]) - (mat(entity_vec[e1])-mat(tmp1*A[rel])) - mat(relation_vec[rel])
	sum1 = sum1.getA()[0]
	if L1_FLAG:
	#	for i in xrange(dim):
	#		sum1 = sum1 + abs(entity_vec[e2][i]-entity_vec[e1][i]-relation_vec[rel][i])
		sum1 = sum(abs(sum1))
	else:
	#	for i in xrange(dim):	
	#		sum1 = sum1 + (entity_vec[e2][i]-entity_vec[e1][i]-relation_vec[rel][i])**2
		sum1 = sum(sum1**2)
	return sum1

def gradient(e1, e2, rel, belta):
#	for i in xrange(dim):
#		x = 2*(entity_vec[e2_a][i]-entity_vec[e1_a][i]-relation_vec[rel_a][i])
#		if L1_FLAG:
#			if x > 0:
#				x = 1
#			else:
#				x = -1
#		relation_tmp[rel_a][i] = relation_tmp[rel_a][i] + rate*x
#		entity_tmp[e1_a][i] = entity_tmp[e1_a][i] + rate*x
#		entity_tmp[e2_a][i] = entity_tmp[e2_a][i] - rate*x
#		x = 2*(entity_vec[e2_b][i]-entity_vec[e1_b][i]-relation_vec[rel_b][i])
#		if L1_FLAG:
#			if x > 0:
#				x = 1
#			else:
#				x = -1
#		relation_tmp[rel_b][i] = relation_tmp[rel_b][i] - rate*x
#		entity_tmp[e1_b][i] = entity_tmp[e1_b][i] - rate*x
#		entity_tmp[e2_b][i] = entity_tmp[e2_b][i] + rate*x
	global relation_tmp, entity_tmp, A_tmp
	tmp1 = sum(numpy.multiply(A[rel], entity_vec[e1]))
	tmp2 = sum(numpy.multiply(A[rel], entity_vec[e2]))
	x = 2*(mat(entity_vec[e2]) - mat(tmp2*A[rel]) - (mat(entity_vec[e1])-mat(tmp1*A[rel])) - mat(relation_vec[rel]))
	x = x.getA()[0]
	if L1_FLAG:
		x[x > 0] = 1
		x[x <= 0] = -1
	sum_x = sum(numpy.multiply(x,A[rel]))
	relation_tmp[rel] -= belta*rate*x
	entity_tmp[e1] -= belta*rate*x
	entity_tmp[e2] += belta*rate*x
	A_tmp[rel] += belta*rate*x*tmp1
	A_tmp[rel] = (mat(A_tmp[rel]) - mat(belta*rate*x*tmp2)).getA()[0]
	A_tmp[rel] += belta*rate*sum_x*entity_vec[e1]
	A_tmp[rel] = (mat(A_tmp[rel]) - mat(belta*rate*sum_x*entity_vec[e2])).getA()[0]
	
	relation_tmp[rel] = norm1(relation_tmp[rel])
	entity_tmp[e1] = norm1(entity_tmp[e1])
	entity_tmp[e2] = norm1(entity_tmp[e2])
	A_tmp[rel] /= norm(A_tmp[rel])
	relation_tmp[rel], A_tmp[rel] = normA(relation_tmp[rel], A_tmp[rel])

# perpare function
f = open(datapath+"entity2id.txt","r")
data = f.readlines()
f.close()
for i in data:
	str1, ind = i.split("\t")
	str1 = str1.strip()
	entity2id[str1] = int(ind)
	id2entity[int(ind)] = str1
	entity_num = entity_num + 1

f = open(datapath+"relation2id.txt","r")
data = f.readlines()
f.close()
for i in data:
	str1, ind = i.split("\t")
	str1 = str1.strip()
	relation2id[str1] = int(ind)
	id2relation[int(ind)] = str1
	relation_num = relation_num + 1
#left_entity = [[0 for i in xrange(entity_num)] for i in xrange(relation_num)]
#right_entity = [[0 for i in xrange(entity_num)] for i in xrange(relation_num)]
left_entity = {}
right_entity = {}
left_mean = [0 for i in xrange(relation_num)]
right_mean = [0 for i in xrange(relation_num)]
left_var = [0 for i in xrange(relation_num)]
right_var = [0 for i in xrange(relation_num)]
entity2num = [0 for i in xrange(entity_num)]

f = open(datapath+"train.txt", "r")
data = f.readlines()
f.close()
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
		print "miss relationship: ", rel
	if left_entity.has_key(relation2id[rel]) and left_entity[relation2id[rel]].has_key(entity2id[left]):
		left_entity[relation2id[rel]][entity2id[left]] = left_entity[relation2id[rel]][entity2id[left]] + 1
	elif left_entity.has_key(relation2id[rel]):
		left_entity[relation2id[rel]][entity2id[left]] = 1
	else:
		left_entity[relation2id[rel]] = {entity2id[left]:1}
	if right_entity.has_key(relation2id[rel]) and right_entity[relation2id[rel]].has_key(entity2id[right]):
		right_entity[relation2id[rel]][entity2id[right]] = right_entity[relation2id[rel]][entity2id[right]] + 1
	elif right_entity.has_key(relation2id[rel]):
		right_entity[relation2id[rel]][entity2id[right]] = 1
	else:
		right_entity[relation2id[rel]] = {entity2id[right]:1}
	fb_h += [entity2id[left]]
	fb_t += [entity2id[right]]
	fb_r += [relation2id[rel]]
	#ok = {entity2id[left]:{relation2id[rel]:{entity2id[right]:1}}}
	ok[(str(entity2id[left])+"@"+str(relation2id[rel])+"@"+str(entity2id[right]))] = 1
	left_candidate_ok[str(entity2id[left])+"@"+str(relation2id[rel])] = 1
	right_candidate_ok[str(entity2id[right])+"@"+str(relation2id[rel])] = 1
	entity2num[entity2id[left]] += 1
	entity2num[entity2id[right]] += 1
for key, value in left_entity.items():
	sum1 = 0.0
	sum2 = 0.0
	sum3 = 0.0
	for v in value.values():
		sum1 += 1
		sum2 += v
		sum3 += sqrt(v)
	left_mean[key] = 1.0*sum2/sum1
	left_var[key] = sum3/sum1 - left_mean[key]**2

for key, value in right_entity.items():
	sum1 = 0.0
	sum2 = 0.0
	sum3 = 0.0
	for v in value.values():
		sum1 += 1
		sum2 += v
		sum3 += sqrt(v)
	right_mean[key] = 1.0*sum2/sum1
	right_var[key] = sum3/sum1 - right_mean[key]**2

# run function
#entity_vec = mat([[random.uniform(-6.0/sqrt(dim), 6.0/sqrt(dim)) for i in xrange(dim)] for i in xrange(entity_num)])
#relation_vec = mat([[random.uniform(-6.0/sqrt(dim), 6.0/sqrt(dim)) for i in xrange(dim)] for i in xrange(relation_num)])
#entity_vec = [[random.uniform(-6.0/sqrt(dim), 6.0/sqrt(dim)) for i in xrange(dim)] for i in xrange(entity_num)]
#relation_vec = [[random.uniform(-6.0/sqrt(dim), 6.0/sqrt(dim)) for i in xrange(dim)] for i in xrange(relation_num)]
#entity_vec = [[random.gauss(0, 1.0/dim) for i in xrange(dim)] for i in xrange(entity_num)]
#relation_vec = [[random.gauss(0, 1.0/dim) for i in xrange(dim)] for i in xrange(relation_num)]
entity_vec = [[normalize(0,1.0/dim, -1, 1) for i in xrange(dim)] for i in xrange(entity_num)]
relation_vec = [[normalize(0,1.0/dim, -1, 1) for i in xrange(dim)] for i in xrange(relation_num)]
A = [[normalize(0, 1.0/dim, -1, 1) for i in xrange(dim)] for i in xrange(relation_num)]

for i in xrange(entity_num):
	entity_vec[i] = norm1(entity_vec[i])
	#entity_vec[i] = entity_vec[i] / len(entity_vec[i])
for i in xrange(relation_num):
	A[i] /= norm(A[i])

#bfgs function
res = 0
nbatches=50
neval=1000
batchsize=len(fb_h)/nbatches
cnt = 0
rate_flag = []

print "relation_num: ", relation_num, "entity_num: ", entity_num
A_tmp = A
relation_tmp = relation_vec
entity_tmp = entity_vec
for eval1 in xrange(neval):
	res = 0
	count = 0
	start = datetime.datetime.now()
	for batch in xrange(nbatches):
#		batch_start = datetime.datetime.now()
		for k in xrange(batchsize):
			i = (random.randint(0, len(fb_h)-1) * random.randint(0, len(fb_h)-1))%len(fb_h)
			j = (random.randint(0, entity_num-1) * random.randint(0, entity_num-1))%entity_num
			pr = right_mean[fb_r[i]]/(right_mean[fb_r[i]]+left_mean[fb_r[i]])
			if(method=="unif"):
				pr = 0.5 
			if(random.uniform(0, 1)<pr):
				while ok.has_key((str(fb_h[i])+"@"+str(fb_r[i])+"@"+str(j))):
					j = (random.randint(0, entity_num-1) * random.randint(0, entity_num-1))%entity_num
#				s = datetime.datetime.now()
				train_kb(fb_h[i], fb_t[i], fb_r[i], fb_h[i], j, fb_r[i])
#				e = datetime.datetime.now()
#				print "train_kb:", (e - s).microseconds
			else:
				while ok.has_key((str(j)+"@"+str(fb_r[i])+"@"+str(fb_t[i]))):
					j = (random.randint(0, entity_num-1) * random.randint(0, entity_num-1))%entity_num
				train_kb(fb_h[i], fb_t[i], fb_r[i], j, fb_t[i], fb_r[i])
	                relation_tmp[fb_r[i]] =  norm1(relation_tmp[fb_r[i]])
			entity_tmp[fb_h[i]] = norm1(entity_tmp[fb_h[i]])
			entity_tmp[fb_t[i]] = norm1(entity_tmp[fb_t[i]])
			entity_tmp[j] = norm1(entity_tmp[j])
			entity_tmp[fb_h[i]], A_tmp[fb_r[i]] = normA(entity_tmp[fb_h[i]], A_tmp[fb_r[i]])
			entity_tmp[fb_t[i]], A_tmp[fb_r[i]] = normA(entity_tmp[fb_t[i]], A_tmp[fb_r[i]])
			entity_tmp[j], A_tmp[fb_r[i]] = normA(entity_tmp[j], A_tmp[fb_r[i]])
		A = A_tmp
		entity_vec = entity_tmp
		relation_vec = relation_tmp
#		batch_end = datetime.datetime.now()	
#		print "batch ", batch, (batch_end - batch_start).microseconds
	end = datetime.datetime.now()             
	print "epoch:", eval1, res, "time: ", (end - start).seconds, " rate: ", rate, "gradient:", count
	if rate_flag:
		tmp = numpy.mean(rate_flag)
		if res < tmp and (tmp - res)/tmp > 0.01:
			rate_flag = []
		else:
			if cnt > 5:
				rate_flag = []
				rate = rate / 10.0
				if rate < 0.0001:
					rate = 0.001
				cnt = 0
				print "change rate: ", rate
			else:
				rate_flag += [res]
				cnt = cnt + 1
        else:
		rate_flag += [res]
	f1 = open("relation2vec."+method, "w")
	f2 = open("entity2vec."+method, "w")
	f3 = open("A."+method, "w")
	pickle.dump(relation_vec, f1)
	pickle.dump(entity_vec, f2)
	pickle.dump(A, f3)
	f1.close()
	f2.close()
	f3.close()


import numpy as np 

def accuracy(lis_of_tuples):
	n = len(lis_of_tuples)
	ctr = 0
	for i in lis_of_tuples:
		if i[0] == i[1]:
			ctr += 1

	accuracy_percentage = ctr/(1.0 * n)
	accuracy_percentage *= 100
	return accuracy_percentage
	
import numpy as np 

def load_data(filename = "dataset.csv"):

	data = np.loadtxt(filename, delimiter=',')    
	np.random.shuffle(data)

	data = data.astype(float)

	# maxx = np.max(data,axis=0)
	# minn = np.min(data,axis=0)
	# data = (data - minn)/(maxx-minn)

	n_rows = int(0.66*len(data))
	n_cols = len(data[0])
	rest_setx = data[:n_rows, :n_cols-1]
	rest_sety = data[:n_rows, -1]
	test_setx = data[n_rows:, :n_cols-1]
	test_sety = data[n_rows:, -1]
	return rest_setx, rest_sety.astype(int), test_setx, test_sety.astype(int)

def main():
	filename = input()
	load_data(filename)

if __name__=="__main__":
	main()

import numpy as np 

def load_data(filename):

	data = np.loadtxt(filename, delimiter=',')    
	np.random.shuffle(data)

	data = data.astype(float)
	traindata = data

	n_rows = len(data)
	n_cols = len(data[0])

	rest_setx = data[:n_rows, :n_cols-1]
	rest_sety = data[:n_rows, n_cols-1:]
	test_setx = data[n_rows:, :n_cols-1]
	test_sety = data[n_rows:, n_cols-1:]
  
	#return (rest_setx, rest_sety, test_setx, test_sety)

def main():
	filename = input()
	load_data(filename)

if __name__=="__main__":
	main()

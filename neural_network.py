import numpy as np
from neural_network_functions import Neural_Network

if __name__ == "__main__":
	# init data
	X = np.array(([3,5], [5,1], [10,2]), dtype=float)
	Y = np.array(([75], [82], [93]), dtype=float)
	X = X/np.amax(X, axis=0)
	Y = Y/100 # Max test score is 100
	NN = Neural_Network()
	
	yHat = NN.forward(X)
	cost1 = NN.costFunction(X, Y)
	print(yHat)
	print(cost1)
	
	print()
	for i in range(100000):
		NN.reduceCost(X, Y)
	
	yHat2 = NN.forward(X)
	cost2 = NN.costFunction(X, Y)
	print(yHat2)
	print(cost2)

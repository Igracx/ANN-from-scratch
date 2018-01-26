import numpy as np

class Neural_Network():
	def __init__(self):
		# Define HyperParameters
		self.inputLayerSize = 2
		self.hiddenLayerSize = 3
		self.outputLayerSize = 1
		
		# Weights (Parameters)
		self.W1 = np.random.randn(self.inputLayerSize, 
								  self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize,
								  self.outputLayerSize)
	
	def forward(self, X):
		# Propagate inputs through network
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat

	def sigmoid(self, z):
		# Apply sigmoid activation function
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self, z):
		# Derivative of Sigmoid Function
		return np.exp(-z)/((1+np.exp(-z)**2))
	
	def costFunctionPrime(self, X, Y):
		# Compute derivative with respect ot W1 and W2
		self.yHat = self.forward(X)
		
		delta3 = np.multiply(-(Y-self.yHat), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)
		
		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)
		
		return dJdW1, dJdW2
	
	def reduceCost(self, X, Y):
		dJdW1, dJdW2 = self.costFunctionPrime(X,Y)
		self.W1 = self.W1 - 0.1*dJdW1
		self.W2 = self.W2 - 0.1*dJdW2
	
	def costFunction(self, X, Y):
		yHat = self.forward(X)
		cost = sum(0.5*(Y - yHat)**2)
		return cost

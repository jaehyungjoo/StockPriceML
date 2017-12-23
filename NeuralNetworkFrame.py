import numpy as np
import cornell

# x = np.array(([3,5],[5,1],[10,2]), dtype=float)
# y = np.array(([75],[82],[93]), dtype=float)


class Neural_Network(object):
	'''
	
	Attributes:
		inputLayerSize: [int]
		hiddenLayerSize: [int]
		outputLayerSize: [int]
		w1: weights between input and hidden layers [numpy.ndarray]
		w2: weights between hidden and output layers [numpy.ndarray]
	'''
	#getters setters    
	def getParams(self):
		"""
		returns: w1 and w2 flattened to 1-d vector
		"""
		params = np.append(self.w1.ravel(), self.w2.ravel())
		return params

	def setParams(self, params):
		"""
		sets w1 and w2 to the values in params.
		
		Parameter params: 1-d vector that contains values to set in w1 and w2
		Precondition: [numpy.ndarray]
		"""
		assert type(params) == np.ndarray, "params is not an array"
	
		w1_start = 0
		w1_end = self.inputLayerSize*self.hiddenLayerSize
		self.w1 = np.reshape(params[w1_start:w1_end], (self.inputLayerSize, self.hiddenLayerSize))
	
		w2_end = w1_end + self.hiddenLayerSize*self.outputLayerSize
		self.w2 = np.reshape(params[w1_end:w2_end], (self.hiddenLayerSize, self.outputLayerSize))
		
		
	def __init__(self, inputLayerSize=2, hiddenLayerSize=3, outputLayerSize=1):
		"""
		Initialises neural network. 
		
		Parameter inputLayerSize: number of nodes in input layer.
		Precondition: [int] >= 1
		
		Parameter hiddenLayerSize: number of nodes in hidden layer.
		Precondition: [int] >= 1
		
		Parameter outputLayerSize: number of nodes in output layer.
		Precondition: [int] >= 1
		"""
		assert type(inputLayerSize) == int and type(hiddenLayerSize) == int and type(outputLayerSize) == int
		assert inputLayerSize >= 1 and hiddenLayerSize >= 1 and outputLayerSize >= 1
		
		#define hyperparameters
		self.inputLayerSize = inputLayerSize
		self.hiddenLayerSize = hiddenLayerSize
		self.outputLayerSize = outputLayerSize

		#initialise weight arrays with sudo-random numbers (parameters)
		self.w1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.w2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
        
	def forward(self, x):
		'''
		returns: output of neural network a3 (yHat)
		propagates inputs through network
	
		Parameter x: vector of inputs 
		Precondition: [numpy.ndarray]
		'''
		self.z2 = np.dot(x, self.w1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.w2)
		yHat = self.sigmoid(self.z3)
		return yHat
    
    
	def sigmoid(self,z):
		'''
		returns: float f(z) where f is the sigmoid activation function
	
		Parameter z: value(s) to apply sigmoid function 
		Precondition: [int] or [float] or [numpy.ndarray]
		'''
		return 1/(1+np.exp(z*-1))
    
    
	def sigmoidPrime(self,z):
		'''
		returns: sigmoid prime of z
	
		Paramter z: value(s) to apply prime of sigmoid function [int][float][numpy.ndarray]
		'''
		return np.exp(-1*z)/((1+np.exp(-1*z))**2)
    
    
	def costFunction(self, x, y):
		'''
		returns: value of cost at current weights of input x and output y
	
		parameter x - vector of input values [numpy.ndarray]
		parameter y - vector of output values [numpy.ndarray]
		'''
		self.yHat = self.forward(x)
	
		error = np.subtract(y, self.yHat)
		costs = 0.5*(error)**2
		costs = np.ravel(costs)
		costs_sum = np.sum(costs)
		return costs_sum
	

	def costFunctionPrime(self, x, y):
		#returns: two vectors containing derivative of cost function with respect to w1 and w2
		self.yHat = self.forward(x)
	
		delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))	#element multiplication
		dJdw2 = np.dot(self.a2.T, delta3)		#matrix dot product
	
		delta2 = np.dot(delta3, self.w2.T)*self.sigmoidPrime(self.z2)
		dJdw1 = np.dot(x.T, delta2)
	
		return dJdw1, dJdw2


	def computeGradients(self, x, y):
		#returns: ravelled 1 row vector of all costFunctionPrime values
		dJdw1, dJdw2 = self.costFunctionPrime(x, y)
		return np.append(dJdw1.ravel(), dJdw2.ravel())
    

def computeNumericGradients(N, x, y):
	'''
	returns: 
	
	parameter N - the neural network object [Neural_Network]
	parameter x - vector of inputs [numpy.ndarray]
	parameter y - vector of outputs [numpy.ndarray]
	'''
	assert isinstance(N,Neural_Network), repr(N) + "is not an instance of Neural_Network"
	assert type(x) == np.ndarray, repr(x) + "is not a numpy.ndarray"
	assert type(y) == np.ndarray, repr(y) + "is not a numpy.ndarray"
	
	paramsInitial = N.getParams()
	numgrad = np.zeros(paramsInitial.shape)
	perturb = np.zeros(paramsInitial.shape)
	e = 1e-4
	
	for pos in range(len(paramsInitial)):
		perturb[pos] = e
		N.setParams(paramsInitial + perturb)
		loss2 = N.costFunction(x,y)
		
		N.setParams(paramsInitial - perturb)
		loss1 = N.costFunction(x,y)
		
		numgrad[pos] = (loss2 - loss1) / (2*e)
		
		perturb[pos] = 0
	
	N.setParams(paramsInitial)
	
	return numgrad




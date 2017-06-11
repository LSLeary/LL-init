import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams()
theano.config.floatX = 'float32'
theano.config.intX = 'int32'
import numpy as np

from neuralNetwork import Representation
from regularisation import regFD
from initialisations import initD
from utilityFunctions import (getWithName, sharedVar, prod, randomReflect,
		randomTranslate, sharedSeg)

# Layers {{{

class Layer(object): # {{{

	def __init__(self, outputShape=None, initialisation=None, # {{{
			initKWArgs={}, reg=0., regFunction=None, trainable=True):
		self.outputShape = outputShape
		self.trainable = trainable
		self.trainableWeights = []
		self.weights = []
		self.reg = reg
		self.regFunction = getWithName(regFD, regFunction)
		self.regWeights = []
		self.manualUpdates = []
		self.initialisation = getWithName(initD, initialisation)
		self.initKWArgs = initKWArgs.copy()
		self.built = False
		self.debugCompiled = False
		self.depth = 0
		self.numWeights = 0
		self.name = "Layer"
		self.description= "A layer." # }}}

	def build(self, rep): # {{{
		self.outputShape = self.inputShape

	def _build(self, rep):
		self.inputShape = rep.shape
		self.build(rep)
		self.built = True # }}}

	def call(self, x): # {{{
		return x

	def testCall(self, x):
		return self.call(x)

	def __call__(self, rep):
		if not self.built:
			self._build(rep)
		nextRep = Representation(
			shape = self.outputShape
		,	layers = rep.layers + [self]
		,	trainTensor = self.call(rep.trainTensor)
		,	testTensor = self.testCall(rep.testTensor)
		)
		return nextRep # }}}

	def compileDebugCall(self): # {{{
		numDim = len(self.inputShape)+1
		z = T.TensorType(
			dtype=theano.config.floatX
		,	broadcastable=(False,)*numDim
		)()
		y = self.call(z)
		self._debugCall = theano.function([z], y)
		self.debugCompiled = True

	def debugCall(self, x):
		if self.debugCompiled:
			return self._debugCall(x)
		else:
			self.compileDebugCall()
			return self._debugCall(x) # }}}

	def saveWeights(self, f): # {{{
		if self.weights == []:
			pass
		else:
			f.create_group(self.name)
			for i, W in enumerate(self.weights):
				f[self.name].create_dataset(str(i),
						data=self.weights[i].get_value())

	def loadWeights(self, f):
		for i, W in enumerate(self.weights):
			self.weights[i].set_value(np.array(f[self.name][str(i)])) # }}}

	def reinit(self): # {{{
		if self.initialisation is None:
			return
		newWs = self.initialisation(self.inputShape,
				self.outputShape, **self.initKWArgs)
		for oldW, newW in zip(self.weights, newWs):
			oldW.set_value(newW.get_value()) # }}}

	def show(self, arc): # {{{
		arc.log("")
		arc.log(self.description)
		if self.inputShape != self.outputShape:
			arc.log(str(self.inputShape) + " ---> " + str(self.outputShape))
	# }}}

# }}}

class FC(Layer): # {{{

	def __init__(self, size, activation='linear', **kwargs):
		Layer.__init__(self, outputShape=(size,), **kwargs)
		self.activation = activation
		self.name = "FC"
		self.description = "A Fully Connected layer with "+str(size)+" neurons."
		self.depth = 1

	def build(self, rep):
		self.weights = self.initialisation(
				self.inputShape, self.outputShape, **self.initKWArgs)
		self.numWeights = (self.inputShape[0] + 1)*self.outputShape[0]
		if self.trainable:
			self.trainableWeights = self.weights
		else:
			self.trainableWeights = []
		if self.regFunction is not None:
			self.regWeights = [self.weights[0]]

	def call(self, x):
		linPart = T.dot(x, self.weights[0]) + self.weights[1]
		if self.activation == 'linear':
			return linPart
		elif self.activation == 'relu':
			return T.maximum(linPart, 0.)
		else:
			return self.activation(linPart) # }}}

class Activation(Layer): # {{{

	def __init__(self, f):
		Layer.__init__(self, trainable=False)
		if type(f) is str:
			self.description = "A " + f + " activation layer."
			if f == 'linear':
				self.f = id
			elif f == 'relu' or f == 'ReLU':
				self.f = lambda x: T.maximum(x, 0.)
			else:
				raise Exception("The Activation argument was not understood.")
		else:
			self.description = "An activation layer."
			self.f = f

	def call(self, x):
		return self.f(x) # }}}

class ParReLU(Layer): # {{{
	# PReLU but with more params. In piecewise notation:
	# f(x) = { a(x-c), if x < c
	#        { b(x-c), else
	# Where a, b and c are all trainable.

	def __init__(self, a=1., b=1., c=0.,
			aReg=0., regFunction=None, trainable=True):
		Layer.__init__(self, None, reg=aReg,
				regFunction=regFunction, trainable=trainable)
		self.tuple = (a, b, c)
		self.name = "ParReLU"
		self.description = "A Kappa/ParReLU layer."

	def build(self, rep):
		self.outputShape = self.inputShape
		self.a = sharedVar(self.tuple[0] * np.ones(self.inputShape))
		self.b = sharedVar(self.tuple[1] * np.ones(self.inputShape))
		self.c = sharedVar(self.tuple[2] * np.ones(self.inputShape))
		self.numWeights = 3*prod(self.inputShape)
		self.weights = [self.a, self.b, self.c]
		if self.trainable:
			self.trainableWeights = self.weights
		else:
			self.trainableWeights = []
		if self.regFunction is not None:
			self.regWeights = [self.a]

	def call(self, x):
		xCen = x - self.c
		left = T.minimum(xCen, 0.)
		right = T.maximum(xCen, 0.)
		return self.a * left + self.b * right # }}}

class Flatten(Layer): # {{{

	def __init__(self):
		Layer.__init__(self, trainable=False)
		self.description = "A Flattening layer."

	def build(self, rep):
		self.outputShape = (prod(self.inputShape),)

	def call(self, x):
		return T.reshape(x, (x.shape[0], -1)) # }}}

class Convolution(Layer): # {{{

	def __init__(self, numFilters, s=(3, 3), # {{{
			strides=(1, 1), bias=True, border_mode=None, **kwargs):
		Layer.__init__(self, **kwargs)
		self.numFilters = numFilters
		self.s = s
		self.strides = strides
		if self.strides == (2, 2):
			self.initKWArgs['pooling'] = True
		self.bias = bias
		self.depth = 1
		if border_mode in [None, 'half']:
			self.border_mode = ((s[0] - 1)/2, (s[1] - 1)/2)
		else:
			self.border_mode = border_mode
		self.initKWArgs['s'] = self.s
		self.name = "Convolution"
		self.description = ("A " + str(s[0]) + "x" + str(s[1])
				+ " Convolutional layer with " + str(numFilters) + " filters.")
		# }}}

	def build(self, rep): # {{{
		sfun = (lambda i: (self.inputShape[i + 1]
				+ 2*self.border_mode[i] - self.s[i] + 1)/self.strides[i])
		self.outputShape = (self.numFilters, sfun(0), sfun(1))

		weights = self.initialisation(self.inputShape,
				self.outputShape, **self.initKWArgs)
		self.kernel = weights[0]
		self.numWeights = self.numFilters * self.inputShape[0] * prod(self.s)
		if self.bias:
			self.b = weights[1]
			self.weights = [self.kernel, self.b]
			self.numWeights += self.outputShape[0]
		else:
			self.weights = [self.kernel]
		if self.trainable:
			self.trainableWeights = self.weights
		if self.regFunction is not None:
			self.regWeights = [self.kernel] # }}}

	def call(self, x): # {{{
		con = T.nnet.conv2d(
			x, self.kernel, input_shape=(
				None, self.inputShape[0], self.inputShape[1], self.inputShape[2]
			), filter_shape=(
				self.numFilters, self.inputShape[0], self.s[0], self.s[1]
			), border_mode=self.border_mode, subsample=self.strides
		)
		if self.bias:
			return con + self.b
		else:
			return con # }}}

# }}}

class Pooling(Layer): # {{{

	def __init__(self, mode='max', block=(2, 2), stride=None): # {{{
		Layer.__init__(self, trainable=False)
		if mode in ['avg', 'average', 'average_exc_pad']:
			self.mode = 'average_exc_pad'
			mode = 'average'
		else:
			self.mode = mode
		self.block = block
		if stride is None:
			self.stride = block
			strToken = "."
		else:
			self.stride = stride
			strToken = (" with strides of "
					+ str(self.stride[0]) + "x" + str(self.stride[1])) + "."

		if mode is 'global_avg':
			dimToken = ""
		else:
			dimToken = str(block[0]) + "x" + str(block[1]) + " "
		self.description = "A " + dimToken + mode + " pooling layer" + strToken
		# }}}

	def build(self, rep): # {{{
		s = list(self.inputShape)
		if self.mode is 'global_avg':
			self.block = (s[-2], s[-1])
			self.outputShape = tuple(s[:-2])
		else:
			sfun = lambda i: (s[i-2] - self.block[i]) / self.stride[i] + 1
			s[-2] = sfun(0)
			s[-1] = sfun(1)
			self.outputShape = tuple(s) # }}}

	def call(self, x): # {{{
		if self.mode == 'global_avg':
			return T.mean(x, axis=(2, 3))
		return T.signal.pool.pool_2d(x, ws=self.block,
				stride=self.stride, ignore_border=False, mode=self.mode) # }}}

# }}}

class Dropout(Layer): # {{{

	def __init__(self, p=0.5):
		Layer.__init__(self, trainable=False)
		self.p = p
		self.description = "A Dropout " + str(p) + " layer."

	def testCall(self, x):
		return x

	def call(self, x):
		if self.p in [0., 0]:
			return x
		else:
			# The square root preserves variance.
			scale = 1./np.sqrt(1.-self.p)
			mask = srng.choice(size=self.outputShape, a=[0., scale],
					p=[self.p, 1.-self.p], dtype=theano.config.floatX)
			return x * mask # }}}

class BatchNormalisation(Layer): # {{{

	def __init__(self, beta=0., gamma=1., mu=0.9, trainable=True): # {{{
		Layer.__init__(self, trainable=trainable)
		self.tuple = (beta, gamma)
		self.mu = sharedVar(mu)
		self.name = "BatchNormalisation"
		self.description = "A Batch Normalisation layer." # }}}

	def build(self, rep): # {{{
		self.outputShape = self.inputShape
		self.numWeights = 2*prod(self.inputShape)

		self.rM = sharedVar(np.zeros(self.inputShape))
		self.rV = sharedVar(np.ones(self.inputShape))

		self.beta = sharedVar(self.tuple[0] * np.ones(self.inputShape))
		self.gamma = sharedVar(self.tuple[1] * np.ones(self.inputShape))
		self.weights = [self.beta, self.gamma, self.rM, self.rV]

		if self.trainable:
			self.trainableWeights = [self.beta, self.gamma]
		else:
			self.trainableWeights = [] # }}}

	def testCall(self, x): # {{{
		out = T.nnet.bn.batch_normalization_test(x,
				self.gamma, self.beta, mean=self.rM, var=self.rV)
		return out

	def call(self, x):
		out, _, _, newRM, newRV = T.nnet.bn.batch_normalization_train(x,
				self.gamma, self.beta, running_average_factor=(1-self.mu),
				running_mean=self.rM, running_var=self.rV)

		if not self.manualUpdates:
			self.manualUpdates = [(self.rM, newRM), (self.rV, newRV)]
		return out # }}}

# }}}

class CReLU(Layer): # {{{

	def __init__(self):
		Layer.__init__(self, trainable=False)
		self.description = "A CReLU layer."

	def build(self, rep):
		s = list(self.inputShape)
		s[0] *= 2
		self.outputShape = tuple(s)

	def call(self, x):
		xp = T.maximum(x, 0.)
		xn = -T.minimum(x, 0.)
		xout = T.concatenate([xp, xn], axis=1)
		return xout # }}}

class Augmentation(Layer): # {{{

	def __init__(self, translate=4, hFlip=True, vFlip=False):
		Layer.__init__(self, trainable=False)
		self.translate = translate
		self.hFlip = hFlip
		self.vFlip = vFlip
		if hFlip and vFlip:
			flipToken = ", flipping both vertically and horizontally."
		elif hFlip:
			flipToken = ", flipping horizontally."
		elif vFlip:
			flipToken = ", flipping vertically."
		else:
			flipToken = "."
		self.description = ("A Data Augmentation layer translating up to "
				+ str(translate) + " pixels" + flipToken)

	def testCall(self, x):
		return x

	def call(self, x):
		x = randomReflect(x, self.vFlip, self.hFlip)
		x = randomTranslate(x, self.translate)
		return x # }}}

class Observation(Layer): # {{{

	def __init__(self):
		Layer.__init__(self, trainable=False)
		self.name = "Observation"
		self.description = "An Observation layer."
		self.f = id
		self.predecessors = []

	def build(self, rep):
		self.outputShape = self.inputShape
		self.tensor = rep.trainTensor

	def setPredecessors(self, predecessors):
		self.predecessors = predecessors

	def setF(self, f):
		self.f = f # }}}

# Pseudolayers {{{

def merge(rep1, rep2, f=lambda x, y: 0.5*(x + y), shapef=lambda x, y: x):
	sha, fir, sec = sharedSeg(rep1.layers, rep2.layers)
	l = sha + fir + sec
	rep3 = Representation(
		shapef(rep1.shape, rep2.shape)
	,	layers=l
	,	trainTensor = f(rep1.trainTensor, rep2.trainTensor)
	,	testTensor = f(rep1.testTensor, rep2.testTensor)
	)
	return rep3 # }}}

# }}}


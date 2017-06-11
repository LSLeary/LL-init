import theano.tensor as T

from utilityFunctions import sharedVar, cast, zeroClone, zeroShared

# Optimisers {{{

class Optimiser(object): # {{{

	def __init__(self, loss=None, learningRate=None):
		self.refreshables = []
		self.dict = {}
		self.compiled = False
		self.loss = None
		self.learningRate = None
		self.compile(loss=loss, learningRate=learningRate)
		self.name = "Optimiser"

	def compile(self, loss=None, learningRate=None):
		if loss is not None:
			self.loss = loss
		if learningRate is not None:
			if self.learningRate is not None:
				self.learningRate.set_value(cast(learningRate))
			else:
				self.learningRate = sharedVar(learningRate)

		if self.loss is not None and self.learningRate is not None:
			self.compiled = True

	def call(self, weights):
		pass

	def __call__(self, weights):
		if self.compiled:
			return self.call(weights)
		else:
			self.compile()
			if self.compiled:
				return self.call(weights)
			else:
				raise Exception("The Optimiser must be compiled before use.")

	def refresh(self, lr=None):
		if lr is not None:
			if self.learningRate is not None:
				self.learningRate.set_value(cast(lr))
			else:
				self.learningRate = sharedVar(lr)
		for svar in self.refreshables:
			zeroShared(svar)

	def getLearningRate(self):
		return self.learningRate.get_value() # }}}

class SGD(Optimiser): # {{{

	def __init__(self, loss=None, momentum='none', mu=0.9, learningRate=None):
		Optimiser.__init__(self, loss=loss, learningRate=learningRate)
		if momentum in ['plain', 'nesterov', 'none']:
			self.momentum = momentum
		else:
			raise Exception(
				"Momentum argument not understood; use plain, nesterov, or none."
			)
		self.mu = sharedVar(mu)
		self.name = "SGD"

	def call(self, weights):
		grads = T.grad(self.loss, weights)
		if self.momentum == 'none':
			tups = zip(weights, grads)
			newW = lambda w, g: w - self.learningRate * g
			updates = [(w, newW(w, g)) for w, g in tups]
		else:
			us = [zeroClone(w) for w in weights]
			self.refreshables.extend(us)
			tups = zip(weights, us, grads)
			newU = lambda u, g: self.mu * u + g
			uUpdates = [(u, newU(u, g)) for _, u, g in tups]
			if self.momentum == 'plain':
				newW = lambda w, u, g: w - self.learningRate * newU(u, g)
			elif self.momentum == 'nesterov':
				newW = (lambda w, u, g:
					w - self.learningRate*((1. + self.mu)*g + self.mu**2 * u))
			wUpdates = [(w, newW(w, u, g)) for w, u, g in tups]
			updates = wUpdates + uUpdates
		return updates # }}}

class Adam(Optimiser): # {{{

	def __init__(self, loss=None, beta1=0.9, beta2=0.999,
			learningRate=0.001, momentum='normed'):
		Optimiser.__init__(self, loss=loss, learningRate=learningRate)
		assert momentum in ['normed', 'plain', 'nesterov']
		# Normed is classic Adam as per the paper.
		# Plain and Nesterov are momentum as usually used with SGD.
		self.momentum = momentum
		self.beta1 = beta1
		self.beta2 = beta2
		self.beta1BC = sharedVar(1./(1.-self.beta1))
		self.beta2BC = sharedVar(1./(1.-self.beta2))
		self.name = "Adam"

	def refresh(self, lr=None):
		Optimiser.refresh(self, lr=lr)
		self.beta1BC.set_value(1./(1.-self.beta1))
		self.beta2BC.set_value(1./(1.-self.beta2))

	def call(self, weights):
		ms = [zeroClone(w) for w in weights]
		self.refreshables.extend(ms)

		vs = [zeroClone(w) for w in weights]
		self.refreshables.extend(vs)

		grads = T.grad(self.loss, weights)
		tups = zip(weights, ms, vs, grads)

		mHat = lambda m: self.beta1BC * m
		vHat = lambda v: self.beta2BC * v

		newV = lambda v, g: self.beta2 * v + (1. - self.beta2) * g ** 2

		if self.momentum == 'normed':
			newM = lambda m, g: self.beta1 * m + (1. - self.beta1) * g
		else:
			newM = lambda m, g: self.beta1 * m + g

		if self.momentum in ['normed', 'plain']:
			newW = (lambda w, m, v, g: w - self.learningRate*mHat(newM(m, g))
					/ (T.sqrt(vHat(newV(v, g))) + 10.**(-6)))
		else:
			newW = (lambda w, m, v, g: w - self.learningRate*((1. + self.beta1)
					* g + self.beta1**2 * mHat(m))
					/ (T.sqrt(vHat(newV(v, g))) + 10.**(-6)))

		bCUp = lambda bc, beta: 1./(1.-(1.-1./bc)*beta)

		mUpdates = [(m, newM(m, g)) for _, m, _, g in tups]
		vUpdates = [(v, newV(v, g)) for _, _, v, g in tups]
		wUpdates = [(w, newW(w, m, v, g)) for w, m, v, g in tups]
		bcUpdates = [
			(self.beta1BC, bCUp(self.beta1BC, self.beta1))
		,	(self.beta2BC, bCUp(self.beta2BC, self.beta2))
		]
		return wUpdates + mUpdates + vUpdates + bcUpdates # }}}

optD = {
	'SGD' : SGD
,	'Adam' : Adam
}

# }}}


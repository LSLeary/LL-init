import theano
import theano.tensor as T
import numpy as np
import time
import h5py
import cPickle as pickle

from utilityFunctions import (preds, printStats, geoMean, floatToString,
		getWithName, intCounter)
from objectives import objecD
from optimisers import optD, Optimiser

# Wrapper object for the theano tensors and associated information that
# get passed though the network when it's constructed.
class Representation(object): # {{{

	def __init__(self, shape, trainTensor=None, testTensor=None, layers=None):
		if layers is None:
			layers = []
		self.shape = shape
		self.layers = layers
		nd = len(self.shape)+1
		if trainTensor is None:
			self.trainTensor = T.TensorType(
				dtype=theano.config.floatX, broadcastable=(False,)*nd
			)('X_tr')
		else:
			self.trainTensor = trainTensor
		if testTensor is None:
			self.testTensor = T.TensorType(
				dtype=theano.config.floatX, broadcastable=(False,)*nd
			)('X_te')
		else:
			self.testTensor = testTensor

	# Given an archiver object, reports the layers the rep has passed through.
	def show(self, arc):
		arc.log("")
		arc.log("A representation having moved through:")
		for layer in self.layers:
			layer.show(arc) # }}}

# Actual Neural Network object.
class NN(object): # {{{

	def __init__(self, inRep, outRep, archiver=None, # {{{
			objective=None, optimiser=None, postBatchFunctions=None,
			postEpochFunctions=None, classify=True):

		if postBatchFunctions is None:
			postBatchFunctions = []
		if postEpochFunctions is None:
			postEpochFunctions = []
		if archiver is None:
			archiver = Archiver()

		self.testInTensor = inRep.testTensor
		self.trainInTensor = inRep.trainTensor
		self.testOutTensor = outRep.testTensor
		self.trainOutTensor = outRep.trainTensor

		self.classify = classify
		self.labelTensor = T.matrix('y')

		self.set(objective, optimiser)

		self.trainCompiled = False
		self.evaluateCompiled = False
		self.observeCompiled = False

		self.layers = outRep.layers
		self.trainableWeights = []
		self.depth = 0
		self.numWeights = 0
		ndict = {}
		for layer in self.layers:
			self.trainableWeights += layer.trainableWeights
			self.depth += layer.depth
			self.numWeights += layer.numWeights
			if layer.name not in ndict:
				ndict[layer.name] = 0
			ndict[layer.name] += 1
			layer.name += "_"+str(ndict[layer.name])
		self.pbfs = postBatchFunctions
		self.pefs = postEpochFunctions
		self.LSUVFlag = False
		self.LSUVdata = None
		self.restarts = 0
		self.observationLayers = []
		self.observationFs = []
		self.arc = archiver # }}}

	def set(self, objective, optimiser): # {{{
		if objective is not None:
			self.objective = getWithName(objecD, objective)

		if type(optimiser) is str:
			self.optimiser = getWithName(optD, optimiser)() # passed the name
		elif type(optimiser) is type and issubclass(optimiser, Optimiser):
			self.optimiser = optimiser() # passed the class
		elif optimiser is not None:
			self.optimiser = optimiser # assumed to have passed an instance }}}

	def compileTrain(self, objective=None, optimiser=None): # {{{
		self.set(objective, optimiser)
		self.trainLoss = self.objective(self.trainOutTensor, self.labelTensor)
		for layer in self.layers:
			for W in layer.regWeights:
				self.trainLoss += 0.5 * layer.reg * layer.regFunction(W)
		self.optimiser.compile(self.trainLoss)

		self.updates = []
		self.weightList = []
		for layer in self.layers:
			self.weightList += layer.trainableWeights
			self.updates += layer.manualUpdates
		self.updates += self.optimiser(self.weightList)

		out = self.trainLoss
		if self.classify:
			_, accuracy = preds(self.trainOutTensor, self.labelTensor)
			out = [out, accuracy]
		self._trainOnBatch = theano.function(
			[self.trainInTensor, self.labelTensor]
		,	out
		,	updates=self.updates
		)
		self.trainCompiled = True

	def trainOnBatch(self, x, y):
		if not self.trainCompiled:
			self.compileTrain()
		return self._trainOnBatch(x, y) # }}}

	def compileEvaluate(self): # {{{
		self.testLoss = self.objective(self.testOutTensor, self.labelTensor)
		out = [self.testLoss]
		if self.classify:
			predictions, accuracy = preds(self.testOutTensor, self.labelTensor)
			out += [predictions, accuracy]
		else:
			out += [self.testOutTensor]
		self._eval = theano.function(
			[self.testInTensor, self.labelTensor]
		,	out
		)
		self.evaluateCompiled = True

	def evaluate(self, x, y, batchSize=64):
		if not self.evaluateCompiled:
			self.compileEvaluate()
		numBatches = x.shape[0]/batchSize
		batches = [
			(x[i*batchSize:(i + 1)*batchSize], y[i*batchSize:(i + 1)*batchSize])
			for i in range(numBatches)
		]
		if x.shape[0] % batchSize != 0:
			batches.append((x[numBatches*batchSize:], y[numBatches*batchSize:]))
		lS = []
		aS = []
		predS = []
		for xBatch, yBatch in batches:
			if self.classify:
				l, pred, a = self._eval(xBatch, yBatch)
				aS.append(xBatch.shape[0]*a)
			else:
				l, pred = self._eval(xBatch, yBatch)
			lS.append(xBatch.shape[0]*l)
			predS.append(pred)
		l = sum(lS)/x.shape[0]
		pred = np.concatenate(predS, axis=0)
		if self.classify:
			a = sum(aS)/x.shape[0]
			return l, pred, a
		else:
			return l, pred # }}}

	def trainForEpoch(self, x, y, batchSize, # {{{
			validationData=None, verbosity=1, plotting=False, r=6):
		# x, y can be numpy arrays wth data and labels respectively,
		# or x can be a callable returning a batch: data, labels
		# in which case y must specify the number of batches to perform,
		# and x must accept the batchSize keyword argument.
		if type(y) is int:
			generating = True
			batchIndices = range(y)
			epochLen = y * batchSize
		else:
			generating = False
			epochLen = x.shape[0]
			inds = np.arange(epochLen)
			np.random.shuffle(inds)
			batchIndices = [inds[batchSize*i:batchSize*(i + 1)]
					for i in range(epochLen / batchSize)]
			if epochLen % batchSize != 0:
				batchIndices += [inds[batchSize*(epochLen / batchSize):]]

		loss = 0.
		acc = 0.
		a = 0.
		if plotting:
			losses = []
			accs = []

		t0 = time.time()
		for i, binds in enumerate(batchIndices):
			if generating:
				xBatch, yBatch = x(batchSize=batchSize)
			else:
				xBatch = x[binds]
				yBatch = y[binds]
			if self.classify:
				l, a = self.trainOnBatch(xBatch, yBatch)
			else:
				l = self.trainOnBatch(xBatch, yBatch)
			if plotting:
				losses.append(l)
				accs.append(a)
			loss = (i*loss + l) / (i + 1)
			acc = (i*acc + a) / (i + 1)
			tcur = time.time() - t0
			if i == 0:
				tproj = (epochLen / batchSize) * tcur
				t0 = time.time()
			else:
				tproj = (epochLen / batchSize) * (tcur / i)

			if verbosity == 1:
				self.arc.log(
					("(%d/%d) - Loss: " + floatToString(loss, r) + " - Acc: "
					+ floatToString(acc, r) + " - T: %d/%ds  "
					) % (
						min(batchSize*(i + 1), epochLen)
					,	epochLen
					,	int(tcur)
					,	int(tproj)
					)
				,	eol='\r'
				)
			for f in self.pbfs:
				f(model=self, l=l, a=a, avgLoss=loss, avgAcc=acc, t=tcur)
			if np.isnan(l):
				self.arc.log("")
				self.arc.log("Loss is NaN; cancelling epoch.")
				return None, None, [[], [], None, None]
		tf = time.time()

		report = "Loss: " + floatToString(loss, r)
		report += " - Acc: " + floatToString(acc, r)
		if validationData is not None:
			vloss, _, vacc = self.evaluate(validationData[0], validationData[1])
			report += " - V loss: " + floatToString(vloss, r)
			report += " - V acc: " + floatToString(vacc, r)
		else:
			eplToken = "(%d/%d) - " % (epochLen, epochLen)
			report = eplToken + report
		tLen = max(4, 2*len(str(int(tf-t0))) + 1)
		report += " - T: " + floatToString(tf-t0, tLen) + "s"
		self.arc.log(report)

		if plotting:
			if validationData is None:
				return loss, acc, [losses, accs]
			else:
				return loss, acc, [losses, accs, vloss, vacc]
		else:
			return loss, acc, [] # }}}

	def schedule(self, x, y, scheduler, batchSize=64, # {{{
			validationData=None, verbosity=1, plotting=False,
			saveWeights=True, name='1', restartDecay=1.):

		scheduler.set(optimiser=self.optimiser, archiver=self.arc)

		if plotting:
			ds = ['plots', name]
			cuunter = 0
			losses = []
			accs = []
			allEpochLs = []
			allEpochAs = []
			if validationData is not None:
				vlosses = []
				vaccs = []

		if saveWeights:
			self.arc.saveWeights(self, name + " - epoch 0 (init).h5")

		loss = None
		acc = None
		epochLs = None
		epochAs = None
		vloss = None
		vacc = None
		while scheduler.epoch(loss=loss, acc=acc):

			self.arc.log("")
			self.arc.log("Epoch #%d, with learning rate %f and batch size %d"
					% (scheduler.currentEpoch, scheduler.lr, batchSize))

			loss, acc, l = self.trainForEpoch(x, y, batchSize,
					validationData=validationData,
					verbosity=verbosity, plotting=plotting)
			if plotting:
				losses.append(loss)
				cuunter += 1
				accs.append(acc)
				epochLs = l[0]
				epochAs = l[1]
				allEpochLs.extend(epochLs)
				allEpochAs.extend(epochAs)

				pe_ds=['epochal data']
				self.arc.plot([], [epochLs], name='Loss - Epoch %d'
						% scheduler.currentEpoch, directoryStructure=ds+pe_ds)
				self.arc.plot([], [epochAs], name='Accuracy - Epoch %d'
						% scheduler.currentEpoch, directoryStructure=ds+pe_ds)

				lossesL = [losses]
				accsL = [accs]
				if validationData is not None:
					vloss = l[2]
					vacc = l[3]
					vlosses.append(vloss)
					vaccs.append(vacc)
					lossesL.append(vlosses)
					accsL.append(vaccs)

				self.arc.plot([], lossesL, name='Per Epoch Loss',
						directoryStructure=ds)
				self.arc.plot([], accsL, name='Per Epoch Accuracy',
						directoryStructure=ds)

				self.arc.plot([], [allEpochLs], name='Loss',
						directoryStructure=ds)
				self.arc.plot([], [allEpochAs], name='Accuracy',
						directoryStructure=ds)

				self.arc.pickle(
					{'loss':loss, 'acc':acc, 'l':l}
				,	"epoch %d.b" % scheduler.currentEpoch
				,	directoryStructure=ds+pe_ds
				)
			for f in self.pefs:
				restartFlag = f(model=self, epoch=scheduler.currentEpoch,
						avgLoss=loss, avgAcc=acc, losses=epochLs,
						accs=epochAs, valLoss=vloss, valAcc=vacc)
				if restartFlag:
					self.restarts += 1
					self.arc.log("")
					self.arc.log(
						"Restarting for the "+intCounter(self.restarts)+" time."
					)
					self.reinit()
					scheduler.refresh(lr=scheduler.lr*restartDecay)
					return self.schedule(x, y, scheduler=scheduler,
							batchSize=batchSize, validationData=validationData,
							verbosity=verbosity, plotting=plotting,
							saveWeights=saveWeights, name=name,
							restartDecay=restartDecay)
			if saveWeights:
				self.arc.saveWeights(self,
						name + " - epoch %d.h5" % scheduler.currentEpoch)
		if validationData is not None:
			return l[2], l[3]
		else:
			return loss, acc # }}}

	def loadWeights(self, name): # {{{
		f = h5py.File(name, 'r')
		for layer in self.layers:
			layer.loadWeights(f)
		f.close() # }}}

	def reinit(self, newArchiver=None): # {{{
		if self.optimiser is not None:
			self.optimiser.refresh()
		for layer in self.layers:
			layer.reinit()
		if self.LSUVFlag:
			self.LSUV(self.LSUVdata)
		if newArchiver is not None:
			self.arc = newArchiver # }}}

	def show(self): # {{{
		self.arc.log("")
		self.arc.log("A depth %d Neural Net on %d parameters, with layers:" %
				(self.depth, self.numWeights))
		for layer in self.layers:
			layer.show(self.arc) # }}}

	def debugInit(self, x): # {{{
		m, v = printStats(self.arc, x)
		for i, layer in enumerate(self.layers):
			self.arc.log("")
			self.arc.log("The batch of reprentations is shape "+str(x.shape))
			self.arc.log("Now passing through:")
			layer.show(self.arc)
			x = layer.debugCall(x)
			m, v = printStats(self.arc, x) # }}}

	def compileObservation(self): # {{{
		previousTensor = self.trainInTensor
		tlBuffer = []
		for layer in self.layers:
			if layer.name[:11] == "Observation":
				self.observationLayers.append(layer)
				f = theano.function([previousTensor], layer.tensor)
				self.observationFs.append(f)
				layer.setF(f)
				layer.setPredecessors(tlBuffer)
				tlBuffer = []
				previousTensor = layer.tensor
			elif layer.trainable:
				tlBuffer.append(layer)
		self.observeCompiled = True

	def observe(self, x):
		if not self.observeCompiled:
			self.compileObservation()
		self.arc.log("")
		self.arc.log("The batch of reprentations is shape "+str(x.shape))
		m, v = printStats(self.arc, x)
		for layer in self.observationLayers:
			x = layer.f(x)
			self.arc.log("")
			self.arc.log("The batch of reprentations is shape "+str(x.shape))
			m, v = printStats(self.arc, x) # }}}

	# {{{ Not quite LSUV as originally proposed, but closely based on.
	# Passes one batch to start, making the required corrections to the weights.
	# Next pass, it passes twice as many batches though at the same time,
	# applying the (geometric) mean of their corrective factors.
	# The variance is measured at observation points that are inserted by using
	# Observation layers in the model, and the corrective factors are
	# distributed over all trainable layers in between the observation points.
	def LSUV(self, data, batchSize=256, debug=False, passes=5):
		if not self.observeCompiled:
			self.compileObservation()
		for j in range(passes):
			inds = np.arange(data.shape[0]); np.random.shuffle(inds)
			xs = [data[inds[batchSize*i:batchSize*(i + 1)]] for i in range(2**j)]

			for layer in self.observationLayers:
				vs = []
				for x in xs:
					xtmp = layer.f(x)
					m = xtmp.mean()
					v = ((xtmp-m)**2).mean()
					if v > 10**(-8):
						vs.append(v)
				std = pow(geoMean(vs), 1./(2*len(layer.predecessors)))
				std += 10**(-6)
				for pred in layer.predecessors:
					for W in pred.trainableWeights:
						W.set_value(W.get_value()/std)
				for i, x in enumerate(xs):
					xs[i] = layer.f(x)
		if debug:
			inds = np.random.random_integers(size=(batchSize,), low=0,
					high=(data.shape[0]-1))
			self.arc.log("")
			self.arc.log("Dry run with fresh batch:")
			self.observe(data[inds])

		self.LSUVdata = data
		self.LSUVFlag = True # }}}

# }}}


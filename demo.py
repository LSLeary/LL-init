# {{{ Imports

# Standard
import numpy as np
from itertools import repeat
from sys import setrecursionlimit
setrecursionlimit(50000)

# Provided
from utilityFunctions import restart
from schedulers import AutoScheduler, ManualScheduler
from archivers import Archiver, ExperimentArchiver
from neuralNetwork import Representation, NN
from optimisers import SGD, Adam
from objectives import mse, softmax
from layers import (FC, Activation, ParReLU, Flatten, Convolution, Pooling,
		Dropout, BatchNormalisation, CReLU, Augmentation, Observation, merge)
from initialisations import glorotNormal, orthogonal
from regularisation import l1, l2

# }}}

# Config {{{

# Top Level {{{

# Load your data in here. The default parameters in this file are choosen for
# normalised or whitened CIFAR10. Format should be numpy arrays with shapes:
# x: (data, channels, vertical axis, horizontal axis)
# y: (data, classes), or (data, 1) for regression to a point.
xTrain, yTrain, xVal, yVal = 

# Data
trainingData = (xTrain, yTrain)
valData = (xVal, yVal)
# If testing
#trainingData = (
#	np.concatenate([xTrain, xVal], axis=0)
#,	np.concatenate([yTrain, yVal], axis=0)
#)
#valData = None

# Print a summary of the model before training.
showModel = True
# Perform the forward pass with a test batch, reporting on the state of the
# representation after each layer, or just every observation layer if the
# observing flag is set below.
debugInit = False
observing = False
# Perform a variant of LSUV; checking and correcting variances at observation
# layers. No version of LSUV is implemented for resnet, so this will be ignored
# if the resNet flag is set.
doLSUV = False
# Path to weight-file to load.
loadWeights = None

# Use the AS
autoSchedule = True
# Save weights to file after each epoch.
saveWeights = False
# Plot graphs of the training. Does nothing if the basic Archiver object is used.
plotting = True
# Post epoch functions; these get run after each epoch.
# If a PEF returns x s.t. bool(x) is True, the machine will reinitialise and
# start over. They receive certain keyword arguments from the model; see source.
# The restart function returns a function that forces a restart if accuracy
# on the training set is less than its first argument and accuracy on the
# validation set is less than its second. (bad init detection)
pefs = [restart(0.12, 0.15)]
# LR is multiplied by this factor with each restart.
restartDecay=1.

# Initial learning rate.
# Suggested values for LL: inversely proportional to depth,
# also decreasing with width. If the model crashes the LR is too high.
initLR = 150*10**(-6)
# Factor by which to decay the learning rate when the AS stops making progress,
# or when the manual scheduler asks.
decay = 1./8.
# Optimisers. Adam with default settings is recommended.
opt = Adam
# SGD works perfectly well when it starts.
#opt = SGD(momentum='nesterov', mu=0.99)
# Batch size.
bs = 256

# Regularisation function.
#regF = l2
regF = None
# Reg parameter. Does nothing is regF is None.
regP = 0.0006

# Manual scheduling. These settings do nothing when autoSchedule is True.
# Number of passes over the training set.
epochs = 200
# Increasing list of floats in (0, 1); the learning rate will drop by decay
# after each proportion of epochs has been completed.
schedl = [1./2., 3./4., 7./8.]

# Use the convolutional model.
conv = True
# Make residual connections.
resNet = False

# Initialisation.
init = orthogonal
#init = glorotNormal
# Perform LL-init by using CReLUs and mirroring the init.
mirroring = True

# Conv only.
# Basic model building parameters; each section is a number of modules and
# a spatial extent reduction.
numSections = 4
modsPerSection = 49
# Perform max pooling. If False, downsampling is done via strides of 2.
maxPooling = False
# Width of the model. In plain rectified networks and resnets, this is the
# number of filters per module in the first section. If CReLUs are used,
# the number of filters is reduced to use the ~same number of parameters
# per module instead.
width = 8
# Factor by which width increases with each section.
widthFactor = 2
# Where width is capped.
maxWidth = 512

# Dropout.
doDropout = True
# Base level of dropout. Suggested value for deep LL-init: ~1/1000.
baseDropout = 0.0012
# Amount to increment dropout by with each section. Conv only.
incrDropoutBy = baseDropout

# Data augmentation. Conv only.
# Maximum number of pixels to translate by in both x and y-axes.
# Translations and flips are picked uniformly and naively applied to the
# whole batch.
translateBy = 4
# Flip horizontally with p = 0.5.
hFlip = True
# Flip vertically with p = 0.5.
vFlip = False

# Sequence of layer sizes for FC model.
# Final layer is omitted since it's determined by the data set.
fcwidths = [500 for i in range(40)]

# Objective function.
obj = softmax

# }}}

# This function institutes the config that depends on the other config.
# It should be re-run before training a new model if config has changed in the
# meantime.
def reconf(): # {{{

	# Key-word arguments for the initialisation function.
	global initKWArgs
	initKWArgs = {
		'mirror' : mirroring
	,	'relu'   : True
	}

	global scheduler
	if autoSchedule:
		scheduler = AutoScheduler(
			initLR
		,	lRDecay=decay
		)
	else:
		scheduler = ManualScheduler(
			initLR
		,	end=epochs
		,	schedule=schedl
		,	decays=repeat(decay)
		)

	# For resnet.
	# These factors determine how much signal from each active path is mixed
	# into the signal from the skip connection. (see 3.4 of the paper)
	# Currently un-trained.
	global gamma1
	global gamma2
	# Suggested value: 1/n < gamma2 < 1/sqrt(n), where n is the number of
	# skip connections in the model.
	gamma2 = 2./(numSections*modsPerSection)
	# This line shouldn't change; we need gamma1**2 + gamma2**2 == 1
	gamma1 = pow(1.-gamma2**2, 1./2)

	# This section just automatically names some things; not important.
	global linToken
	global archToken
	global regToken
	if not mirroring:
		linToken = " rectified "
	elif mirroring:
		linToken = " CReLU-lin-init'd "
	else:
		linToken = " linear "
	if conv:
		archToken = "ConvNet"
	else:
		archToken = "FCNet"
	if regF is None:
		regToken = ' unregd '
	else:
		regToken = ' regd ' # }}}

reconf()

# Basic 'Archiver' object. Only capable of weight saving and printing lines.
arc = Archiver()
# More advanced version. See source.
#arc = ExperimentArchiver(
#	["archivers.py", "initialisations.py", "layers.py", "neuralNetwork.py",
#			"objectives.py", "optimisers.py", "regularisation.py",
#			"schedulers.py", "utilityFunctions.py"]
#,	name=init.__name__+linToken+archToken+regToken+"w="+str(width)
#,	directory="Archive"
#,	plotFormat="pdf"
#)

# }}}

# Model building functions  {{{

# Building block for FC models.
# Also useful for adding FC layers to convolutional models.
def fcModule(flow, w, dropout, regf=regF, reg=regP): # {{{
	if mirroring:
		flow = CReLU()(flow)
	flow = FC(w, initialisation=init, initKWArgs=initKWArgs, reg=regP, regFunction=regF)(flow)
	if observing and not resNet:
		flow = Observation()(flow)
	if not mirroring:
		flow = Activation('relu')(flow)
	if doDropout:
		flow = Dropout(dropout)(flow)
	return flow # }}}

# A basic FC model.
def modFC(): # {{{
	inRep = Representation(shape=(3, 32, 32))
	flow = Flatten()(inRep)

	for w in fcwidths:
		flow = fcModule(flow, w, baseDropout)

	if mirroring:
		flow = CReLU()(flow)
	outRep = FC(10, initialisation=init, initKWArgs=initKWArgs, reg=regP, regFunction=regF)(flow)
	if observing and not resNet:
		outRep = Observation()(outRep)

	return NN(
		inRep
	,	outRep
	,	archiver=arc
	,	optimiser=opt
	,	objective=obj
	,	postEpochFunctions=pefs
	) # }}}

# Building block for convolutional models.
def convModule(flow, filters, dropout, strides=(1, 1), s=(3, 3)): # {{{
	if mirroring:
		flow = CReLU()(flow)
	flow = Convolution(filters, s=s, initialisation=init, initKWArgs=initKWArgs, strides=strides, regFunction=regF, reg=regP)(flow)
	if observing and not resNet:
		flow = Observation()(flow)
	if not mirroring:
		flow = Activation('relu')(flow)
	if doDropout:
		flow = Dropout(dropout)(flow)
	return flow # }}}

# Building block for residual models.
def resModule(gamma2, x, filters, dropout, s=(3, 3)): # {{{
	y = convModule(x, filters, dropout, s=s)
	y = convModule(y, filters, dropout, s=s)
	f = lambda x, y: gamma1*x + gamma2*y
	out = merge(x, y, f=f)
	if observing:
		out = Observation()(out)
	return out # }}}

# Convolutional model.
def modConv(): # {{{

	dropout = baseDropout

	if mirroring:
		filters = int(width/np.sqrt(2.))
		fmax = int(maxWidth/np.sqrt(2.))
	else:
		filters = width
		fmax = maxWidth

	if maxPooling:
		st = (1, 1)
	else:
		st = (2, 2)

	a = 1
	if mirroring:
		a = np.sqrt(2.)

	inRep = Representation(shape=(3, 32, 32))
	flow = Augmentation(translate=translateBy, hFlip=hFlip, vFlip=vFlip)(inRep)

	for chunk in range(numSections):

		filters = int(width * widthFactor**chunk / a)
		if filters > fmax:
			filters = fmax

		if chunk != 0:
			if maxPooling:
				flow = Pooling('max')(flow)
			flow = convModule(flow, filters, dropout, strides=st)
		else:
			flow = convModule(flow, filters, dropout)
		if observing and resNet:
			flow = Observation()(flow)
		if resNet:
			assert modsPerSection % 2 == 1
			for j in range((modsPerSection-1)/2):
				flow = resModule(gamma2, flow, filters, dropout)
		else:
			for j in range(modsPerSection-1):
				flow = convModule(flow, filters, dropout)

		if dropout+incrDropoutBy > 0.5:
			dropout = 0.5
		else:
			dropout += incrDropoutBy

	flow = convModule(flow, filters, dropout, strides=st)
	flow = Flatten()(flow)

	if mirroring:
		flow = CReLU()(flow)
	outRep = FC(10, initialisation=init, initKWArgs=initKWArgs, reg=regP, regFunction=regF)(flow)
	if observing:
		outRep = Observation()(outRep)

	mod = NN(
		inRep
	,	outRep
	,	archiver=arc
	,	optimiser=opt
	,	objective=obj
	,	postEpochFunctions=pefs
	)
	return mod # }}}

# }}}

# Main {{{

def main(name, model=None):

	reconf()

	if model is None:
		if conv:
			model = modConv()
		else:
			model = modFC()

	if showModel:
		model.show()

	if debugInit and doLSUV and not resNet:
		if observing:
			model.observe(xTrain[:bs])
		else:
			model.debugInit(xTrain[:bs])
		arc.log("")
		arc.log("  ^---   Pre LSUV  ---^")
		arc.log("-------------------------------------------------------------")
		arc.log("  v---  Post LSUV  ---v")

	if loadWeights is not None:
		model.loadWeights(loadWeights)
	elif doLSUV and not resNet:
		model.LSUV(xTrain, batchSize=bs)

	if debugInit:
		if observing:
			model.observe(xTrain[:bs])
		else:
			model.debugInit(xTrain[:bs])

	return model, model.schedule(trainingData[0], trainingData[1], scheduler,
		batchSize=bs, validationData=valData, plotting=plotting,
		saveWeights=saveWeights, name=name, restartDecay=restartDecay) # }}}

main("demo")

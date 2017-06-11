import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams()
import numpy as np
import operator
import time
import copy

# Utility functions {{{
# Simple functions for use throughout the script.

def prod(l):
	return reduce(operator.mul, l, 1)

def ell2(x):
	return np.sqrt((x ** 2).sum())

def geoMean(l): # {{{
	if l == []:
		return 1.
	n = len(l)
	p = 1.
	for a in l:
		p *= a
	return p ** (1./n) # }}}

def stats(x): # {{{
	m = x.mean()
	v = ((x-m) ** 2).mean()
	std = np.sqrt(v)
	return m, v, std

def printStats(arc, x):
	m, v, _ = stats(x)
	arc.log("")
	arc.log("%f mean" % m)
	arc.log("%f variance" % v)
	return m, v # }}}

# Just a convenience function to format floats for some other functions.
def floatToString(l, r=7): # {{{
	if type(l) is list:
		accum = floatToString(l[0], r=r)
		for f in l[1:]:
			accum += ", " + floatToString(f, r=r)
		return accum
	elif l <= 10**7 and l >= 10**(-4):
		return ("{:{width}.{prec}f}").format(l, width=r, prec=r-2)[:r]
	else:
		return ("{:{width}.{prec}e}").format(l, width=r, prec=r-6) # }}}

def intCounter(i): # {{{
	if i in [11, 12, 13]:
		return "%dth" % i
	elif i % 10 == 1:
		return "%dst" % i
	elif i % 10 == 2:
		return "%dnd" % i
	elif i % 10 == 3:
		return "%drd" % i
	else:
		return "%dth" % i # }}}

# Takes a time0 and naively produces a timestamp for the current time.
def timeStamp(t0, digits_left=5, digits_right=3): # {{{
	timest = str(time.time()-t0)
	timest = timest.split('.')
	l1 = len(timest[0])
	l2 = len(timest[1])
	if l1 < digits_left:
		timest[0] = "000000000000000000000000"[:digits_left-l1] + timest[0]
	if l2 < digits_right:
		timest[1] = timest[1] + "000000000000000000000000"[:digits_left-l2]
	if l2 > digits_right:
		timest[1] = timest[1][:digits_right]
	return "["+timest[0] +"."+ timest[1]+"]" # }}}

def cast(x):
	return np.cast[theano.config.floatX](x)

def sharedVar(x, **kwargs):
	return theano.shared(cast(x), **kwargs)

# {{{ Zeroing shared variables
# Returns a clone of a shared variable, all values zero'd.
def zeroClone(svar):
	v = theano.shared(0.*svar.get_value())
	v.type = svar.type
	v.tag = copy.copy(svar.tag)
	return v

# Returns /the same/ shared variable, all values zero'd.
def zeroShared(svar):
	curVal = svar.get_value()
	newVal = np.zeros_like(curVal)
	return svar.set_value(newVal) # }}}

# Non-failing dictionary lookup so objects can be passed directly /or/ looked up.
def getWithName(dic, name): # {{{
	if name in dic:
		return dic[name]
	else:
		return name # }}}

# Theano shortcut; with probability p do th, else do el
def doWithProb(p, th, el):
	boolean = srng.choice(size=(1,), a=2, p=[1.-p, p], dtype='int8')
	return theano.ifelse.ifelse(boolean[0], th, el)

# Takes a 4D theano variable x and a maximum number of pixels,
# then picks uniformly and performs a translation on x in the last two axes.
def randomTranslate(x, pixels): # {{{
	if pixels == 0:
		return x
	p = 1./(1.+2.*pixels)
	xhat = T.zeros(x.shape)
	hv = srng.random_integers(
		size=(2,)
	,	low=1
	,	high=pixels
	,	dtype=theano.config.intX
	)
	x = doWithProb(p, x, doWithProb(0.5
	,	T.set_subtensor(xhat[:,:,hv[0]:,:], x[:,:,:-hv[0],:])
	,	T.set_subtensor(xhat[:,:,:-hv[0],:], x[:,:,hv[0]:,:])
	))
	x = doWithProb(p, x, doWithProb(0.5
	,	T.set_subtensor(xhat[:,:,:,hv[1]:], x[:,:,:,:-hv[1]])
	,	T.set_subtensor(xhat[:,:,:,:-hv[1]], x[:,:,:,hv[1]:])
	))
	return x # }}}

# Takes a 4D theano variable and reflects with p=0.5 in the directions specified.
def randomReflect(x, vertical, horizontal): # {{{
	if vertical:
		x = doWithProb(0.5, x, x[:,:,::-1,:])
	if horizontal:
		x = doWithProb(0.5, x, x[:,:,:,::-1])
	return x # }}}

# For generating functions to pass to models, which will use them to decide
# whether or not to start over.
def restart(minAcc=0.14, minVacc=1.): # {{{
	def r(avgAcc=None, valAcc=None, **kwargs):
		return (avgAcc < minAcc) and (valAcc < minVacc)
	return r # }}}

# Takes two lists and returns an initial segment of shared objects,
# the rest of the first list, and the rest of the second list.
def sharedSeg(xs, ys): # {{{
	acc = []
	for x, y in zip(xs, ys):
		if x is y:
			acc = acc + [x]
		else:
			break
	n = len(acc)
	return acc, xs[n:], ys[n:] # }}}

# Takes the (theano variable) output of a classifier and the associated labels,
# returns the predictions and accuracy.
def preds(out, labels): # {{{
	predictions = T.argmax(out, axis=-1)
	correctness = T.eq(
		T.argmax(labels, axis=-1)
	,	predictions
	)
	accuracy = T.mean(correctness)
	return predictions, accuracy # }}}

# }}}


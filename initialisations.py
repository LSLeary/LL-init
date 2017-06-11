import numpy as np

from utilityFunctions import sharedVar

# Initialisation functions take (input shape, output shape, **kwargs) and
# return a list of theano shared variables populated with initialised weights.

# Initialisations {{{

def glorotNormal(inputShape, outputShape, s=(3, 3), relu=False, # {{{
		mirror=False, signalPriorities=(1.,1.), **kwargs):

	if mirror:
		assert inputShape[0] % 2 == 0
		iS = list(inputShape)
		iS[0] = iS[0]/2
		inputShape = tuple(iS)

	if len(inputShape) == 1:
		conv = False
		fanIn = inputShape[0]
		fanOut = outputShape[0]
	elif len(inputShape) == 3:
		conv = True
		fanIn = s[0]*s[1]*inputShape[0]
		fanOut = s[0]*s[1]*outputShape[0]

	if relu and not mirror:
		a = 2.
	else:
		a = 1.
	prioritySum = sum(signalPriorities)
	var = prioritySum * a / (signalPriorities[1] * fanIn
			+ signalPriorities[0] * fanOut)

	if conv:
		w = np.sqrt(var)*np.random.randn(outputShape[0],inputShape[0],s[0],s[1])
		b = np.zeros((1, outputShape[0], 1, 1))
		bcast = (True, False, True, True)
		ax = 1
	else:
		w = np.sqrt(var)*np.random.randn(fanIn, fanOut)
		b = np.zeros((1, outputShape[0]))
		bcast = (True, False)
		ax = 0

	if mirror:
		w = np.concatenate([w, -w], axis=ax)

	w = sharedVar(w)
	b = sharedVar(b, broadcastable=bcast)
	return [w, b] # }}}

# Takes shapes and returns a list containing an orthogonally initialised theano
# matrix or kernel, and bias vector. For Fully Connected layers, inputShape and
# outputShape are just numbers wrapped in (,) e.g. outputShape=(500,) for
# a 500 neuron layer. For Convolutions, inputShape and outputShape are of
# the form (Channels, Feature Map axis, other Feature Map axis)
# e.g. outputShape=(8, 32, 32) for 8 filters on 32x32 images.
# s is the size of each filter.
# If relu=True, tries to preserve variance in cases where ReLUs are used.
# If mirror=True, produces the LL-version of the orthogonal init for CReLU.
# If pooling=True, assumes you're using the resulting convolutional kernel
# for downsampling with strides of 2. In which case, rather than only place
# a non-zero weight in the middle of a filter, it distributes each weight
# across a 2x2 square, effectively performing average pooling before convolving.
def orthogonal(inputShape, outputShape, # {{{
		s=(3, 3), relu=False, mirror=False, pooling=False, **kwargs):

	# Figuring out if the init is for a convolution or not.
	forConv = (len(inputShape) == 3)

	# Determining the number of channels in / out.
	channelsIn = inputShape[0] # fan in if FC
	numFilters = outputShape[0] # fan out if FC
	if mirror:
		assert channelsIn % 2 == 0
		channelsIn /= 2

	# SVD of a randomly generated matrix gets us a nice orthogonal matrix.
	m = max(numFilters, channelsIn)
	u = np.random.normal(size=(m, m))
	u, _, _ = np.linalg.svd(u)
	u = u[:numFilters, :channelsIn]

	# If forConv, sets the central column of each filter to one of the orthonormal
	# vectors we just generated. Otherwise just makes the appropriate matrix.
	if forConv:
		w = np.zeros((numFilters, channelsIn, s[0], s[1]))
		cx = (s[0]-1)/2 # central coordinate each filter (x-axis)
		cy = (s[1]-1)/2 #                                (y-axis)
		w[:,:,cx,cy] = u
		if pooling:
			w[:,:,cx-1,cy] = u
			w[:,:,cx,cy-1] = u
			w[:,:,cx-1,cy-1] = u
			# This factor is fairly arbitrary, and probably not important.
			if mirror:
				w = 0.28*w
			else:
				w = 0.4*w
			# 1/4 would be a true mean, 1/2 would be a variance preserving mean
			# given the assumption that adjacent values in feature maps are
			# uncorrelated. A value just a little higher than 1/4 is ideal in
			# an LL-init'd model where nearby activations maintain the strong
			# correlation present in the data. Otherwise use a higher value.
		b = np.zeros((1, numFilters, 1, 1))
		bcast = (True, False, True, True)
		ax = 1
	else:
		w = u.T
		b = np.zeros((1, numFilters))
		bcast = (True, False)
		ax = 0

	# Performing mirroring.
	if mirror:
		w = np.concatenate([w, -w], axis=ax)

	# ReLU compensation is accounted for here.
	# The relu setting is ignored if mirror=True, since it doesn't apply.
	if relu and not mirror:
		a = 2.
	else:
		a = 1.

	# Rescaling for variance preservation. This step is not necessary.
	# Whenever channelsIn = numFilters (= channels out), orthogonality will
	# ensure variances are preserved. In the other cases, glorot style
	# rescaling will ensue, balancing variance/gradient preservation.
	w *= np.sqrt(2.*a*m/(channelsIn+numFilters))

	w = sharedVar(w)
	b = sharedVar(b, broadcastable=bcast)
	return [w, b] # }}}

initD = {
	'glorotNormal' : glorotNormal
,	'orthogonal' : orthogonal
}

# }}}


import theano.tensor as T

# Regularisation functions {{{

def l1(x):
	return T.sum(abs(x))

def l2(x):
	return T.sum(x ** 2)

regFD = {
	'l1' : l1
,	'l2' : l2
}

# }}}


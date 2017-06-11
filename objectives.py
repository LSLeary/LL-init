import theano.tensor as T

# Objectives {{{

def mse(yPred, yTrue):
	err = yPred - yTrue
	sqe = err**2
	msqe = T.mean(sqe)
	return msqe

def softmax(yPred, yTrue):
	yPred = yPred - T.max(yPred, axis=-1).reshape((-1, 1))
	sum_eys = T.sum(T.exp(yPred), axis=-1)
	losses = -T.sum(yPred*yTrue, axis=-1) + T.log(sum_eys)
	return T.mean(losses, axis=0)

objecD = {
	'mse' : mse
,	'softmax' : softmax
}

# }}}


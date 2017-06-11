import numpy as np
from itertools import repeat

# {{{ Scheduler
# A class for scheduling the training of an NN object.

class Scheduler(object): # {{{

	def __init__(self, initLR, archiver=None):
		self.lr = initLR
		self.arc = archiver
		self.currentEpoch = 0
		self.ready = False

	# Sets the optimiser and archiver
	def set(self, optimiser=None, archiver=None):
		if archiver is not None:
			self.arc = archiver
		if optimiser is not None:
			self.optimiser = optimiser
		if self.optimiser is not None:
			self.optimiser.compile(learningRate=self.lr)
			self.ready = True

	# Refreshes the scheduler instance to a ~starting state.
	def refresh(self, lr=None):
		self.currentEpoch = 0
		self.ready = False
		if lr is not None:
			self.lr = lr
			self.set()

	def log(self, message):
		if self.arc is not None:
			self.arc.log(message)

	# Runs between epochs, determining if another happens (return True),
	# and what the learning rate is.
	def epoch(self, **kwargs):
		if not self.ready:
			self.set()
		assert self.ready
		return False # }}}

# Starting with learning rate initLR, this auto-scheduler looks at the average
# rate of change over the last useLast epochs and reduces the learning rate
# by a factor of lRDecay when it falls below the threshold minGrad.
# After maxDrops drops, the next call for a drop will instead halt training.
# drops is used to specify the number of drops that have already happened;
# for continuing with a machine that was left partially trained.
class AutoScheduler(Scheduler): # {{{

	def __init__(self, initLR, lRDecay=1./8., minGrad=0.0015, # {{{
			mGDecay=2./3., useLast=10, archiver=None, drops=0, maxDrops=2):
		Scheduler.__init__(self, initLR, archiver=archiver)
		self.lRDecay = lRDecay
		self.minGrad = minGrad
		self.mGDecay = mGDecay
		self.useLast = useLast
		self.maxDrops = maxDrops
		self.numDrops = drops
		self.losses = []
		self.belowFor = 0
		for drop in range(drops):
			self.lr *= self.lRDecay
			self.minGrad *= self.mGDecay # }}}

	def refresh(self, lr=None): # {{{
		Scheduler.refresh(self, lr=lr)

		self.losses = []
		self.numDrops = 0
		self.belowFor = 0
		return self # }}}

	# {{{ Performs least-squares linear regression on the last however many
	# losses (using sequential integers for the x-values) and returns negative
	# the slope.
	def m(self):
		if len(self.losses) < self.useLast:
			return
		ys = np.array(self.losses[-self.useLast:])
		xs = np.arange(self.useLast)
		numer = xs.mean() * ys.mean() - (xs * ys).mean()
		denom = (xs.mean()) ** 2 - (xs ** 2).mean()
		return numer / denom # }}}

	def epoch(self, loss=None, acc=None, **kwargs): # {{{
		Scheduler.epoch(self)
		self.currentEpoch += 1
		if loss is not None:
			self.losses.append(loss)
		if acc is not None and acc < 0.11:
			self.lr *= 0.9
			self.optimiser.compile(learningRate=self.lr)
		if len(self.losses) < self.useLast:
			return True
		m = -self.m()
		if m < self.minGrad:
			self.belowFor += 1
		else:
			self.belowFor = 0
		if self.belowFor >= 5:
			self.numDrops += 1
			if self.numDrops <= self.maxDrops:
				self.log("")
				self.log("Dropping learning rate.")
			self.lr *= self.lRDecay
			self.minGrad *= self.mGDecay
			self.belowFor = 0
			self.optimiser.compile(learningRate=self.lr)
		else:
			self.log("")
			self.log(
				"Loss decreasing by %f per epoch [drop after %d epochs under %f]"
				% (m, 5 - self.belowFor, self.minGrad))
		if self.numDrops > self.maxDrops:
			self.log("")
			self.log("Done.")
			return False
		else:
			return True # }}}

# }}}

# Starting training with initial learning rate initLR, the manual scheduler
# takes a starting and ending epoch, and a schedule in the form of an
# increasing iterator of proportions, and an associated iterator of decays.
# E.g. schedule=[1./2., 3./4.], decays=[1./8., 1./9.] means drop the
# learning rate by a factor of 8 after doing half the epochs,
# then again by a factor of 9 after 3/4.
class ManualScheduler(Scheduler): # {{{

	def __init__(self, initLR, start=1, end=64, schedule=[0.5, 0.75],
			decays=repeat(1./8.), archiver=None):
		Scheduler.__init__(self, initLR, archiver=archiver)
		self.initLR = self.lr
		self.start = start
		self.end = end
		self.schedule = schedule + [1.]
		self.decays = decays
		self.scheduleIndex = 0
		for epoch in range(1,start):
			self.epoch()

	def refresh(self, lr=None):
		if lr is None:
			lr = self.initLR
		Scheduler.refresh(self, lr=lr)

		self.start = 1
		self.scheduleIndex = 0
		return self

	def epoch(self, **kwargs):
		Scheduler.epoch(self)
		if self.currentEpoch == self.end:
			return False
		self.currentEpoch += 1
		propDone = float(self.currentEpoch)/self.end
		if propDone >= self.schedule[self.scheduleIndex]:
			self.lr *= self.decays.next()
			self.scheduleIndex += 1
			self.optimiser.compile(learningRate=self.lr)
		return True # }}}

# }}}


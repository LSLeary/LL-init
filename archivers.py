from itertools import repeat

from sys import stdout
import os
import time
import subprocess
import __main__ as main

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import h5py
import cPickle as pickle

import gc

from utilityFunctions import timeStamp

# {{{ Archivers
# Objects that control the data produced during training or experiments.

# This archiver object is a basic version that only supports writing lines
# to stdout and saving weights.
class Archiver(object): # {{{

	def __init__(self, *args, **kwargs):
		self.weightDir = "."

	def collectGarbage(self):
		pass

	def stashFile(self, *args, **kwargs):
		pass

	def pickle(self, *args, **kwargs):
		pass

	def plot(self, *args, **kwargs):
		pass

	def saveWeights(self, nn, name):
		self.collectGarbage()
		path = self.weightDir+"/"+name
		f = h5py.File(path, 'w')
		for layer in nn.layers:
			layer.saveWeights(f)
		f.close()

	def log(self, line="", eol='\n'):
		assert eol in ['\n', '\r']
		if '\n' in line:
			raise Exception("Use the eol option for your newline.")
		if '\r' in line:
			raise Exception("Use the eol option for your return carriage.")
		stdout.write(line+eol)
		stdout.flush()
# }}}

# This one performs a plethora of simple IO tasks, intended to organise and
# preserve all output from running an experiment, as well as allow the
# experiment to be re-run at leisure.
#
# Some code is OS specific, and only tested on Debian.
# Should work on other flavours of Linux, and possibly other unix-like OSes.
class ExperimentArchiver(Archiver): # {{{

	def __init__(self, files, name=None, directory="Archive", # {{{
			plotFormat='png', doGC=True, reportGC=False):
		self.t0 = time.time()
		self.doGC = doGC
		self.reportGC = reportGC
		self.lastGC = self.t0
		self.files = [os.path.abspath(main.__file__)] + files
		self.name = name
		self.initTime = time.strftime("%Y-%m-%d %H-%M-%S")
		if name is None:
			name = ""
		else:
			name = " " + name
		self.saveDirectory = directory + "/" + self.initTime + name
		self.weightDir = self.saveDirectory + "/weights"
		subprocess.call(["mkdir", "-p", self.weightDir])
		self.logfile = open(self.saveDirectory+"/log", 'w', 1)
		for path in self.files:
			self.stashFile(path)
		if plotFormat[0] == '.':
			self.plotFormat = plotFormat
		else:
			self.plotFormat = '.' + plotFormat # }}}

	def log(self, line="", eol='\n'): # {{{
		if line != "":
			line = timeStamp(self.t0) + " " + line
		Archiver.log(self, line=line, eol=eol)
		if eol == '\n':
			self.logfile.write(line+eol) # }}}

	def plot(self, xs, ys, name, directoryStructure=None, # {{{
			linestyles=repeat('-'), xlim=None, ylim=None):
		if directoryStructure is None:
			directoryStructure = []
		if xs == []:
			xs = [range(len(y)) for y in ys]
		for x, y, ls in zip(xs, ys, linestyles):
			plt.plot(x, y, linestyle=ls)
		path = "/".join([self.saveDirectory]+directoryStructure)
		if not os.path.isdir(path):
			self.collectGarbage()
			subprocess.call(["mkdir", "-p", path])
		if xlim is not None:
			plt.xlim(xlim)
		if ylim is not None:
			plt.ylim(ylim)
		plt.savefig(path+"/"+name+self.plotFormat, bbox_inches='tight')
		plt.close() # }}}

	def stashFile(self, name, directoryStructure=None, move=False): # {{{
		if directoryStructure is None:
			directoryStructure = []
		self.collectGarbage()
		if move:
			command = "mv"
		else:
			command = "cp"
		directory = "/".join([self.saveDirectory]+directoryStructure)
		if not os.path.isdir(directory):
			subprocess.call(["mkdir", "-p", directory])
		subprocess.call([command, name, directory+"/"]) # }}}

	def pickle(self, obj, name, directoryStructure=None): # {{{
		if directoryStructure is None:
			directoryStructure = []
		path = "/".join([self.saveDirectory]+directoryStructure+[name])
		f = open(path, 'wb')
		pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close() # }}}

	def collectGarbage(self): # {{{
		if self.doGC and time.time()-self.lastGC > 120:
			if self.reportGC:
				self.log("")
				self.log("Collecting garbage...")
			collected = gc.collect()
			self.lastGC = time.time()
			if self.reportGC:
				self.log("Garbage collected: %d" % collected) # }}}

# }}}

# }}}


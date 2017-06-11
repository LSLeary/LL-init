# LL Init Code Release

For: The Shattered Gradients Problem (ICML2017)

### Overview of the files
- initialisations.py: with both the orthogonal kernels and the LL-init generated here, one heavily commented 69-line function `orthogonal` constitutes 100% of the novel code in this repository. It should be easy to port to other ML frameworks and programming languages.
- neuralNetwork.py: the core functionality is here; this file provides the Representation and NN objects. An NN object takes an initial and a final Representation.
- layers.py: Layer objects used to build models. An instantiated Layer takes a Representation and returns a new one.
- schedulers.py: objects that schedule the training of an NN object; taking the results of a completed epoch as arguments and determining whether or not another one is performed, as well as the learning rate.
- optimisers.py: objects that handle the details of weight-updates.
- archivers.py: objects that handle the data produced during training and experiments, including logging, weight saving and plotting.
- utilityFunctions.py: simple functions used throughout the other files.
- objectives.py: objective functions. Only provides mse and softmax.
- regularisation.py: regularisation functions. Only provides l1 and l2.
- demo.py: A demo frontend. Makes use of the provided framework simple by specifying and documenting config in a dedicated section, and providing functions which build and train models as dictated by that config.

A sensible default configuration is given for a 198 layer LL-init'd convnet; merely supply training data to the variables as indicated in the file and the demo should work out of the box.

**The files utilise text folds**; for ease of navigation it's highly recommended you configure your text editor to fold on {{{ and }}}.
E.g. in a .vimrc
```vim
set foldmethod=marker
set foldmarker={{{,}}}
```

### Dependencies
- python 2.7
- numpy 1.12.1
- theano developers version #354097d395789861a3120b4f3d99e7a919683e0c
    - 0.9 onwards is probably fine.
- h5py 2.6.0
- matplotlib 1.5.3

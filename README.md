# MIDI-VAE


## PREPARATION

- Install common libraries like
	numpy
	matplotlib
	pickle
	numpy
	progressbar
	sklearn
	scipy
	csv
	keras
	tensorflow
	theano	(some functions are only supported with theano because of recurrentshop)
- Make sure you have installed the following packages
	https://github.com/craffel/pretty-midi 
	https://github.com/farizrahman4u/recurrentshop/tree/master/recurrentshop
	https://github.com/nschloe/matplotlib2tikz

- Put your midi data in the folder 'data/original/'
- Group them into folders and name than for example 'style1', 'style2'
- Make sure you have at least 10 midi files per style, otherwise it can't form a test set
- Insert your style names into classes variable in settings.py
- Adjust parameters for training in settings.py
- Make sure you have all these files in the same folder

## TRAINING

- Run either vae_training.py to use the full MIDI-VAE model or
- Run any of the style classifiers pitch_classifier.py, velocity_classifer.py or instrument_classifer.py

The models will be stored in the automatically generated folder models/

## EVALUATION

- Change the model_name and epoch of your MIDI-VAE model that you want to evaluate
- Change the model names and epochs and weights for all the style classifiers
- Make sure you have set the same parameters as were used during training
- Run vae_evaluation.py

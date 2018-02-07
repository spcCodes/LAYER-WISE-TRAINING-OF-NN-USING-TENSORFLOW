# LAYER-WISE-TRAINING-OF-NN-USING-TENSORFLOW


This code deals with the layer wise training of Neural Network. The idea is to train a neural network on imbalance dataset but using a single layer of hidden nodes. This actually helps to deal with 2 problems
1) It deals with the vanishing gradient problem. Thus preventing the network to stop beacuse of very low weights while updation
2) it prevents overfitting.

Morever it helps in separability of the data in space as beacuse as we go for multiple layer of training, the separabilty between the points increases.

The entire project has been divided into 2 parts and has been written in Tensorflow. The work has been done for 21 UCI datasets. Here only pendigits datsets has been provided as it takes up a lot of size.

a) Data processing -pendigitsPreprocessing.py

b) Data training of the neural network and testing for the remaining data since the data has been divided into (70:30) -pendigitsNN.py

All the helper functions are as follows:
1) Neural Network functions for forward propagation,back propagation,cost function, updating parameters are written in NNFunctions.py
2) Separability measures (scatter matrix) for checking if the sperability has increased -seperabiltyCodes.py
3) Tensorflow utility functions to allow data to be divided into mini batches, etc... - tf_utils.py

The experiments has been succesfully conducted and the accuracy was seen to be increasing as the layer wise training of NN was performed
Below are the results that justify the above claim

1 LAYER TRAINING

NAME	  |      TRAINING ACCURACY   |	TESTING ACCURACY   |	SCATTER MATRIX INITIAL |	SCATTER MATRIX UPDATED
pendigits	|         99.4	      |      98.44	    |           23.033	        |          95.64

		2 LAYER TRAINING		
NAME	    |    TRAINING ACCURACY  |	TESTING ACCURACY   |	SCATTER MATRIX INITIAL  |	SCATTER MATRIX UPDATED

pendigits	|        99.93	       |      98.99	        |       23.03	        |         91.2



#!/usr/bin/env python

import numpy as np
import random

def genData(seed, size, L0, amp):
	# seed
	np.random.seed( seed ) 
	random.seed( seed )
	
	# generate arrays
	randomArray = (np.random.random_sample( L0 ) * 2 - 1) * amp
	zeroArray = np.zeros( size - L0 )

	# concatenate
	concatArray = np.r_[randomArray, zeroArray] 

	# shuffle
	random.shuffle( concatArray )
	
	# return 
	return concatArray



if __name__ == "__main__":
	import matplotlib.pyplot as pyplot
	seed = 1
	n = 1000
	k = 20
	pyplot.plot( genData(seed, n ,k ,10 ), '-x')
	pyplot.show()


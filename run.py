#!/usr/bin/env python
# -*- coding: utf-8 -*-

import genData as gd # this is used to generate x. indeed.
import matplotlib.pyplot as pyplot
import numpy as np
from sklearn import linear_model as lm # scikit learn (ridge, lasso)

import pyublas # This is always required when using the module that uses the PyUblas
import cpp.myTest as mt

#----------------------------------------------------------------------------------------
lowerBound = lambda n, k : (int)(2 * k * np.log(n))
l0norm = lambda xs: sum([1 for x in xs if x != 0])
CorrectPeak = lambda xs, ys: sum([1 for x in xs * ys if x != 0])
myRandom = lambda m: np.random.random_sample (m) * 2 - 1

#----------------------------------------------------------------------------------------
def dataDisp(w, l0, err, peak, n, k):
	fm = []
	for i in range(len(l0)):
		precision= 1.0 * peak[i] / (l0[i] + 1e-100)
		recall= 1.0 * peak[i] / k
		fmeasure = 2.* precision * recall /(precision + recall + 1e-100)
		fm.append(fmeasure)
		print "w = %10.5f \t l0 = %3d \t peak = %3d \t error = %10.5f, prec=%5.1f, recall = %5.1f, f= %5.1f" % \
			(w[i], l0[i], peak[i], err[i], precision * 100, recall * 100, fmeasure * 100)
	return fm


#----------------------------------------------------------------------------------------
def genData(seed, n, m, k, amp, noiseRate):
	x = gd.genData(seed, n, k, amp)
	np.random.seed(seed)
	A = myRandom(m * n).reshape(m, n)
	A = A / np.sqrt(sum(A * A))

	y = np.dot(A, x)
	xx = sum(x * x); yy = sum(y * y); 
	
	# add noise
	noise = myRandom(m) * amp * noiseRate
	snr = 10 * np.log(sum(y**2)/sum(noise**2))/np.log(10)

	print("noise:", sum(noise ** 2))
	print(snr, "dB")

	y = y + noise
	return [y, A, x]
		
#----------------------------------------------------------------------------------------
def LASSO(A, y, w): # scikit learn
	lasso = lm.Lasso(alpha = w)
	lasso.fit(A, y)
	return lasso.coef_

def LassoExec(A, y, wlist):
	xs=[]; l0s=[]; peaks=[]; errs=[]
	print "LASSO"
	for w in wlist:
		x = LASSO(A, y, w)
		l0 = l0norm(x)
		peak = CorrectPeak(x, x_truth)
		err = sum((y - np.dot(A, x))**2)
		xs.append(x); l0s.append(l0); peaks.append(peak); errs.append(err)
	return (xs, l0s, peaks, errs)

#----------------------------------------------------------------------------------------
def LOGP(A, y, w, it):
	Xi = np.dot(A.T, A)
	return mt.logp(A, Xi, y, w, it)

def LOGPExec(A, y, wlist):
	xs=[]; l0s=[]; peaks=[]; errs=[]
	for w in wlist:
		x = LOGP(A, y, w, 30)
		l0 = l0norm(x)
		peak = CorrectPeak(x, x_truth)
		err = sum((y - np.dot(A, x))**2)
		xs.append(x); l0s.append(l0); peaks.append(peak); errs.append(err)
	return (xs, l0s, peaks, errs)


#----------------------------------------------------------------------------------------
if __name__ == '__main__':
	# parameters
	seed = 1
	n = 100; k = 10
	amp = 10
	m = (int)(lowerBound(n,k))
	noiseRate = 0.1
	
	# Create Data
	[y, A, x_truth] = genData(seed, n, m, k, amp, noiseRate)

	wlist = [1e-6, 2e-6, 5e-6,
			 1e-5, 2e-5, 5e-5,
			 1e-4, 2e-4, 5e-4,
			 1e-3, 2e-3, 5e-3,
			 1e-2, 2e-2, 5e-2,
			 1e-1, 2e-1, 5e-1,
			 1e0,  2e0,  5e0,
			 1e1,  2e1,  5e1,
			 1e2,  2e2,  5e2,
			 1e3,  2e3,  5e3,
			 ]

	wlist = [w * 1 for w in wlist]

	# LASSO
	xs_1, l0s_1, ps_1, errs_1 = LassoExec(A, y, wlist)
	fm = dataDisp(wlist, l0s_1, errs_1, ps_1, n, k)

	# LOG PENALTY
	xs_2, l0s_2, ps_2, errs_2 = LOGPExec(A, y, wlist)
	fm = dataDisp(wlist, l0s_2, errs_2, ps_2, n, k)
	
	# plot 1
	pyplot.clf()
	pyplot.xlabel("Number of Nonzero elements ($|\bm{x}|_0$)", fontsize=20)
	pyplot.ylabel("Error ($|\bm{y} - \mathbf{A}\bm{x}_{\mathrm{estim}}|_2^2$)", fontsize=20)
	pyplot.title("")
	pyplot.xticks(fontsize=16)
	pyplot.yticks(fontsize=16)

	pyplot.plot(l0s_1, errs_1,  "-x", label = "LASSO");
	pyplot.plot(l0s_2, errs_2,  "-x", label="LOG Penalty"); 

	pyplot.legend(loc="upper right")
	pyplot.show()

	# plot 2
	pyplot.clf()
	pyplot.xlabel("$i$", fontsize=20)
	pyplot.ylabel("$x_i$", fontsize=20)
	pyplot.title("")
	pyplot.xticks(fontsize=16)
	pyplot.yticks(fontsize=16)

	pyplot.plot(x_truth, "o", label = "Ground Truth");
	pyplot.plot(xs_1[13], "o", label = "LASSO");
	pyplot.plot(xs_2[17], "o", label="LOG Penalty"); 

	pyplot.legend(loc="upper right")
	pyplot.show()




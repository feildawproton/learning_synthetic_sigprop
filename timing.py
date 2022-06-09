from matplotlib import image
from matplotlib import pyplot as plt
import os
import numpy as np
import tensorflow as tf

import ctypes
from ctypes import *

from tensorflow import keras
import math

import time
	
def get_calc_losses():
	dll = ctypes.CDLL("calc_losses.so", mode=ctypes.RTLD_GLOBAL)
	func = dll.calc_losses
	func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint, POINTER(c_float)]
	return func

_func = get_calc_losses()

model_name = os.path.join("models", "256w_4d")
model = keras.models.load_model(model_name)

n = 1
while n <= 7:
	numSamples = math.pow(10, n)
	numSamples = int(numSamples)
	print(numSamples)
	
	
	h_0_min	= 0.0			#in meters
	h_0_max 	= 2000.0		#in meters
	h_1_min 	= 0.0			#in meters
	h_1_max 	= 2000.0		#in meters
	h_2_min 	= 0.0			#in meters
	h_2_max 	= 2000.0		#in meters
	d_1_min 	= .01			#in kilometers
	d_1_max 	= 10.0			#in kilometers
	d_2_min	= .01			#in kilometers
	d_2_max	= 10.0			#in kilometers
	f_min		= 230000000.0		#in hertz
	f_max		= 9190000000.0		#in hertz

	h_0 	= np.random.rand(numSamples).astype(np.float32) * (h_0_max - h_0_min) + h_0_min
	h_1 	= np.random.rand(numSamples).astype(np.float32) * (h_1_max - h_1_min) + h_1_min
	h_2 	= np.random.rand(numSamples).astype(np.float32) * (h_2_max - h_2_min) + h_2_min
	d_1	= np.random.rand(numSamples).astype(np.float32) * (d_1_max - d_1_min) + d_1_min
	d_2	= np.random.rand(numSamples).astype(np.float32) * (d_2_max - d_2_min) + d_2_min
	freq	= np.random.rand(numSamples).astype(np.float32) * (f_max - f_min) + f_min
	Loss	= np.zeros(numSamples)


	X = np.concatenate((np.expand_dims(h_0, axis = 1), np.expand_dims(h_1, axis = 1), np.expand_dims(h_2, axis = 1), np.expand_dims(d_1, axis = 1), np.expand_dims(d_2, axis = 1),
	np.expand_dims(freq, axis = 1)), axis = 1)
	print(X.shape)

	ph_0	= h_0.ctypes.data_as(POINTER(c_float))
	ph_1	= h_1.ctypes.data_as(POINTER(c_float))
	ph_2	= h_2.ctypes.data_as(POINTER(c_float))
	pd_1	= d_1.ctypes.data_as(POINTER(c_float))
	pd_2	= d_2.ctypes.data_as(POINTER(c_float))
	pfreq	= freq.ctypes.data_as(POINTER(c_float))
	iNum	= int(numSamples)
	pLoss	= Loss.ctypes.data_as(POINTER(c_float))

	#time the cuda code 
	start = time.time()
	
	_func(ph_0, ph_1, ph_2, pd_1, pd_2, pfreq, iNum, pLoss)
	
	end = time.time()
	delta = end - start
	print("it took %f seconds for the cuda itm implementation" % delta)
	
	#time the trained tensorflow nn
	start = time.time()
	
	predictions = model.predict(X)
	
	end = time.time()
	delta = end - start
	print("it took %f seconds for the trained tensorflow nn" % delta)
	
	n += 1





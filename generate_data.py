import numpy as np
import ctypes
from ctypes import *
import pickle
import os

def get_calc_losses():
	dll = ctypes.CDLL("calc_losses_extern.so", mode=ctypes.RTLD_GLOBAL)
	func = dll.calc_losses
	func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint, POINTER(c_float)]
	return func
	

def generate_data(numSamples):
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
	
	#convert to ctypes
	ph_0	= h_0.ctypes.data_as(POINTER(c_float))
	ph_1	= h_1.ctypes.data_as(POINTER(c_float))
	ph_2	= h_2.ctypes.data_as(POINTER(c_float))
	pd_1	= d_1.ctypes.data_as(POINTER(c_float))
	pd_2	= d_2.ctypes.data_as(POINTER(c_float))
	pfreq	= freq.ctypes.data_as(POINTER(c_float))
	iNum	= int(numSamples)
	pLoss	= Loss.ctypes.data_as(POINTER(c_float))
	
	_calc_losses = get_calc_losses()
	
	_calc_losses(ph_0, ph_1, ph_2, pd_1, pd_2, pfreq, iNum, pLoss)
	
	#conver back to numpy type
	npLoss = np.fromiter(pLoss, dtype = np.float32, count = numSamples)
	
	data = {"h_0" : h_0, "h_1" : h_1, "h_2" : h_2, "d_1" : d_1, "d_2" : d_2, "freq" : freq, "Loss" : npLoss}
	
	return data
	
	
	
if __name__ == "__main__":
	numSamples = 1000000
	data = generate_data(numSamples = numSamples)
	print("calculated %i sampels" % (numSamples))
	
	dataset_path = os.path.join("generated_data", "synthetic_data.pkl")
	pickle.dump(data, open(dataset_path, "wb"))
	
	print("calculated data out as %s" % ("synthetic_data.pkl"))


	

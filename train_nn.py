from matplotlib import image
from matplotlib import pyplot as plt
import os
import numpy as np
import tensorflow as tf

import ctypes
from ctypes import *

def define_model(width = 512, drop_ratio = 0.1):
	#initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 1.0)
	in_layer = tf.keras.layers.Input(shape = 6)
	
	norm_1 = tf.keras.layers.BatchNormalization()(in_layer)
	
	layer_1 = tf.keras.layers.Dense(width, activation = "relu")(norm_1, training = True)
	drop_1 = tf.keras.layers.Dropout(drop_ratio)(layer_1)
	
	layer_2 = tf.keras.layers.Dense(width, activation = "relu")(drop_1)
	drop_2 = tf.keras.layers.Dropout(drop_ratio)(layer_2)
	
	layer_3 = tf.keras.layers.Dense(width, activation = "relu")(drop_2)
	drop_3 = tf.keras.layers.Dropout(drop_ratio)(layer_3)
	
	
	layer_4 = tf.keras.layers.Dense(width, activation = "relu")(drop_3)
	drop_4 = tf.keras.layers.Dropout(drop_ratio)(layer_4)
	'''
	layer_5 = tf.keras.layers.Dense(width, activation = "relu")(drop_4)
	drop_5 = tf.keras.layers.Dropout(drop_ratio)(layer_5)
	'''
	
	#output_3 = tf.keras.layers.Dense(1, activation = "relu")(drop_3)
	output_4 = tf.keras.layers.Dense(1, activation = "relu")(drop_4)
	#output_5 = tf.keras.layers.Dense(1, activation = "relu")(drop_5)

	return tf.keras.Model(inputs = in_layer, outputs = output_4)


def get_dummy():
	dll = ctypes.CDLL("calc_losses.so", mode=ctypes.RTLD_GLOBAL)
	func = dll.calc_losses_dummy
	func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint, POINTER(c_float)]
	return func

	
def get_calc_losses():
	dll = ctypes.CDLL("calc_losses.so", mode=ctypes.RTLD_GLOBAL)
	func = dll.calc_losses
	func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint, POINTER(c_float)]
	return func

numSamples = 10000

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


X = np.concatenate((np.expand_dims(h_0, axis = 1), np.expand_dims(h_1, axis = 1), np.expand_dims(h_2, axis = 1), np.expand_dims(d_1, axis = 1), np.expand_dims(d_2, axis = 1), np.expand_dims(freq, axis = 1)), axis = 1)
print(X.shape)

ph_0	= h_0.ctypes.data_as(POINTER(c_float))
ph_1	= h_1.ctypes.data_as(POINTER(c_float))
ph_2	= h_2.ctypes.data_as(POINTER(c_float))
pd_1	= d_1.ctypes.data_as(POINTER(c_float))
pd_2	= d_2.ctypes.data_as(POINTER(c_float))
pfreq	= freq.ctypes.data_as(POINTER(c_float))
iNum	= int(numSamples)
pLoss	= Loss.ctypes.data_as(POINTER(c_float))

'''	
_dummy_func = get_dummy()
_dummy_func(ph_0, ph_1, ph_2, pd_1, pd_2, pfreq, iNum, pLoss)
'''

_func = get_calc_losses()
_func(ph_0, ph_1, ph_2, pd_1, pd_2, pfreq, iNum, pLoss)


#conver back to numpy type
npLoss = np.fromiter(pLoss, dtype = np.float32, count = numSamples)

'''
why = h_0 * h_1 + h_2 * d_1 + d_2 + freq;

for i, elem in enumerate(why):
	print("why: %f, and npLoss: %f" % (elem, npLoss[i]))
'''

model = define_model()
model.summary()

#use default learning rate of 0.001 for Adam optimizer
model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Nadam(), metrics = ["mse"])
history = model.fit(X, npLoss, epochs = 10240, batch_size = 4096)

#now evaluate
print(X.shape)
predictions = model.predict(X)

#show stuff


fig, ax = plt.subplots(ncols = 2, nrows = 1)

ax[0].plot(history.history["mse"])
ax[1].scatter(npLoss, predictions)

plt.show()

for i, elem in enumerate(npLoss):
	print("Predicions: %f, and npLoss: %f" % (predictions[i], elem))

model_name = os.path.join("models", "512w_4d")
model.save(model_name)



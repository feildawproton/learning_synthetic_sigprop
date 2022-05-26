import numpy as np
import ctypes
from ctypes import *
import pickle
import os

from matplotlib import image
from matplotlib import pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split

def get_calc_losses():
	dll = ctypes.CDLL("calc_losses.so", mode=ctypes.RTLD_GLOBAL)
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

def define_model(dim_in = 6, dim_out = 1, width = 1024, drop_ratio = 0.5):
	#initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 1.0)

	in_layer = tf.keras.layers.Input(shape = (dim_in,))
	
	layer_1 = tf.keras.layers.Dense(width, activation = "tanh")(in_layer)
	drop_1 = tf.keras.layers.Dropout(drop_ratio)(layer_1)
	
	layer_2 = tf.keras.layers.Dense(width, activation = "tanh")(drop_1)
	drop_2 = tf.keras.layers.Dropout(drop_ratio)(layer_2)
	
	layer_3 = tf.keras.layers.Dense(width, activation = "tanh")(drop_2)
	drop_3 = tf.keras.layers.Dropout(drop_ratio)(layer_3)
	
	layer_4 = tf.keras.layers.Dense(width, activation = "tanh")(drop_3)
	drop_4 = tf.keras.layers.Dropout(drop_ratio)(layer_4)

	out_layer = tf.keras.layers.Dense(dim_out, activation = "linear")(drop_4)

	return tf.keras.Model(inputs = in_layer, outputs = out_layer)	

#could use data store in pickle
numSamples = 10000
data = generate_data(numSamples = numSamples)
print("calculated %i samples" % (numSamples))

#split train and valication
X = np.asarray([data["h_0"], data["h_1"], data["h_2"], data["d_1"], data["d_2"], data["freq"]])
X = np.transpose(X)
print(X.shape)
y = np.asarray(data["Loss"])
print(y.shape)
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

fig, ax = plt.subplots(nrows = 3, ncols = 2)
ax[0,0].scatter(X[:,0], y)
ax[1,0].scatter(X[:,1], y)
ax[2,0].scatter(X[:,2], y)
ax[0,1].scatter(X[:,3], y)
ax[1,1].scatter(X[:,4], y)
ax[2,1].scatter(X[:,5], y)

plt.show()

model = define_model()
model.summary()

#using Adam optimizer with default learning rate of 0.001
model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Nadam(), metrics = ["mse"])
epochs = 5000
batch_size = 1024
history = model.fit(X, y, epochs = epochs, batch_size = batch_size) 

numSamples2 = 10000
data2 = generate_data(numSamples = numSamples2)
print("calculated %i samples for validation" % (numSamples2))

X_val = np.asarray([data2["h_0"], data2["h_1"], data2["h_2"], data2["d_1"], data2["d_2"], data2["freq"]])
X_val = np.transpose(X_val)
print(X_val.shape)
y_val = np.asarray(data2["Loss"])
print(y_val.shape)

prediction = model.predict(X_val)
val_mse = np.square(np.subtract(y_val, prediction)).mean()
print("validation mse: %i" % (val_mse))

fig, ax = plt.subplots()
ax.plot(history.history["mse"])
plt.show()

model_name = os.path.join("models", "latest_model")
model.save(model_name)

print(prediction)

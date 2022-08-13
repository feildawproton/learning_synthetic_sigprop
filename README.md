# Generating synthetic sigprop samples
Generating individual synthetic signal propegation data samples.

A pretty darn fast implementation of some of the ITM signal propagation equations.

Used for training a fully connected NN for single instance inference.

## To compile the python compatible version for data generation

	$ nvcc -Xcompiler -fPIC -shared -o calc_losses.so calc_losses.cu

## Run python version of data generation

	$ python generate_data.py

data is saved in the generated_data folder.

The demo uses small number of examples to reduce generated data file size.  If not writing out to disk, use this code as an example of how to generate more data for RAM.

# Train NN
$ python train_nn.py

This script generates it's own data in the same way as generate_samples.py.  This is done because it is fast and I wanted to use more data than is good to have in a github file.

## Timing of individual sample inference
A sufficiently trained neural network is quite a bit slower that the cuda it implementation.  This is probably because:

- the cuda itm implementation is fast
- the computations don't have no spatial interactios.  This is not a 2d problem; this is many independent equations done in parallel.  So, we cannot take advantage of cnns or fourier transforms in this case.

## Copy compute overlap in data generation examination
Time with

	$ nsys profile --stats=true --force-overwrite=true -o some_report python generate_samples.py
	


no_overlap.png
          
tried_overlap.png
          

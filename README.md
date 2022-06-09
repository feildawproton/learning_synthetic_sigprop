# generating synthetic sigprop
Generating synthetic signal propegation data

A pretty darn fast implementation of some of the ITM signal propagation equations.  I will use this for transfer learning.

The loss calcualtions are in calc_losses.cu and the neccessary interface for it are int calc_losses.cuh.

Generate_data.cu is a work in progress and currently has an example of the usage of calc_losses().

## To compile the python compatible version for data generation
$ nvcc -Xcompiler -fPIC -shared -o calc_losses.so calc_losses_extern.cu

## Run python version of data generation
$ python generate_data.py

data is saved in the generated_data folder.

The demo uses small number of examples to reduce generated data file size.  If not writing out to disk, use this code as an example of how to generate more data for RAM.

# Train NN
$ python train_nn.py

This script generates it's own data in the same way as generate_data.py.  This is done because it is fast and I wanted to use more data than is good to have in a github file.

# Timing
A sufficiently trained neural network is quite a bit slower that the cuda it implementation.  This is probably because:

- the cuda itm implementation is fast
- the computations don't have no spatial interactios.  This is not a 2d problem; this is many independent equations done in parallel.  So, we cannot take advantage of cnns or fourier transforms in this case.

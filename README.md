# learning_synthetic_sigprop
Generating synthetic signal propegation data

A pretty darn fast implementation of some of the ITM signal propagation equations.  I will use this for transfer learning.

The loss calcualtions are in calc_losses.cu and the neccessary interface for it are int calc_losses.cuh.

Generate_data.cu is a work in progress and currently has an example of the usage of calc_losses().

## To compile and run the CUDA example
CUDA needed.  I don't have a makefile for this project so run:

$ nvcc generate_data.cu calc_losses.cu -o generate_data

If you want to run and look at performace metrics:

$ nsys nvprof ./generate_data

or

$ nvprof ./generate_data

## To compile the python compatible version
$ nvcc -Xcompiler -fPIC -shared -o cuda_sum.so cuda_sum.cu


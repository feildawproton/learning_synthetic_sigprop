# learning_synthetic_sigprop
Generating synthetic signal propegation data and learning it

A pretty darn fast implementation of some of the ITM signal propagation equations.  I will use this for transfer learning.

The come for calcualtions is in calc_losses.cu and the neccessary interface in it are int calc_losses.cuh.

Generate_data.cu is a work in progress and currently has an example of the usage of calc_losses().

##To compile and run
CUDA needed.  I don't have a makefile for this project so run:

nvcc generate_data.cu calc_losses.cu -o generate_data

If you want to run and look at performace metrics:

nsys nvprof ./generate_data

or

nvprof ./generate_data


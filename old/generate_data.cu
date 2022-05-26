#include <stdio.h>
#include <stdlib.h>
#include "calc_losses.cuh"

const unsigned numSamples = 10000000;

const float h_0_min	= 0.0;			//in meters
const float h_0_max 	= 2000.0;		//in meters
const float h_1_min 	= 0.0;			//in meters
const float h_1_max 	= 2000.0;		//in meters
const float h_2_min 	= 0.0;			//in meters
const float h_2_max 	= 2000.0;		//in meters
const float d_1_min 	= .01;			//in kilometers
const float d_1_max 	= 10.0;		//in kilometers
const float d_2_min	= .01;			//in kilometers
const float d_2_max	= 10.0;		//in kilometers
const float f_min	= 230000000.0;		//in hertz
const float f_max	= 9190000000.0;	//in hertz

int main()
{
	
	
	size_t mem_size = numSamples * sizeof(float);
	
	float *ph_0 = (float*)malloc(mem_size);
	float *ph_1 = (float*)malloc(mem_size);
	float *ph_2 = (float*)malloc(mem_size);
	float *pd_1 = (float*)malloc(mem_size);
	float *pd_2 = (float*)malloc(mem_size);
	float *pfreq = (float*)malloc(mem_size);
	
	float *pLoss = (float*)malloc(mem_size);	//calle will write to this
	
	time_t t;
	srand((unsigned) time(&t));
	for(unsigned i = 0; i < numSamples; i++)
	{
		ph_0[i] = ((float)rand() / (float)RAND_MAX) * (h_0_max - h_0_min) + h_0_min;
		ph_1[i] = ((float)rand() / (float)RAND_MAX) * (h_1_max - h_1_min) + h_1_min;
		ph_2[i] = ((float)rand() / (float)RAND_MAX) * (h_2_max - h_2_min) + h_2_min;
		pd_1[i] = ((float)rand() / (float)RAND_MAX) * (d_1_max - d_1_min) + d_1_min;
		pd_2[i] = ((float)rand() / (float)RAND_MAX) * (d_2_max - d_2_min) + d_2_min;
		pfreq[i] = ((float)rand() / (float)RAND_MAX) * (f_max - f_min) + f_min;
	}	
	
	calc_losses(ph_0, ph_1, ph_2, pd_1, pd_2, pfreq, numSamples, pLoss);
	
	/*
	for(unsigned i = 0; i < numSamples; i++)
	{
		printf("h_0 %f h_1 %f h_2 %f d_1 %f d_2 %f freq %f loss %f\n", ph_0[i], ph_1[i], ph_2[i], pd_1[i], pd_2[i], pfreq[i], pLoss[i]);
	}
	*/
	
	free(pLoss);
	free(ph_0);
	free(ph_1);
	free(ph_2);
	free(pd_1);
	free(pd_2);
	free(pfreq);
	
	return 0;
}











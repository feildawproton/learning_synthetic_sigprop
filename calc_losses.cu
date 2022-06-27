//#include "calc_losses.cuh" 
#include <math.h>
#include <stdlib.h>

//this is based off of Nicole Patterson's right up on Signal Propagation Equations for ITM
//this function (insert here) takes arrays a parameters, each index representing a single examples
//and returns a array of power losses, each eantry being the result for a single example

//this functions calculates a vaiable h
//as far as I can tell this is alse C_obs
//h_0 is the height of the obstruction in METERS
//h_1 is the height of the transmitter in METERS
//h_2 is the height of the receiver in METERS
//d_1 is the distance from the transmitter the obstruction point in KILOMETERS
//d_2 is the distance from the obstruction to the receiver in KILOMETERS
__device__ float calc_h(const float h_0, const float h_1, const float h_2, const float d_1, const float d_2)
{
	//h_ER is the height of the surface curvature at the obstruction point in meters
	float h_ER = (d_1 * d_2) / 16.944;
	float h = h_0 + h_ER - h_1 - ((h_2 - h_1) / (d_1 + d_2))*d_1;
	return h;
}

//lambda is the wavelenght
//freq is the frequency in Hertz (1/s)
__device__ float calc_lambda(float freq)
{
	return (299792458.0 / freq);
}


//v is the geometry factor
//h_0 is the height of the obstruction in METERS
//h_1 is the height of the transmitter in METERS
//h_2 is the height of the receiver in METERS
//d_1 is the distance from the transmitter the obstruction point in KILOMETERS
//d_2 is the distance from the obstruction to the receiver in KILOMETERS
//freq is the frequency in Hertz (1/s)
__device__ float calc_v(const float h_0, const float h_1, const float h_2, const float d_1, const float d_2, const float freq)
{
	//lambda is the wavelenght
	float lam = calc_lambda(freq);
	float h = calc_h(h_0, h_1, h_2, d_1, d_2);
	
	//v is the geometry factor
	//using sqrtf to ensure float version.  even though nvcc will perform it's own insertion
	float v = h * sqrtf((2.0*(d_1 + d_2)) / (lam * d_1 * d_2));
	return v;
}

//R_FR is 60% of the first Fresnel Zone radius
//d_1 is the distance from the transmitter the obstruction point in KILOMETERS
//d_2 is the distance from the obstruction to the receiver in KILOMETERS	
//freq is the frequency in Hertz (1/s)
__device__ float calc_R_FR(const float d_1, const float d_2, const float freq)
{
	float f_MHz = freq / 1000000.0;
	//using sqrtf to ensure float version.  even though nvcc will perform it's own insertion
	float R_FR = 0.6*(547.533*sqrtf((d_1*d_2) / (f_MHz*(d_1 + d_2) ) ) );
	return R_FR;
}

//this is the loss for a single example
//h_0 is the height of the obstruction in METERS
//h_1 is the height of the transmitter in METERS
//h_2 is the height of the receiver in METERS
//d_1 is the distance from the transmitter the obstruction point in KILOMETERS
//d_2 is the distance from the obstruction to the receiver in KILOMETERS
//freq is the frequency in Hertz (1/s)
__device__ float calc_loss(const float h_0, const float h_1, const float h_2, const float d_1, const float d_2, const float freq)
{
	float loss = 0.0;
	
	//v is the geometry factor
	float v = calc_v(h_0, h_1, h_2, d_1, d_2, freq);
	
	//accumulate loss from these various factors
	//FSPL loss occurs in the Fresnel Zone
	if(v <= -1.0)
	{
		//assumig the base is 10
		//using the float version instead of the default double version
		//hopefully nvcc makes the appropriate replacements
		float f_GHz = freq / 1000000000.0;
		loss += 20.0 * log10f(d_1 + d_2) + 20.0 * log10f(f_GHz) + 92.45;
	}
	//LOS loss occurs when the Freznel Zone is obstructed but the LOS line remains unobstructed
	if(v > 0.0 && v < 1.0)
	{
		//C_obs is the distance betweent he LOS and the obstruction
		float C_obs = calc_h(h_0, h_1, h_2, d_1, d_2);
		//R_FR is 60% of the first Fresnel Zone radius
		float R_FR = calc_R_FR(d_1, d_2, freq);
		loss += 6.0*(1.0 - (C_obs / R_FR));
	}
	//NLOS occurs whe the LOS is obstructed
	if(v >= 0.0)
	{
		//using log10f base 10
		//using float version of both log and sqrt
		loss += 6.9 + 20.0*log10f(sqrtf((v-0.1)*(v-0.1) + 1.0) + v - 0.1);
	}
	return loss;	
}

//This would be the function to interface with
//handles the cuda kernel
//assumes these arrays are on the cpu and creates copies on the gpu
//ph_0 is the array of heights of the obstruction in METERS
//ph_1 is the array of heights of the transmitter in METERS
//ph_2 is the array of heights of the receiver in METERS
//pd_1 is the array of distances from the transmitter the obstruction point in KILOMETERS
//pd_2 is the array of distances  from the obstruction to the receiver in KILOMETERS
//pfreq is the array of frequencies in Hertz (1/s)
//pLoss is the array used for returning loss values.  this function will try to write to it but will not allocate to it
//caller should handle memory for each of these arrays
//the index for each of these entries is the specific example
__global__ void calc_losses_kernel(const float* ph_0, const float* ph_1, const float* ph_2, 
			const float* pd_1, const float* pd_2, const float* pfreq, 
			const unsigned numSamples, const unsigned numThreads, float* pLoss)
{
	unsigned gindx = threadIdx.x + blockDim.x * blockIdx.x;	
	for(unsigned i = gindx; i < numSamples; i += numThreads)
	{
		pLoss[i] = calc_loss(ph_0[i], ph_1[i], ph_2[i], pd_1[i], pd_2[i], pfreq[i]);
	}
} 

__global__ void calc_losses_dummy_kernel(const float* ph_0, const float* ph_1, const float* ph_2, 
			const float* pd_1, const float* pd_2, const float* pfreq, 
			const unsigned numSamples, const unsigned numThreads, float* pLoss)
{
	unsigned gindx = threadIdx.x + blockDim.x * blockIdx.x;	
	for(unsigned i = gindx; i < numSamples; i += numThreads)
	{
		pLoss[i] = ph_0[i] * ph_1[i] + ph_2[i] * pd_1[i] + pd_2[i] + pfreq[i];
	}
} 

//This would be the function to interface with
//handles the cuda kernel
//assumes these arrays are on the cpe and creates copies on the gpu
//ph_0 is the array of heights of the obstruction in METERS
//ph_1 is the array of heights of the transmitter in METERS
//ph_2 is the array of heights of the receiver in METERS
//pd_1 is the array of distances from the transmitter the obstruction point in KILOMETERS
//pd_2 is the array of distances  from the obstruction to the receiver in KILOMETERS
//pfreq is the array of frequencies in Hertz (1/s)
//pLoss is the array used for returning loss values
//the index for each of these entries is the specific example
extern "C" {
void calc_losses(const float *ph_0, const float *ph_1, const float *ph_2, 
				const float *pd_1, const float *pd_2, const float *pfreq, 
				const unsigned numSamples, float *pLoss)
{
	size_t mem_size = numSamples * sizeof(float);
	//copy to gpu memory
	float *ph_0_dev, *ph_1_dev, *ph_2_dev, *pd_1_dev, *pd_2_dev, *pfreq_dev;
	
	cudaError_t status;
	status = cudaMalloc((void**)&ph_0_dev, mem_size);
	status = cudaMemcpy(ph_0_dev, ph_0, mem_size, cudaMemcpyHostToDevice); 
	
	status = cudaMalloc((void**)&ph_1_dev, mem_size);
	status = cudaMemcpy(ph_1_dev, ph_1, mem_size, cudaMemcpyHostToDevice); 

	status = cudaMalloc((void**)&ph_2_dev, mem_size);
	status = cudaMemcpy(ph_2_dev, ph_2, mem_size, cudaMemcpyHostToDevice); 
	
	status = cudaMalloc((void**)&pd_1_dev, mem_size);
	status = cudaMemcpy(pd_1_dev, pd_1, mem_size, cudaMemcpyHostToDevice); 
	
	status = cudaMalloc((void**)&pd_2_dev, mem_size);
	status = cudaMemcpy(pd_2_dev, pd_2, mem_size, cudaMemcpyHostToDevice); 
	
	status = cudaMalloc((void**)&pfreq_dev, mem_size);
	status = cudaMemcpy(pfreq_dev, pfreq, mem_size, cudaMemcpyHostToDevice); 
	
	//device id and properties
	int deviceID;
	cudaGetDevice(&deviceID);
	
	//gpu memory for results
	float* pLoss_dev;
	status = cudaMalloc((void**)&pLoss_dev, mem_size);
	
	//get properties to make best use of device
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, deviceID);
	
	//threads per block should be some multiple warpsize
	//or just set it to maxThreadsPerBlock
	//ThreadsPerBlock = props.maxThreadsPerBlock;
	unsigned ThreadsPerBlock = props.warpSize * 4;
	//blocks per grid should be some multiple of the number of streaming multiprocessors
	unsigned BlocksPerGrid = props.multiProcessorCount * 2;
	
	//then launch
	unsigned numThreads = BlocksPerGrid * ThreadsPerBlock;
	calc_losses_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(ph_0_dev, ph_1_dev, ph_2_dev, pd_1_dev, pd_2_dev, pfreq_dev, numSamples, numThreads, pLoss_dev);
	status = cudaGetLastError();
	
	//need to let work end before moving on
	status = cudaDeviceSynchronize();
	
	//copy back to host
	status = cudaMemcpy(pLoss, pLoss_dev, mem_size, cudaMemcpyDeviceToHost); 
	
	//don't need to the cuda memory anymore
	cudaFree(pLoss_dev);
	cudaFree(pfreq_dev);
	cudaFree(pd_2_dev);
	cudaFree(pd_1_dev);
	cudaFree(ph_2_dev);
	cudaFree(ph_1_dev);
	cudaFree(ph_0_dev);
} 
}

extern "C" {
void calc_losses_dummy(const float *ph_0, const float *ph_1, const float *ph_2, 
				const float *pd_1, const float *pd_2, const float *pfreq, 
				const unsigned numSamples, float *pLoss)
{
	size_t mem_size = numSamples * sizeof(float);
	//copy to gpu memory
	float *ph_0_dev, *ph_1_dev, *ph_2_dev, *pd_1_dev, *pd_2_dev, *pfreq_dev;
	
	cudaError_t status;
	status = cudaMalloc((void**)&ph_0_dev, mem_size);
	status = cudaMemcpy(ph_0_dev, ph_0, mem_size, cudaMemcpyHostToDevice); 
	
	status = cudaMalloc((void**)&ph_1_dev, mem_size);
	status = cudaMemcpy(ph_1_dev, ph_1, mem_size, cudaMemcpyHostToDevice); 

	status = cudaMalloc((void**)&ph_2_dev, mem_size);
	status = cudaMemcpy(ph_2_dev, ph_2, mem_size, cudaMemcpyHostToDevice); 
	
	status = cudaMalloc((void**)&pd_1_dev, mem_size);
	status = cudaMemcpy(pd_1_dev, pd_1, mem_size, cudaMemcpyHostToDevice); 
	
	status = cudaMalloc((void**)&pd_2_dev, mem_size);
	status = cudaMemcpy(pd_2_dev, pd_2, mem_size, cudaMemcpyHostToDevice); 
	
	status = cudaMalloc((void**)&pfreq_dev, mem_size);
	status = cudaMemcpy(pfreq_dev, pfreq, mem_size, cudaMemcpyHostToDevice); 
	
	//device id and properties
	int deviceID;
	cudaGetDevice(&deviceID);
	
	//gpu memory for results
	float* pLoss_dev;
	status = cudaMalloc((void**)&pLoss_dev, mem_size);
	
	//get properties to make best use of device
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, deviceID);
	
	//threads per block should be some multiple warpsize
	//or just set it to maxThreadsPerBlock
	//ThreadsPerBlock = props.maxThreadsPerBlock;
	unsigned ThreadsPerBlock = props.warpSize * 4;
	//blocks per grid should be some multiple of the number of streaming multiprocessors
	unsigned BlocksPerGrid = props.multiProcessorCount * 2;
	
	//then launch
	unsigned numThreads = BlocksPerGrid * ThreadsPerBlock;
	calc_losses_dummy_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(ph_0_dev, ph_1_dev, ph_2_dev, pd_1_dev, pd_2_dev, pfreq_dev, numSamples, numThreads, pLoss_dev);
	status = cudaGetLastError();
	
	//need to let work end before moving on
	status = cudaDeviceSynchronize();
	
	//copy back to host
	status = cudaMemcpy(pLoss, pLoss_dev, mem_size, cudaMemcpyDeviceToHost); 
	
	//don't need to the cuda memory anymore
	cudaFree(pLoss_dev);
	cudaFree(pfreq_dev);
	cudaFree(pd_2_dev);
	cudaFree(pd_1_dev);
	cudaFree(ph_2_dev);
	cudaFree(ph_1_dev);
	cudaFree(ph_0_dev);
} 
}

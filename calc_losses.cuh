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
void calc_losses(const float *ph_0, const float *ph_1, const float *ph_2, 
				const float *pd_1, const float *pd_2, const float *pfreq, 
				const unsigned numSamples, float *pLoss);

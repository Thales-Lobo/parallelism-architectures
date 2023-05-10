/*********************************************************************************/
/* Matrix product program for a multi-core CPU and for a many-core GPU           */
/* S. Vialle - November 2021                                                     */
/*********************************************************************************/

#ifndef __MATPROD_GPUOP__
#define __MATPROD_GPUOP__

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h> 
#include <cuda_runtime.h>


#define CHECK_CUDA_SUCCESS(exp,msg)   {if ((exp) != cudaSuccess) {\
                                         fprintf(stderr,"Error on CUDA operation (%s)\n",msg);\
                                         exit(EXIT_FAILURE);}\
                                      }

#define CHECK_CUBLAS_SUCCESS(exp,msg)   {if ((exp) != CUBLAS_STATUS_SUCCESS) {\
                                         fprintf(stderr,"Error on CUBLAS operation (%s)\n",msg);\
                                         exit(EXIT_FAILURE);}\
                                      }


void gpuInit(void);
void gpuFinalize(void);
void gpuSetDataOnGPU(void);
void gpuGetResultOnCPU(void);
void gpuProduct(gkid_t kid);

#endif


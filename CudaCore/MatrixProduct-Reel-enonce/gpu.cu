/*********************************************************************************/
/* Matrix product program for a multi-core CPU and for a many-core GPU           */
/* S. Vialle - November 2021                                                     */
/*********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "main.h"
#include "gpu.h"


/*-------------------------------------------------------------------------------*/
/* GPU symbols and global vars                                                   */
/*-------------------------------------------------------------------------------*/
// Symbols used by all kernels
__device__ T_complex GPU_A[2][SIZE][SIZE];
__device__ T_complex GPU_B[2][SIZE][SIZE];
__device__ T_complex GPU_C[2][SIZE][SIZE];

// New Symbol and vars to call Cublas lib.
__device__ T_complex GPU_Ctmp[2][SIZE][SIZE];   // New matrix buffer

T_complex *AdrGPU_A = NULL;                  // Adresses of the symbols
T_complex *AdrGPU_B = NULL;
T_complex *AdrGPU_C = NULL;
T_complex *AdrGPU_Ctmp = NULL; 

cublasHandle_t cublasHandle;              // Handle on the Cublas lib.


/*-------------------------------------------------------------------------------*/
/* Init and finalize the GPU device.                                             */
/*-------------------------------------------------------------------------------*/
void gpuInit(void)
{
  cuInit(0);
  
  // Extract address of GPU matrix "symbols"
  CHECK_CUDA_SUCCESS(cudaGetSymbolAddress((void **)&AdrGPU_A,GPU_A),"GPU_A adr extraction");
  CHECK_CUDA_SUCCESS(cudaGetSymbolAddress((void **)&AdrGPU_B,GPU_B),"GPU_B adr extraction");
  CHECK_CUDA_SUCCESS(cudaGetSymbolAddress((void **)&AdrGPU_C,GPU_C),"GPU_C adr extraction");
  CHECK_CUDA_SUCCESS(cudaGetSymbolAddress((void **)&AdrGPU_Ctmp,GPU_Ctmp),"GPU_Ctmp adr extraction");
  
  // Turn CPU arrays A, B and C into "pinned" memory areas
  CHECK_CUDA_SUCCESS(cudaHostRegister(A,2*SIZE*SIZE*sizeof(T_complex),
                                      cudaHostRegisterPortable),
                     "Turning into pinned memory the A CPU array");
  CHECK_CUDA_SUCCESS(cudaHostRegister(B,2*SIZE*SIZE*sizeof(T_complex),
                                      cudaHostRegisterPortable),
                     "Turning into pinned memory the B CPU array");
  CHECK_CUDA_SUCCESS(cudaHostRegister(C,2*SIZE*SIZE*sizeof(T_complex),
                                      cudaHostRegisterPortable),
                     "Turning into pinned memory the C CPU array");
  
  // Initialize CUBLAS lib usage
  CHECK_CUBLAS_SUCCESS(cublasCreate(&cublasHandle), "Init of the CUBLAS lib handle"); 
}


void gpuFinalize(void)
{
  // Turn "pinned" CPU arrays into std array
  CHECK_CUDA_SUCCESS(cudaHostUnregister(A),
                     "Turning into std memory the A CPU array");
  CHECK_CUDA_SUCCESS(cudaHostUnregister(B),
                     "Turning into std memory the B CPU array");
  CHECK_CUDA_SUCCESS(cudaHostUnregister(C),
                     "Turning into std memory the C CPU array");

  // Free CUBLAS lib usage
  CHECK_CUBLAS_SUCCESS(cublasDestroy(cublasHandle), "Free the CUBLAS lib");
}


/*-------------------------------------------------------------------------------*/
/* Transfer of CPU input data into GPU symbols                                   */
/*-------------------------------------------------------------------------------*/
void gpuSetDataOnGPU(void)
{
  // Set GPU_A symbol
  CHECK_CUDA_SUCCESS(cudaMemcpyToSymbol(GPU_A, A, 2*SIZE*SIZE*sizeof(T_complex), 0, cudaMemcpyHostToDevice),
                     "Transfer A-->GPU_A");

  // Set GPU_B symbol
  CHECK_CUDA_SUCCESS(cudaMemcpyToSymbol(GPU_B, B, 2*SIZE*SIZE*sizeof(T_complex), 0, cudaMemcpyHostToDevice),
                     "Transfer B-->GPU_B");
}


/*-------------------------------------------------------------------------------*/
/* Transfer of GPU results into CPU array                                        */
/*-------------------------------------------------------------------------------*/
void gpuGetResultOnCPU(void)
{
  // Get GPU_C symbol
  CHECK_CUDA_SUCCESS(cudaMemcpyFromSymbol(C, GPU_C, 2*SIZE*SIZE*sizeof(T_complex), 0, cudaMemcpyDeviceToHost),
                     "Transfer GPU_C-->C");
}


/*-------------------------------------------------------------------------------*/
/* Transposition kernel using global memory and registers.                       */
/*-------------------------------------------------------------------------------*/
__global__ void TransposeKernel_v0(T_complex *MT, T_complex *M, int mLig, int nCol)
{
 int lig = threadIdx.y + blockIdx.y*BLOCK_SIZE_XY_KT0;
 int col = threadIdx.x + blockIdx.x*BLOCK_SIZE_XY_KT0;
 
 if (lig < mLig && col < nCol)
   MT[col*mLig + lig] = M[lig*nCol + col];
}


/*-------------------------------------------------------------------------------*/
/* Small matrix product on the local GPU - 1D & generic matrix size              */
/*-------------------------------------------------------------------------------*/
__global__ void MatrixProductKernel_v0(void)
{
  // Index computations
  int row = blockIdx.y;
  int col = threadIdx.x + blockIdx.x * BLOCK_SIZE_X_K0;

  // Matrix product computation
  if (row < SIZE && col < SIZE) {
    T_complex real_res = 0.0;
    T_complex imaginary_res = 0.0;
    for (int k = 0; k < SIZE; k++) {
      real_res += (GPU_A[0][row][k] * GPU_B[0][k][col]) - (GPU_A[1][row][k] * GPU_B[1][k][col]);
      imaginary_res += (GPU_A[0][row][k] * GPU_B[1][k][col]) + (GPU_A[1][row][k] * GPU_B[0][k][col]);
    }

    GPU_C[0][row][col] = real_res;
    GPU_C[1][row][col] = imaginary_res;
  }
}


/*-------------------------------------------------------------------------------*/
/* Small matrix product on the local GPU - 2D & generic matrix size              */
/*-------------------------------------------------------------------------------*/
__global__ void MatrixProductKernel_v1(void)
{
  // Index computations
  int row = threadIdx.y + blockIdx.y * BLOCK_SIZE_Y_K1;
  int col = threadIdx.x + blockIdx.x * BLOCK_SIZE_X_K1;

  // Matrix product computation
  if (row < SIZE && col < SIZE) {
    T_complex real_res = 0.0;
    T_complex imaginary_res = 0.0;
    for (int k = 0; k < SIZE; k++) {
      real_res += (GPU_A[0][row][k] * GPU_B[0][k][col]) - (GPU_A[1][row][k] * GPU_B[1][k][col]);
      imaginary_res += (GPU_A[0][row][k] * GPU_B[1][k][col]) + (GPU_A[1][row][k] * GPU_B[0][k][col]);
    }

    GPU_C[0][row][col] = real_res;
    GPU_C[1][row][col] = imaginary_res;
  }
}

__global__ void MatrixProductKernel_v2(void)
{
  // Index computations
  int row = threadIdx.y + blockIdx.y * BLOCK_SIZE_XY_K2;
  int col = threadIdx.x + blockIdx.x * BLOCK_SIZE_XY_K2;

  int bloc_row = threadIdx.y;
  int bloc_col = threadIdx.x;

  T_complex real_res = {0.0};
  T_complex imaginary_res = {0.0};

  // Shared memory
  __shared__ T_complex sharedA[2][BLOCK_SIZE_XY_K2][BLOCK_SIZE_XY_K2];
  __shared__ T_complex sharedB[2][BLOCK_SIZE_XY_K2][BLOCK_SIZE_XY_K2];

  for (int bloc = 0; bloc < SIZE / BLOCK_SIZE_XY_K2 + 1; bloc++) {
    if ((row < SIZE) && (bloc_col + bloc * BLOCK_SIZE_XY_K2 < SIZE)) {
      sharedA[0][bloc_row][bloc_col] = GPU_A[0][row][bloc_col + bloc * BLOCK_SIZE_XY_K2];
      sharedA[1][bloc_row][bloc_col] = GPU_A[1][row][bloc_col + bloc * BLOCK_SIZE_XY_K2];
    }
    else {
      sharedA[0][bloc_row][bloc_col] = 0.0; 
      sharedA[1][bloc_row][bloc_col] = 0.0; 
    } 
    if ((col < SIZE) && (bloc_row + bloc * BLOCK_SIZE_XY_K2 < SIZE)) { 
      sharedB[0][bloc_row][bloc_col] = GPU_B[0][bloc_row + bloc * BLOCK_SIZE_XY_K2][col];
      sharedB[1][bloc_row][bloc_col] = GPU_B[1][bloc_row + bloc * BLOCK_SIZE_XY_K2][col];
    }
    else {
      sharedB[0][bloc_row][bloc_col] = 0.0;
      sharedB[1][bloc_row][bloc_col] = 0.0;
    }
    __syncthreads();
    
    // Matrix product computation
    for (int k = 0; k < BLOCK_SIZE_XY_K2; k++) {
      real_res += (sharedA[0][bloc_row][k] * sharedB[0][k][bloc_col]) - (sharedA[1][bloc_row][k] * sharedB[1][k][bloc_col]);
      imaginary_res += (sharedA[0][bloc_row][k] * sharedB[1][k][bloc_col]) + (sharedA[1][bloc_row][k] * sharedB[0][k][bloc_col]);
    }
  
    __syncthreads();
  }

  if(row < SIZE && col < SIZE) {
    GPU_C[0][row][col] = real_res;
    GPU_C[1][row][col] = imaginary_res;
  }
}


/*-------------------------------------------------------------------------------*/
/* Small matrix product on the local GPU.                                        */
/*-------------------------------------------------------------------------------*/
void gpuProduct(gkid_t kid)
{
 dim3 Dg = {0,0,0};   // Grid descriptor
 dim3 Db = {0,0,0};   // Block descriptor
 
 //T_complex alpha;        // When using CUBLAS
 //T_complex beta;         // When using CUBLAS

 switch(kid) {

 case GK0 : // Kernel v0 - 1D kernel using only resgisters and cache with generic matrix size
   // - init the grid of blocs
   // block
   Db.x = BLOCK_SIZE_X_K0;
   Db.y = 1;
   Db.z = 1;
   // grid
   Dg.x = SIZE % BLOCK_SIZE_X_K0 == 0 ? SIZE / BLOCK_SIZE_X_K0 : SIZE / BLOCK_SIZE_X_K0 + 1; 
   Dg.y = SIZE;
   Dg.z = 1;
   // - run the Grid of Blocs of threads
   MatrixProductKernel_v0<<<Dg,Db>>>();
   break;

 case GK1 : // kernel v1 : 2D kernel using only registers and cache with generic matrix size
   Db.x = BLOCK_SIZE_X_K1;
   Db.y = BLOCK_SIZE_Y_K1;
   Db.z = 1;
   // grid
   Dg.x = SIZE % BLOCK_SIZE_X_K1 == 0 ? SIZE / BLOCK_SIZE_X_K1 : SIZE / BLOCK_SIZE_X_K1 + 1; 
   Dg.y = SIZE % BLOCK_SIZE_Y_K1 == 0 ? SIZE / BLOCK_SIZE_Y_K1 : SIZE / BLOCK_SIZE_Y_K1 + 1;
   Dg.z = 1;
   // - run the Grid of Blocs of threads
   MatrixProductKernel_v1<<<Dg,Db>>>();
   break;

 case GK2 : // kernel v2 : 2D kernel using the shared memories with generic matrix size
  // block
   Db.x = BLOCK_SIZE_XY_K2;
   Db.y = BLOCK_SIZE_XY_K2;
   Db.z = 1;
   // grid
   Dg.x = SIZE % BLOCK_SIZE_XY_K2 == 0 ? SIZE / BLOCK_SIZE_XY_K2 : SIZE / BLOCK_SIZE_XY_K2 + 1;
   Dg.y = SIZE % BLOCK_SIZE_XY_K2 == 0 ? SIZE / BLOCK_SIZE_XY_K2 : SIZE / BLOCK_SIZE_XY_K2 + 1;
   Dg.z = 1;
   MatrixProductKernel_v2<<<Dg,Db>>>();
   break;
  
 case GK3 : //--
   break;

 case GK4 : //--
   break;
   
 case GK5 : //--
   break;

 case GK6 : //--
   break;

 case GK7 : //--
   break;

 case GK8 : //--
   break;

 default :
   fprintf(stderr,"Unknown GPU kernel!");
   exit(EXIT_FAILURE);
 } // End of switch
}





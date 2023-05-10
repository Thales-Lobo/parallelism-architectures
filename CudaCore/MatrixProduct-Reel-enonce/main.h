/*********************************************************************************/
/* Matrix product program for a multi-core CPU and for a many-core GPU           */
/* S. Vialle - November 2021                                                     */
/*********************************************************************************/

#ifndef __MATPROD_MAIN__
#define __MATPROD_MAIN__


/*-------------------------------------------------------------------------------*/
/* CONSTANTS.                                                                    */
/*-------------------------------------------------------------------------------*/

// Matrix size (side of the 3 matrixes)
#define SIZE              1024        // To debug
//#define SIZE              1027        // To debug
//#define SIZE              2048        // To benchmark
//define SIZE              4096        // To benchmark


// Constants for run configurations
#define DEFAULT_NB_THREADS  1          // Constant for OpenMP configuration
#define DEFAULT_ONGPUFLAG   1          // Constant for computation mode configuration
#define DEFAULT_CPUKID      CK0        // Constant for CPU Kernel config
#define DEFAULT_GPUKID      GK0        // Constant for GPU Kernel config

// Block sizes
#define BLOCK_SIZE_X_K0     512
#define BLOCK_SIZE_X_K1     32
#define BLOCK_SIZE_Y_K1     32
#define BLOCK_SIZE_XY_K2    32
#define BLOCK_SIZE_XY_K3    32

#define BLOCK_SIZE_XY_KT0   32
#define BLOCK_SIZE_XY_KT1   32


/*-------------------------------------------------------------------------------*/
/* Floating point datatype and op                                                */
/*-------------------------------------------------------------------------------*/
#ifdef DP
typedef double T_complex;
#define CBLAS_GEMM cblas_dgemm
#define CBLAS_GEAM cblas_dgeam
#define CUBLAS_GEMM cublasDgemm
#define CUBLAS_GEAM cublasDgeam
#define T_COMPLEX_TEXT "doubles"
#define T_CUBLAS_real CUDA_R_64F
#else
typedef float T_complex;
#define CBLAS_GEMM cblas_sgemm
#define CBLAS_GEAM cblas_sgeam
#define CUBLAS_GEMM cublasSgemm
#define CUBLAS_GEAM cublasSgeam
#define T_COMPLEX_TEXT "floats"
#define T_CUBLAS_real CUDA_R_32F
#endif


/*-------------------------------------------------------------------------------*/
/* Enumerated type of the different kernels                                      */
/*-------------------------------------------------------------------------------*/
typedef enum _ckid_t {
   CK0 = 0,
   CK1,
   NB_OF_CPU_KERNELS
} ckid_t;


typedef enum _gkid_t {
   GK0 = 0,
   GK1,
   GK2,
   GK3,
   GK4,
   GK5,
   GK6,
   GK7,
   GK8,
   NB_OF_GPU_KERNELS
} gkid_t;


/*-------------------------------------------------------------------------------*/
/* Global variable declarations.                                                 */
/*-------------------------------------------------------------------------------*/

/* Matrixes: C = A.B                                                             */
/* We use the Transposed B matrix, in place of B, to improve cache memory usage. */
extern T_complex A[2][SIZE][SIZE];               /* Matrixes : C = A.B           */
extern T_complex B[2][SIZE][SIZE];               /* B Matrix.                    */
extern T_complex TB[2][SIZE][SIZE];
extern T_complex C[2][SIZE][SIZE];

/* Global variables to control OpenMP computations.                              */
extern int NbThreads;

/* Global vars to control computation on the GPU.                                */
extern int OnGPUFlag;
extern ckid_t CPUKernelId;
extern gkid_t GPUKernelId;

/* Result checking flag.                                                         */
extern int check_results;

/*-------------------------------------------------------------------------------*/
/* Global functions.                                                             */
/*-------------------------------------------------------------------------------*/
void Computation(double *dk, double *dt, double *dkt);
void cpuProduct(ckid_t kid);
int main(int argc, char *argv[]);


#endif

// END

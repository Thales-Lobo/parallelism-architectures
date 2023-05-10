/*********************************************************************************/
/* Matrix product program for a multi-core CPU and for a many-core GPU           */
/* S. Vialle - November 2021                                                     */
/*********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <complex.h>

#include <cblas-openblas64.h>
#include <omp.h>

#include "main.h"
#include "init.h"
#include "gpu.h"

/*-------------------------------------------------------------------------------*/
/* Global variable declarations.                                                 */
/*-------------------------------------------------------------------------------*/

/* Matrixes: C = A.B                                                             */
/* We use the Transposed B matrix, in place of B, to improve cache memory usage. */
T_complex A[2][SIZE][SIZE];                            /* A Matrix.                    */
T_complex B[2][SIZE][SIZE];                            /* B Matrix.                    */
T_complex TB[2][SIZE][SIZE];                           /* Transposed B Matrix.         */
T_complex C[2][SIZE][SIZE];                            /* C matrix (result matrix).    */

/* Global variables to control OpenMP computations.                              */
int NbThreads = -1;

/* Global vars to control computation on the GPU.                                */
int OnGPUFlag;
ckid_t CPUKernelId;
gkid_t GPUKernelId;

/* Result checking flag.                                                         */
int check_results = 1;

/*-------------------------------------------------------------------------------*/
/* Parallel computation: local computations and data circulations.               */
/*-------------------------------------------------------------------------------*/
void Computation(double *dk, double *dt, double *dkt)
{
 double t1, t2, t3, t4;            /* Time measures                             */
 
 // Run computation on the GPU on each node
 if (OnGPUFlag) {
 
   // Measure all transfer times
   t1 = omp_get_wtime();
   gpuSetDataOnGPU();
   t2 = omp_get_wtime();
   gpuProduct(GPUKernelId);
   gpuGetResultOnCPU();
   t3 = omp_get_wtime();
   
   gpuGetResultOnCPU();                // Not useful now: just for time measurement
   t4 = omp_get_wtime();
   
   *dt  = (t2 - t1) + (t4 - t3);
   *dkt = t3 - t1;
   *dk  = *dkt - *dt;

 // OR run the computation on the CPU on each node
 } else {
   t1 = omp_get_wtime();
   cpuProduct(CPUKernelId);
   t2 = omp_get_wtime();
   *dkt = t2 - t1;
   *dt  = 0.0;
   *dk  = *dkt;
 }
}


/*-------------------------------------------------------------------------------*/
/* Local matrix product: optimized code!                                         */
/*-------------------------------------------------------------------------------*/
void cpuProduct(ckid_t kid)
{
 switch(kid) {

 case CK0 :
   memset(C,0,2*SIZE*SIZE*sizeof(T_complex));
   #pragma omp parallel for
    for (int i = 0; i < SIZE; i++) {
      for (int k = 0; k < SIZE; k++) {
        for (int j = 0; j < SIZE; j++) {
            C[0][i][j] += (A[0][i][k] * B[0][k][j]) - (A[1][i][k] * B[1][k][j]);
            C[1][i][j] += (A[0][i][k] * B[1][k][j]) + (A[1][i][k] * B[0][k][j]);
        }
    }
}

   break;

 case CK1 :
   // BLAS kernel
   #pragma omp parallel
   {
     int reste = SIZE % omp_get_num_threads();
     int quotient = SIZE / omp_get_num_threads();
     int NbLig = quotient +
                 (omp_get_thread_num() < reste ? 1 : 0);
     int offsetLig = quotient*omp_get_thread_num() +
                     (omp_get_thread_num() < reste ? omp_get_thread_num() : reste);

    T_complex real_A[SIZE][SIZE];
    T_complex real_B[SIZE][SIZE];
    T_complex imaginary_A[SIZE][SIZE];
    T_complex imaginary_B[SIZE][SIZE];
    T_complex real_C_1[SIZE][SIZE];
    T_complex real_C_2[SIZE][SIZE];
    T_complex imaginary_C_1[SIZE][SIZE];
    T_complex imaginary_C_2[SIZE][SIZE];

    
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            real_A[i][j] = A[0][i][j];
            real_B[i][j] = B[0][i][j];         
            imaginary_A[i][j] = A[1][i][j];
            imaginary_B[i][j] = B[1][i][j];
        }
    }

     CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                NbLig, SIZE, SIZE,
                1.0, &real_A[offsetLig][0], SIZE,
                &real_B[0][0], SIZE,
                0.0, &real_C_1[offsetLig][0], SIZE);

     CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                NbLig, SIZE, SIZE,
                1.0, &imaginary_A[offsetLig][0], SIZE,
                &imaginary_B[0][0], SIZE,
                0.0, &real_C_2[offsetLig][0], SIZE);

     CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                NbLig, SIZE, SIZE,
                1.0, &real_A[offsetLig][0], SIZE,
                &imaginary_B[0][0], SIZE,
                0.0, &imaginary_C_1[offsetLig][0], SIZE);

     CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                NbLig, SIZE, SIZE,
                1.0, &imaginary_A[offsetLig][0], SIZE,
                &real_B[0][0], SIZE,
                0.0, &imaginary_C_2[offsetLig][0], SIZE);

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            C[0][i][j] = real_C_1[i][j] - real_C_2[i][j];
            C[1][i][j] = imaginary_C_1[i][j] - imaginary_C_2[i][j];
        }
    }
  }
   break;

 default :
   fprintf(stderr,"Unknown CPU kernel!");
   exit(EXIT_FAILURE);
   break;

 }
}


/*-------------------------------------------------------------------------------*/
/* Toplevel function.                                                            */
/*-------------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    double dk, dt, dkt;            /* Elapsed times ok kernel, transfers and k+t*/
    double gfkt, gfk;              /* Program performances to measure.          */
    double bwt;                    /* Bandwidth of the tranfers.                */

    /* Initialisations --------------------------------------------------------- */
    CommandLineParsing(argc,argv);                /* Cmd line parsing.           */
    LocalMatrixInit();                            /* Initialization of the data  */
    omp_set_num_threads(NbThreads);               /* Max nb of threads/node.     */
    if (OnGPUFlag)                                /* Init the GPU device.        */
      gpuInit();

    /* Matrix product computation ---------------------------------------------- */
    fprintf(stdout,"* Product of two square matrices of %s of size %dx%d %s: *\n",
            T_COMPLEX_TEXT,SIZE,SIZE,(OnGPUFlag ? "on GPU" : "on CPU"));
    if (OnGPUFlag) {
      fprintf(stdout,"- GPU kernel Id: %d\n", GPUKernelId);
    } else {
      fprintf(stdout,"- CPU kernel Id: %d\n", CPUKernelId);
      fprintf(stdout,"- Max number of OpenMP threads per process: %d\n", NbThreads);
    }
    fprintf(stdout,"- Parallel computation starts...\n");

    Computation(&dk,&dt,&dkt);                    /* Parallel Matrix product.    */

    /* Performance computation, results and performance printing --------------- */
    gfkt = (2.0*8.0*pow(SIZE,3))/dkt*1E-9;             /* Performance achieved.      */
    gfk  = (2.0*8.0*pow(SIZE,3))/dk*1E-9;
    bwt  = (3.0*8.0*SIZE*SIZE*sizeof(T_complex))/dt*1E-9;
    PrintResultsAndPerf(dk, dt, dkt,               /* Results and perf printing  */
                        gfk, gfkt, bwt, OnGPUFlag);

    if (OnGPUFlag)                                /* Finalize GPU device usage.  */
      gpuFinalize();

    if (check_results)
      CheckResults();
    /* End of the parallel program --------------------------------------------- */
    return(EXIT_SUCCESS);
}


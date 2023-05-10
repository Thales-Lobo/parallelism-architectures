/*********************************************************************************/
/* Matrix product program for a multi-core CPU and for a many-core GPU           */
/* S. Vialle - November 2021                                                     */
/*********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <cblas-openblas64.h>
#include <omp.h>

#include "main.h"
#include "init.h"


/*-------------------------------------------------------------------------------*/
/* Initialisation of local matrixes A, B and C                                   */
/* Each process initializes its local parts of matrixes: simulates a parallel    */
/* initialization from files on disks.                                           */
/*-------------------------------------------------------------------------------*/
void LocalMatrixInit(void)
{
 int i, j;                                /* Local matrix indexes                */

/* Initialization of the local matrix elements                                   */
 for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
       A[0][i][j] = (T_complex) (0.00001*i*SIZE + 0.000002*j); //Real part of A
       A[1][i][j] = (T_complex) (0.00003*i*SIZE + 0.000005*j); //Imaginary part of A
    }
 }
 for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
       B[0][i][j]  = (T_complex) (0.0001*i*SIZE + 0.0000003*j);  //Real part of B
       TB[0][j][i] = (T_complex) (0.0001*i*SIZE + 0.0000003*j);  //Real part of TB
       B[1][i][j]  = (T_complex) (0.0005*i*SIZE + 0.0000007*j);  //Imaginary part of B
       TB[1][j][i] = (T_complex) (0.0005*i*SIZE + 0.0000007*j);  //Imaginary part of B
    }
 }

 for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
       C[0][i][j] = 0.0;  //Real part of C
       C[1][i][j] = 0.0;  //Imaginary part of C
    }
  }  
}


/*-------------------------------------------------------------------------------*/
/* Command Line parsing.                                                         */
/*-------------------------------------------------------------------------------*/
void usage(int ExitCode, FILE *std)
{
 fprintf(std,"MatrixProduct usage: \n");
 fprintf(std,"\t [-h]: print this help\n");
 fprintf(std,"\t [-t <GPU(default)|CPU>]: run computations on target GPU or on target CPU\n");
 fprintf(std,"\t [-cpu-k <CPU kernel Id [0(default) - %d]>]\n",(NB_OF_CPU_KERNELS-1));
 fprintf(std,"\t [-cpu-nt <number of OpenMP threads> (default %d)]\n",DEFAULT_NB_THREADS);
 fprintf(std,"\t [-gpu-k <GPU kernel Id [0(default) - %d]>]\n",(NB_OF_GPU_KERNELS-1));
 fprintf(std,"\t [-no-check]: stops the results from being checked (suggested for performance measurements)\n");

 exit(ExitCode);
}


void CommandLineParsing(int argc, char *argv[])
{
 // Default init
 NbThreads = DEFAULT_NB_THREADS;
 OnGPUFlag = DEFAULT_ONGPUFLAG;
 CPUKernelId = DEFAULT_CPUKID;
 GPUKernelId = DEFAULT_GPUKID;

 // Init from the command line
 argc--; argv++;
 while (argc > 0) {
     if (strcmp(argv[0],"-t") == 0) {
       argc--; argv++;
       if (argc > 0) {
         if (strcmp(argv[0],"GPU") == 0) {
           OnGPUFlag = 1;
           argc--; argv++;
         } else if (strcmp(argv[0],"CPU") == 0) {
           OnGPUFlag = 0;
           argc--; argv++;
         } else {
           fprintf(stderr,"Error: unknown computation target '%s'!\n",argv[0]);
           exit(EXIT_FAILURE);
         }
       } else {
         usage(EXIT_FAILURE, stderr);
       }

     } else if (strcmp(argv[0],"-cpu-k") == 0) {
       argc--; argv++;
       if (argc > 0) {
         CPUKernelId = (ckid_t) atoi(argv[0]);
         argc--; argv++;
         if (CPUKernelId < 0 || CPUKernelId >= NB_OF_CPU_KERNELS) {
           fprintf(stderr,"Error: CPU kernel Id has to in [0 - %d]!\n",(NB_OF_CPU_KERNELS-1));
           exit(EXIT_FAILURE);
         }
       } else {
         usage(EXIT_FAILURE, stderr);
       }

     } else if (strcmp(argv[0],"-cpu-nt") == 0) {
       argc--; argv++;
       if (argc > 0) {
         NbThreads = atoi(argv[0]);
         argc--; argv++;
         if (NbThreads <= 0) {
           fprintf(stderr,"Error: number of thread has to be >= 1!\n");
           exit(EXIT_FAILURE);
         }
       } else {
         usage(EXIT_FAILURE, stderr);
       }

     } else if (strcmp(argv[0],"-gpu-k") == 0) {
       argc--; argv++;
       if (argc > 0) {
         GPUKernelId = (gkid_t) atoi(argv[0]);
         argc--; argv++;
         if (GPUKernelId < 0 || GPUKernelId >= NB_OF_GPU_KERNELS) {
           fprintf(stderr,"Error: GPU kernel Id has to in [0 - %d]!\n",(NB_OF_GPU_KERNELS-1));
           exit(EXIT_FAILURE);
         }
       } else {
         usage(EXIT_FAILURE, stderr);
       }

     } else if (strcmp(argv[0],"-no-check") == 0) {
       argc--; argv++;
       check_results = 0;

     } else if (strcmp(argv[0],"-h") == 0) {
       usage(EXIT_SUCCESS, stdout);
     } else {
       usage(EXIT_FAILURE, stderr);
     }
 }

 // Complementary inits
 openblas_set_num_threads(1);                    // Set OpenBLAS in sequential mode
}


/*-------------------------------------------------------------------------------*/
/* Print result of the parallel computation and performances                     */
/*-------------------------------------------------------------------------------*/
void PrintResultsAndPerf(double dk, double dt, double dkt,
                         double gfk, double gfkt, double bwt, int ongpu)
{
 //fprintf(stdout,"- Results:\n");
 fprintf(stdout,"\n- Examples of results:\n\t C[%d][%d] = %f + %fi\n",
         0,SIZE-1,(float) C[0][0][SIZE-1],(float) C[1][0][SIZE-1]);
 fprintf(stdout,"\t C[%d][%d] = %f + %fi\n",
         SIZE/2,SIZE/2,(float) C[0][SIZE/2][SIZE/2],(float) C[1][SIZE/2][SIZE/2]);
 fprintf(stdout,"\t C[%d][%d] = %f + %fi\n",
         SIZE-1,0,(float) C[0][SIZE-1][0],(float) C[1][SIZE-1][0]);

 fprintf(stdout,"\n- Performance:\n");
 if(ongpu) {
     fprintf(stdout,"\t Complete Matrix Product:\n");
     fprintf(stdout,"\t   - Elapsed time = %f (s)\n", (float) dkt);
     fprintf(stdout,"\t   - Gflops = %f \n", (float) gfkt);
     fprintf(stdout,"\t Kernel computation:\n");
     fprintf(stdout,"\t   - Elapsed time = %f (s)\n", (float) dk);
     fprintf(stdout,"\t   - Gflops = %f \n", (float) gfk);
     fprintf(stdout,"\t Data transfers:\n");
     fprintf(stdout,"\t   - Elapsed time = %f (s)\n", (float) dt);
     fprintf(stdout,"\t   - BW           = %f (GB/s)\n", (float) bwt);
 } else {
     fprintf(stdout,"\t Complete Matrix Product:\n");
     fprintf(stdout,"\t   - Elapsed time = %f (s)\n", (float) dkt);
     fprintf(stdout,"\t   - Gflops = %f \n", (float) gfkt);
 }
 
 fflush(stdout);

}

/*-------------------------------------------------------------------------------*/
/* Result checking                                                               */
/*-------------------------------------------------------------------------------*/
T_complex C_check[2][SIZE][SIZE];

T_complex real_A[SIZE][SIZE];
T_complex real_B[SIZE][SIZE];
T_complex imaginary_A[SIZE][SIZE];
T_complex imaginary_B[SIZE][SIZE];
T_complex real_C_check_1[SIZE][SIZE];
T_complex real_C_check_2[SIZE][SIZE];
T_complex imaginary_C_check_1[SIZE][SIZE];
T_complex imaginary_C_check_2[SIZE][SIZE];
 

// Different values for epsilon depending if we use float or double
#ifdef DP
#define EPSILON 1e-14
#else
#define EPSILON 1e-4
#endif

void CheckResults(void) {

    fprintf(stdout,"\n- Checking results (comparison with CPU BLAS):\n");
    // Recomputing the matrix product on CPU
    omp_set_num_threads(omp_get_max_threads());

    #pragma omp parallel
   {
     int reste = SIZE % omp_get_num_threads();
     int quotient = SIZE / omp_get_num_threads();
     int NbLig = quotient +
                 (omp_get_thread_num() < reste ? 1 : 0);
     int offsetLig = quotient*omp_get_thread_num() +
                     (omp_get_thread_num() < reste ? omp_get_thread_num() : reste);

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
                0.0, &real_C_check_1[offsetLig][0], SIZE);

     CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                NbLig, SIZE, SIZE,
                1.0, &imaginary_A[offsetLig][0], SIZE,
                &imaginary_B[0][0], SIZE,
                0.0, &real_C_check_2[offsetLig][0], SIZE);

     CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                NbLig, SIZE, SIZE,
                1.0, &real_A[offsetLig][0], SIZE,
                &imaginary_B[0][0], SIZE,
                0.0, &imaginary_C_check_1[offsetLig][0], SIZE);

     CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                NbLig, SIZE, SIZE,
                1.0, &imaginary_A[offsetLig][0], SIZE,
                &real_B[0][0], SIZE,
                0.0, &imaginary_C_check_2[offsetLig][0], SIZE);

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            C_check[0][i][j] = real_C_check_1[i][j] - real_C_check_2[i][j];
            C_check[1][i][j] = imaginary_C_check_1[i][j] + imaginary_C_check_2[i][j];
        }
    }
   }

   // Comparing the different results
   // - maximum difference
   double max_diff = 0.0;
   // - position with the largest relative difference
   int max_X = 0;
   int max_Y = 0;
   // - epsilon for relative differences
   double epsilon = EPSILON;
   // - number of cases where the or is too large
   int cases = 0;

   for(int i = 0; i < SIZE; ++i){
       for(int j = 0; j < SIZE; ++j){
           double diff = fabs(sqrt(pow(C[0][i][j], 2) + pow(C[1][i][j], 2)) - sqrt(pow(C_check[0][i][j], 2) + pow(C_check[1][i][j], 2))); //difference between results
           double standard = fabs(sqrt(pow(C_check[0][i][j], 2) + pow(C_check[1][i][j], 2)));

           // Checks if the difference is large relative to the expected result
           if (diff > standard*epsilon)
               ++cases; // Register the case
           if (standard > 0.0 && diff/standard > max_diff){ // Store the largest difference seen so far
                   max_diff = diff/standard;
                   max_X = i;
                   max_Y = j;
           }
       }
   }

   if(cases == 0){
       fprintf(stdout,"The results are correct for %s with a precision of %.5e.\n", T_COMPLEX_TEXT, epsilon);
       fprintf(stdout,"Maximum relative difference encountered: %.5e.\n", max_diff);
   } else {
       fprintf(stdout,"*** WARNING ***\n");
       fprintf(stdout,"The results are incorrect for %s with a precision of %.5e.\n", T_COMPLEX_TEXT, epsilon);
       fprintf(stdout,"Number of cell with imprecise results: %d\n", cases);
       fprintf(stdout,"Cell C[%d][%d] contained the largest relative difference of %.5e\n", max_X, max_Y, max_diff);
       fprintf(stdout,"Expected value: %15.15lf + %15.15lfi (Modulus = %15.15lf )\n", C_check[0][max_X][max_Y], C_check[1][max_X][max_Y], sqrt(pow(C_check[0][max_X][max_Y], 2) + pow(C_check[1][max_X][max_Y], 2)));
       fprintf(stdout,"Computed value: %15.15lf + %15.15lfi (Modulus = %15.15lf )\n", C[0][max_X][max_Y], C[1][max_X][max_Y], sqrt(pow(C[0][max_X][max_Y], 2) + pow(C[1][max_X][max_Y], 2)));
   }
}




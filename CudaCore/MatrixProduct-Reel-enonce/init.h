/*********************************************************************************/
/* Matrix product program for a multi-core CPU and for a many-core GPU           */
/* S. Vialle - November 2021                                                     */
/*********************************************************************************/

#ifndef __MATPROD_INIT__
#define __MATPROD_INIT__


void LocalMatrixInit(void);                      // Data init

void usage(int ExitCode, FILE *std);             // Cmd line parsing and usage
void CommandLineParsing(int argc, char *argv[]);

void PrintResultsAndPerf(double dk, double dt, double dkt,    // Res printing
                         double gfk, double gfkt, double bwt, int ongpu); 

void CheckResults(void); // Res checking

#endif

// END

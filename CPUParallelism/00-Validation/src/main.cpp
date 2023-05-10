#include <chrono>
#include <random>
#include <iostream>
#include <float.h>

int main(int argc, char* argv[]) {

    if (argc < 4) {
        std::cout << "Usage test.exe width height number_of_iteration" << std::endl;
        exit(1);
    }

    unsigned long int width = atoi(argv[1]);
    unsigned long int height = atoi(argv[2]);
    unsigned long int iter = atoi(argv[3]);

    // Initialize random seed
    srand (time(NULL));

    // Create data structures contiguous in memory
    // A is the input structure
    // S is the result structure
    float *A,*S;
    A = (float *) malloc(width*height * sizeof(float));
    S = (float *) malloc(width*height * sizeof(float));

    // Fill A with random values
    for (unsigned long i = 0; i < width*height; i++) {
        A[i] = (float)(rand() % 360 - 180.0);
    }

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;
    double min_duration = DBL_MAX;

    for (auto it = 0; it < iter; it++) {
        t0 = std::chrono::high_resolution_clock::now();

        // Actual algorithm to monitor
        for (unsigned long j = 0; j < (width*(height-1)); j++) {
            S[j] = A[j] + A[j+width];
        }

        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1-t0).count();
        if (duration < min_duration) min_duration = duration;
    }
    
    
    //std::cout << "width height : ns per point" << std::endl;
    float ops = width*(height-1);
    std::cout << width << ", " << height << ", " << (1e9 * min_duration/ops) << std::endl;

    // Free data structures
    free(A);
    free(S);

    return 0;
}

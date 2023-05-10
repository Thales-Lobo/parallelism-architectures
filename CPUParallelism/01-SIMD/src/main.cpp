#include <chrono>
#include <random>
#include <iostream>
#include <float.h>
#include <immintrin.h>

float sequentiel(float* A, float* B, float* C, float* S, unsigned long int size) {
    for (unsigned long int i = 0; i < size; i++) {
        S[i] = (A[i] + B[i] + C[i])/3;
    }
    return S[3];
}

float parallele(float* A, float* B, float* C, float* S, unsigned long int size) {
    __m512 three = _mm512_set1_ps(3);
    for (unsigned long i = 0; i < size; i+=16) {
        __m512 a = _mm512_loadu_ps(&A[i]);
        __m512 b = _mm512_loadu_ps(&B[i]);
        __m512 c = _mm512_loadu_ps(&C[i]);
        __m512 s = _mm512_add_ps(a, b);
        s = _mm512_add_ps(s, c);        
        s = _mm512_div_ps(s, three);
        _mm512_storeu_ps(&S[i],s);
    }
    return S[3];
}

int main(int argc, char* argv[]) {
    unsigned long int iter = atoi(argv[1]);

    /* initialize random seed: */

    srand (time(NULL));

    const unsigned long int size = atoi(argv[2])*atoi(argv[2]);

    // std::cout << iter << " " << size << std::endl;
    
    // Création des données de travail
    float * A,* B,* C,* S1,* S2;
    A = (float *) malloc(size * sizeof(float));
    B = (float *) malloc(size * sizeof(float));
    C = (float *) malloc(size * sizeof(float));
    S1 = (float *) malloc(size * sizeof(float));
    S2 = (float *) malloc(size * sizeof(float));


    for (unsigned long int i = 0; i < size; i++) {
        A[i] = (float)(rand() % 360 - 180.0);
        B[i] = (float)(rand() % 360 - 180.0);
        C[i] = (float)(rand() % 360 - 180.0);
    }

    /*** Validation ***/
    sequentiel(A,B,C,S1,size);
    parallele(A,B,C,S2,size);
    bool valide = false;
    for (unsigned long int i = 0; i < size; i++) {
        if(S1[i] == S2[i]) {
            valide = true;
        }
        else {
            valide = false;
            break;
        }
    }
    // std::cout << "Le résultat est " << std::boolalpha << valide << std::endl;

    

    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    double min_duration = DBL_MAX;
    double result = 0;
    for (auto it =0; it < iter; it++) {
        t0 = std::chrono::high_resolution_clock::now();
        result += sequentiel(A, B,C, S1, size);
        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1-t0).count();
        if (duration < min_duration) min_duration = duration;
    }

    auto seq_duration = (min_duration/size);
    

    min_duration = DBL_MAX;
    result = 0;
    for (auto it =0; it < iter; it++) {
        t0 = std::chrono::high_resolution_clock::now();
        result += parallele(A, B,C, S2, size);
        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1-t0).count();
        if (duration < min_duration) min_duration = duration;
    }

    
    std::cout << size << " " << seq_duration << " " << (min_duration/size) << " " << result/size << std::endl;
    
    // Libération de la mémoire : indispensable
    free(A);
    free(B);
    free(C);
    free(S1);
    free(S2);    

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char* argv[]) {
    int rank, size;
    double count = 0;
    long long i;
    long long samples;
    double x, y;
    double start_time, end_time;
    double pi;
    double sum = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    samples = 10000000;

#define MAX_THREADS 4

    if (rank == 0) {
        printf("Prueba para %1d procesos\n", size);
        printf("Prueba para %1d hilos\n", MAX_THREADS);
        printf("Samples, Estimación pi, Diferencia pi, Tiempo de ejecución\n");
    }

    for (int row = 0; row < 10; row++) {
        count = 0;

        start_time = MPI_Wtime();

        unsigned int seeds[MAX_THREADS];
        for (int j = 0; j < MAX_THREADS; j++) {
            seeds[j] = (unsigned int)(j + 1);
        }
        
#pragma omp parallel 
        {
            unsigned int seed = seeds[omp_get_thread_num()];
            srand(seed);

#pragma omp for  reduction (+:count) 
            for (i = 0; i < samples; ++i) {
                double x, y, z;
                x = ((double)rand()) / ((double)RAND_MAX);
                y = ((double)rand()) / ((double)RAND_MAX);
                z = x * x + y * y;
                if (z <= 1.0) {
                    ++count;
                }
            }
        }

        double local_pi = 4.0 * count / samples;
        MPI_Reduce(&local_pi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        end_time = MPI_Wtime(); 

        if (rank == 0) {
            double difference = 3.1415926 - pi;
            double execution_time = end_time - start_time;

            printf("%lld,%.7f,%.7f,%.7f\n", samples, pi, difference, execution_time);
        }

        samples *= 2;
    }

    MPI_Finalize();

    return 0;
}
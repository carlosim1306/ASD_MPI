#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char* argv[]) {
    int rank, size;
    unsigned long long count = 0;
    long long i;
    long long samples;
    double sum = 0.0;
    double pi;
    double x, y;
    double start_time, end_time, start_time1, end_time1;

#define MAX_THREADS 8

    samples = 10000000;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Prueba para %1d procesos\n", size);
        printf("Samples, Estimación pi, Diferencia pi, Tiempo de ejecución\n");
    }

    for (int row = 0; row < 10; row++) {
        count = 0;
        
        unsigned int seeds[MAX_THREADS];
        for (int j = 0; j< MAX_THREADS; j++){
    		seeds[j] = (unsigned int)(j+1);
        }

        start_time = omp_get_wtime();

#pragma omp parallel
    {
        unsigned int seed = seeds[omp_get_thread_num()];
        srand(seed);
        
#pragma omp for reduction(+:count)
		for (i = rank; i < samples; i += size) {
		    x = ((double)rand()) / ((double)RAND_MAX);
		    y = ((double)rand()) / ((double)RAND_MAX);
		    double z = x * x + y * y;
		    if (z <= 1.0) {
		        ++count;
		    }
		}
    }
	    double local_pi = 4.0 * count/samples;

        end_time = omp_get_wtime();      
        start_time1 = MPI_Wtime();

        MPI_Reduce(&local_pi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        end_time1 = MPI_Wtime();

        if (rank == 0) {
            double difference = 3.1415926 - pi;
            double execution_time = (end_time - start_time)+(end_time1 - start_time1);
            printf("%lld,%.7f,%.7f,%.7f\n", samples, pi, difference, execution_time);
        }

        // Incrementa el número de samples para la próxima iteración
        samples *= 2;
    }

    MPI_Finalize();

    return 0;
}

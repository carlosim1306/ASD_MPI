#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    unsigned short xi[3] = { 1, 2, 3 };
    long long i;
    long long num_steps = 10000000;
    double start_time, end_time, start_time1, end_time1;
    double step, pi;
    double sum = 0.0;

#define MAX_THREADS 8

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Número de procesos: %d\n", size);
        printf("num_steps, Estimación pi, Diferencia pi, Tiempo de ejecución\n");
    }

    for (int row = 0; row < 10; row++) {
        step = 1.0 / (double)num_steps;
        sum = 0;
        omp_set_num_threads(MAX_THREADS);
        
        start_time = omp_get_wtime();

#pragma omp parallel for shared(step) reduction(+:sum)
            for (i = 0; i < num_steps; i++) {
                double x = (i + 0.5) * step;
                sum += 4.0 / (1.0 + x * x);
            }

        double local_pi = step * sum;

        end_time = omp_get_wtime();      
        start_time1 = MPI_Wtime();

        MPI_Reduce(&local_pi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        end_time1 = MPI_Wtime();

        if (rank == 0) {
            double difference = 3.1415926 - pi;
            double execution_time = (end_time - start_time)+(end_time1 - start_time1);
            printf("%lld,%.7f,%.7f,%.7f\n", num_steps, pi, difference, execution_time);
        }

        num_steps *= 2;
    }

    MPI_Finalize();

    return 0;
}
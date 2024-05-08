#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char* argv[]) {
    int rank, size;
    unsigned short xi[3] = { 1, 2, 3 };
    unsigned long long count = 0;
    long long i;
    long long samples;
    double x, y;
    double start_time, end_time, start_time1, end_time1;
    long long num_steps = 10000000;
    double step, pi;
    double sum = 0.0;

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

        start_time = MPI_Wtime();

#pragma omp parallel for reduction(+:sum)
        for (i = rank; i < num_steps; i += size) {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }

        MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        end_time = MPI_Wtime();

        if (rank == 0) {
            pi *= step;
            double difference = 3.1415926 - pi;
            double execution_time = end_time - start_time;
            printf("%lld,%.7f,%.7f,%.7f\n", num_steps, pi, difference, execution_time);
        }

        // Incrementa el número de steps para la próxima iteración
        num_steps *= 2;
    }

    MPI_Finalize();

    return 0;
}

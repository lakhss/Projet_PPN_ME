#include <omp.h>
#include <iostream>

int main() {
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int n = omp_get_num_threads();

        #pragma omp critical
        std::cout << "Thread " << id << " / " << n << std::endl;
    }
}
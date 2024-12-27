#include <math.h>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <time.h>
#include <sys/time.h>

#define nx 384
#define ny 384
#define nz 384

#define a(i, j, k) a[((i) * ny + (j)) * nz + (k)]
#define tmp1(i, j, k) tmp1[((i) * ny + (j)) * nz + (k)]
#define tmp2(i, j, k) tmp2[((i) * ny + (j)) * nz + (k)]
#define tmp3(i, j, k) tmp3[((i) * nx + (j)) * ny + (k)]

#define Max(a, b) ((a) > (b) ? (a) : (b))
int ox1 = 32, oy1 = 16;
dim3 block1 = dim3((nz + ox1 - 1) / ox1, (ny + oy1 - 1) / oy1);
dim3 thread1 = dim3(ox1, oy1);

int ox2 = 32, oy2 = 16;
dim3 block2 = dim3((nz + ox2 - 1) / ox2, (nx + oy2 - 1) / oy2);
dim3 thread2 = dim3(ox2, oy2);

int ox3 = 32, oy3 = 16;
dim3 block3 = dim3((ny + ox3 - 1) / ox3, (nx + oy3 - 1) / oy3);
dim3 thread3 = dim3(ox3, oy3);

int ox_init = 8, oy_init = 8, oz_init = 8;
dim3 block_init = dim3((nz + ox_init - 1) / ox_init, (ny + oy_init - 1) / oy_init, (nx + oz_init - 1) / oz_init);
dim3 thread_init = dim3(ox_init, oy_init, oz_init);

int ox_revers = 8, oy_revers = 8, oz_revers = 8;
dim3 block_revers = dim3((ny + ox_revers - 1) / ox_revers, (nx + oy_revers - 1) / oy_revers, (nz + oz_revers - 1) / oz_revers);
dim3 thread_revers = dim3(ox_revers, oy_revers, oz_revers);

double maxeps = 0.01, eps;
int itmax = 100;

__global__ void init_parallel(double *a)
{
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int i = blockIdx.z * blockDim.z + threadIdx.z;
        if (i > -1 && i < nx)
                if (j > -1 && j < ny)
                        if (k > -1 && k < nz)
                                if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1) {
                                        a(i, j, k) = 10.0 * i / (nx - 1) + 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
                                } else {
                                        a(i, j, k) = 0;
                                }
}

__global__ void f1(double *a, int ii)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j > 0 && j < ny - 1)
        if (k > 0 && k < nz - 1)
            a(ii, j, k) = (a(ii - 1, j, k) + a(ii + 1, j, k)) / 2;
}

__global__ void f2(double *a, int jj)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < nx - 1)
        if (k > 0 && k < nz - 1)
            a(i, jj, k) = (a(i, jj - 1, k) + a(i, jj + 1, k)) / 2;
}

__global__ void f3(double *tmp3, int kk)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < nx - 1)
        if (j > 0 && j < ny - 1) {
            tmp3(kk, i, j) = (tmp3(kk - 1, i, j) + tmp3(kk + 1, i, j)) / 2;
        }
}

__global__ void f_cp(double *a, double *tmp1)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > -1 && i < nx)
        if (j > -1 && j < ny)
            if (k > -1 && k < nz) {
                tmp1(i, j, k) = a(i, j, k);
            }
}

__global__ void f_cp_k_i_j(double *a, double *tmp3)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > -1 && i < nx)
        if (j > -1 && j < ny)
            if (k > -1 && k < nz) {
                tmp3(k, i, j) = a(i, j, k);
            }
}

__global__ void f_cp_j_k_i(double *tmp3, double *a)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > -1 && i < nz)
        if (j > -1 && j < nx)
            if (k > -1 && k < ny) {
                a(j, k, i) = tmp3(i, j, k);
            }
}

__global__ void f4(double *a, double *tmp1, double *tmp2)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 0 && i < nx - 1)
        if (j > 0 && j < ny - 1)
            if (k > 0 && k < nz - 1) {
                tmp2(i, j, k) = fabs(a(i, j, k) - tmp1(i, j, k));
            }
}

double adi_parallel(double* a)
{
        for (int it = 1; it <= itmax; it++)
    {
        eps = 0;
        for (int ii = 1; ii < nx - 1; ii++) {

            f1 << <block1, thread1 >> > (a, ii);
            cudaDeviceSynchronize();

        }
        for (int jj = 1; jj < ny - 1; jj++) {

            f2 << <block2, thread2 >> > (a, jj);
            cudaDeviceSynchronize();

        }
        thrust::device_vector<double> diff1(nx * ny * nz);
                double* tmp1 = thrust::raw_pointer_cast(&diff1[0]);
        thrust::device_vector<double> diff2(nx * ny * nz);
        double* tmp2 = thrust::raw_pointer_cast(&diff2[0]);

        f_cp << <block_init, thread_init >> > (a, tmp1);
        cudaDeviceSynchronize();

        thrust::device_vector<double> diff3(nx * ny * nz);
        double* tmp3 = thrust::raw_pointer_cast(&diff3[0]);

        f_cp_k_i_j << <block_init, thread_init >> > (a, tmp3);
        cudaDeviceSynchronize();

        for (int kk = 1; kk < nz - 1; kk++) {

            f3 << <block3, thread3 >> > (tmp3, kk);
            cudaDeviceSynchronize();

        }

        f_cp_j_k_i << <block_revers, thread_revers >> > (tmp3, a);
        cudaDeviceSynchronize();

        f4 << <block_init, thread_init >> > (a, tmp1, tmp2);
        cudaDeviceSynchronize();

        eps = thrust::reduce(diff2.begin(), diff2.end(), 0.0, thrust::maximum<double>());
        cudaDeviceSynchronize();

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < maxeps)
            break;
    }
    cudaDeviceSynchronize();
    return eps;
}

void print_benchmark(struct timeval startt, struct timeval endt)
{
    printf(" ADI Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", itmax);
    printf(" Time in seconds =       %12.2lf\n", endt.tv_sec - startt.tv_sec + (endt.tv_usec - startt.tv_usec) * 0.000001);
    printf(" Operation type  =   double precision\n");
    printf(" Verification    =       %12s\n", (fabs(eps - 0.07249074) < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL"));

    printf(" END OF ADI Benchmark\n");
}

int main(int argc, char *argv[])
{
    double *a;
    struct timeval startt, endt;
    cudaMalloc((void**)&a, nx * ny * nz * sizeof(double));
    init_parallel << <block_init, thread_init >> > (a);
    gettimeofday(&startt, NULL);
    adi_parallel(a);
    gettimeofday(&endt, NULL);

    print_benchmark(startt, endt);

    cudaFree(a);

}

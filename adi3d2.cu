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

#define Max(a, b) ((a) > (b) ? (a) : (b))
int ox1 = 2, oy1 = 512;
dim3 block1 = dim3((nx + ox1 - 1) / ox1, (ny + oy1 - 1) / oy1);
dim3 thread1 = dim3(ox1, oy1);

int ox2 = 512, oy2 = 2;
dim3 block2 = dim3((nx + ox2 - 1) / ox2, (ny + oy2 - 1) / oy2);
dim3 thread2 = dim3(ox2, oy2);

int ox3 = 2, oy3 = 512;
dim3 block3 = dim3((nx + ox3 - 1) / ox3, (ny + oy3 - 1) / oy3);
dim3 thread3 = dim3(ox3, oy3);

int ox_init = 8, oy_init = 8, oz_init = 8;
dim3 block_init = dim3((nx + ox_init - 1) / ox_init, (ny + oy_init - 1) / oy_init, (nz + oz_init - 1) / oz_init);
dim3 thread_init = dim3(ox_init, oy_init, oz_init);

double maxeps = 0.01, eps;
int itmax = 100;

__global__ void init_parallel(double *a)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
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
    int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;   
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

__global__ void f3(double *a, int kk)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < nx - 1)
        if (j > 0 && j < ny - 1) {
            a(i, j, kk) = (a(i, j, kk - 1) + a(i, j, kk + 1)) / 2;
        }
}

__global__ void f_cp(double *a, double *tmp1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i > 0 && i < nx - 1)
        if (j > 0 && j < ny - 1)
            if (k > 0 && k < nz - 1) {
                tmp1(i, j, k) = a(i, j, k);
            }
}

__global__ void f4(double *a, double *tmp1, double *tmp2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

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

        for (int kk = 1; kk < nz - 1; kk++) {

		    f3 << <block3, thread3 >> > (a, kk);
            cudaDeviceSynchronize();

        }
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

void init_seq(double *a)
{
    int i, j, k;
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
            for (k = 0; k < nz; k++)
                if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1)
                    a(i, j, k) = 10.0 * i / (nx - 1) + 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
                else
                    a(i, j, k) = 0;
}

double adi_seq(double* a)
{
    int i, j, k;
    for (int it = 1; it <= itmax; it++)
    {
        eps = 0;        
        for (i = 1; i < nx - 1; i++)
            for (j = 1; j < ny - 1; j++)
                for (k = 1; k < nz - 1; k++)
                    a(i, j, k) = (a(i-1, j, k) + a(i+1, j, k)) / 2;

        for (i = 1; i < nx - 1; i++)
            for (j = 1; j < ny - 1; j++)
                for (k = 1; k < nz - 1; k++)
                    a(i, j, k) = (a(i, j-1, k) + a(i, j+1, k)) / 2;

        for (i = 1; i < nx - 1; i++)
            for (j = 1; j < ny - 1; j++)
                for (k = 1; k < nz - 1; k++)
                {
                    double tmp1 = (a(i, j, k-1) + a(i, j, k+1)) / 2;
                    double tmp2 = fabs(a(i, j, k) - tmp1);
                    eps = Max(eps, tmp2);
                    a(i, j, k) = tmp1;
                }

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < maxeps)
            break;
    }
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
    eps = adi_parallel(a);
    gettimeofday(&endt, NULL);
    print_benchmark(startt, endt);

    cudaFree(a);

    a = (double *) malloc(nx * ny * nz * sizeof(double));
    init_seq(a);
    gettimeofday(&startt, NULL);
    adi_seq(a);
    gettimeofday(&endt, NULL);

    print_benchmark(startt, endt);

    free(a);
}

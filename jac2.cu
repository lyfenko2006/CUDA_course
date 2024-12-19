#include <math.h>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <time.h>
#include <sys/time.h>

#define a(i, j, k) a[((i) * nn + (j)) * kk + (k)]
#define b(i, j, k) b[((i) * nn + (j)) * kk + (k)]
#define d(i, j, k) d[((i) * nn + (j)) * kk + (k)]

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define L 384
#define ITMAX 100

int ox = 8, oy = 8, oz = 8;
dim3 block = dim3((L + ox - 1) / ox, (L + oy - 1) / oy, (L + oz - 1) / oz);
dim3 thread = dim3(ox, oy, oz);

int i, j, k, it;
double eps;
double MAXEPS = 0.5f;

__global__ void function(int mm, int nn, int kk, double* a, double* b)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.z * blockDim.z + threadIdx.z;
	if (i > 0 && i < mm - 1)
		if (j > 0 && j < nn - 1)
			if (k > 0 && k < kk - 1)
				b(i, j, k) = (a(i - 1, j, k) + a(i + 1, j, k) + a(i, j - 1, k) + a(i, j + 1, k)
					+ a(i, j, k - 1) + a(i, j, k + 1)) / 6.;
}

__global__ void difference(int mm, int nn, int kk, double* a, double* b, double *d)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.z * blockDim.z + threadIdx.z;
	if (i > 0 && i < mm - 1)
		if (j > 0 && j < nn - 1)
			if (k > 0 && k < kk - 1)
				d(i, j, k) = fabs(a(i, j, k) - b(i, j, k));
}

__global__ void ab(int mm, int nn, int kk, double* a, double* b)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.z * blockDim.z + threadIdx.z;
	if (i > 0 && i < mm - 1)
		if (j > 0 && j < nn - 1)
			if (k > 0 && k < kk - 1)
				a(i, j, k) = b(i, j, k);
}

double jac_parallel(double* a, int mm, int nn, int kk, int itmax, double maxeps)
{
	double* b;

	cudaMalloc((void**)&b, mm * nn * kk * sizeof(double));
	
	for (it = 1; it <= itmax - 1; it++)
	{
		function << <block, thread >> > (mm, nn, kk, a, b);	
	    cudaDeviceSynchronize();
		eps = 0.;

		thrust::device_vector<double> diff(mm * nn * kk);
		double* ptrdiff = thrust::raw_pointer_cast(&diff[0]);
		difference << <block, thread >> > (mm, nn, kk, a, b, ptrdiff);
        cudaDeviceSynchronize();

		eps = thrust::reduce(diff.begin(), diff.end(), 0.0, thrust::maximum<double>());
		ab << <block, thread >> > (mm, nn, kk, a, b);
        cudaDeviceSynchronize();

		//if (TRACE && it % TRACE == 0)
			printf(" IT = %4i   EPS = %14.7E\n", it, eps);

		if (eps < maxeps)
			break;
	}
	cudaFree(b);

	return eps;
}

__global__ void initial(int mm, int nn, int kk, double* a)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.z * blockDim.z + threadIdx.z;
	//printf("%lf\n", a(i, j, k));
	if (i >= 0 && i < mm)
		if (j >= 0 && j < nn)
			if (k >= 0 && k < kk)
				if (i == 0 || j == 0 || k == 0 || i == mm - 1 || j == nn - 1 || k == kk - 1)
                   	 		a(i, j, k) = 0;
                		else
                    		a(i, j, k) = 4 + i + j + k;
}

void initial_seq(int mm, int nn, int kk, double *a)
{
    for (i = 0; i < mm; i++)
        for (j = 0; j < nn; j++)
            for (k = 0; k < kk; k++)
            {
                if (i == 0 || j == 0 || k == 0 || i == mm - 1 || j == nn - 1 || k == kk - 1)
                    a(i, j, k) = 0;
                else
                    a(i, j, k) = 4 + i + j + k;
            }
}

double jac_sequence(double* a, int mm, int nn, int kk, int itmax, double maxeps)
{
    double *b = (double*) malloc(nn * mm * kk * sizeof(double));
    double eps;

    /* iteration loop */
    for (it = 1; it <= ITMAX - 1; it++)
    {
        for (i = 1; i < mm - 1; i++)
            for (j = 1; j < nn - 1; j++)
                for (k = 1; k < kk - 1; k++)
                    b(i, j, k) = (a(i - 1, j, k) + a(i, j - 1, k) + a(i, j, k - 1) + a(i, j, k + 1) + a(i, j + 1, k) + a(i + 1, j, k)) / 6.0f;
        
        eps = 0.;
        
        for (i = 1; i < mm - 1; i++)
            for (j = 1; j < nn - 1; j++)
                for (k = 1; k < kk - 1; k++)
                {
                    double tmp = fabs(b(i, j, k) - a(i, j, k));
                    eps = Max(tmp, eps);
                    a(i, j, k) = b(i, j, k);
                }

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < MAXEPS)
            break;
    }
    free(b);
    return eps;
}

void print_benchmark(double eps, struct timeval startt, struct timeval endt)
{
    printf(" Jacobi3D Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", L, L, L);
    printf(" Iterations      =       %12d\n", ITMAX);
    printf(" Time in seconds =       %12.2lf\n", endt.tv_sec - startt.tv_sec + (endt.tv_usec - startt.tv_usec) * 0.000001);
    printf(" Operation type  =     floating point\n");
    printf(" Verification    =       %12s\n", (fabs(eps - 5.058044) < 1e-11 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    //printf(" Verification    =       %12s\n", (eps <= 5.058044  ? "SUCCESSFUL" : "UNSUCCESSFUL"));

    printf(" END OF Jacobi3D Benchmark\n");
}

int main(void)
{
	double *a;
    double eps;

    struct timeval startt, endt;

	cudaMalloc((void**)&a, L * L * L * sizeof(double));
	initial << <block, thread >> > (L, L, L, a);

    gettimeofday(&startt, NULL);
	eps = jac_parallel(a, L, L, L, ITMAX, MAXEPS);
    cudaDeviceSynchronize();
    gettimeofday(&endt, NULL);

	cudaFree(a);
    
    print_benchmark(eps, startt, endt);

    a = (double*) malloc(L * L * L * sizeof(double));
    initial_seq(L, L, L, a);

    gettimeofday(&startt, NULL);
    eps = jac_sequence(a, L, L, L, ITMAX, MAXEPS);
    gettimeofday(&endt, NULL);

    print_benchmark(eps, startt, endt);

    free(a);
	return 0;
}

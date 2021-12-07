/*
Compilations Instructions: Use the following command to run:
                            "nvcc --std=c++11 *.cu -o executable.out"
*/

/*
 References:
    https://webpages.uncc.edu/abw/coit-grid01.uncc.edu/ITCS4145F12/Assignments/assign5F12.pdf
    http://www.joshiscorner.com/2013/12/2d-heat-conduction-solving-laplaces-equation-on-the-cpu-and-the-gpu/
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include "unistd.h"
#include <list>
#include <cmath>
#include <algorithm>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
using namespace std;

__global__ void Laplace(double *T_old, double *T_new, long X, long Y)
{
    /*
    GPU function to update the new array based on the values in the old array
    */

    //computing 2D indexes for a particular thread
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;

    //computing 1D index from 2D indexes of point and its neighbors
    int P = i + j*X;
    int N = i + (j+1)*X;
    int S = i + (j-1)*X;
    int E = (i+1) + j*X;
    int W = (i-1) + j*X;

    //updating the interior node point
    if ((i > 0) && (i < (X-1)) && (j > 0) && (j < (Y-1)))
    {
        T_new[P] = 0.25*(T_old[E] + T_old[W] + T_old[N] + T_old[S]);
    }
}

void init(double *TEMP, long X, long Y)
{
    /*
    function to initialize the internal temperature of the plate
    and set boundary conditions
    */

    //setting all nodes of plate to 20 C (internal temp)
    for(int i = 0; i < X; i++)
    {
        for(int j = 0; j < Y; j++)
        {
            int idx = i + j*X;
            TEMP[idx] = 20.0000000000;
        }
    }

//
//    //setting Top of plate to 20 C
//    for(int i = 0; i < X; i++)
//    {
//        TEMP[i] = 20.000000;
//    }
//
//    //setting Bottom of plate to 20 C
//    for(int i = (Y-1)*X; i < (Y*X); i++)
//    {
//        TEMP[i] = 20.000000;
//    }
//
//    //setting Left of plate to 20 C
//    for(int j = 0; j < Y; j++)
//    {
//        int idx = j*X;
//        TEMP[idx] = 20.000000;
//    }
//
//    //setting Right of plate to 20 C
//    for(int j = 0; j < Y; j++)
//    {
//        int idx = j*X + (X-1);
//        TEMP[idx] = 20.000000;
//    }
//

    //setting 40% of top side to 100 C
    for(int i = 0; i < X; i++)
    {
        if ((i > 0.3*(X-1)) && (i < 0.7*(X-1)))
        {
            TEMP[i] = 100.0000000000;
        }
    }
}

bool isNumeric(const string &strIn, long &nInputNumber)
{
    /*
    Checks if the argument is numeric and returns true/false accordingly
    checks for the arguments -N and -I
    */

    bool bRC = all_of(strIn.begin(), strIn.end(), [](unsigned char c)
    {
        return ::isdigit(c);                      // http://www.cplusplus.com/reference/algorithm/all_of/
    }                                             // https://www.geeksforgeeks.org/lambda-expression-in-c/
    );                                            // http://www.cplusplus.com/reference/cctype/isdigit/

    if (bRC)
    {
        nInputNumber = stoul(strIn);              // https://www.cplusplus.com/reference/string/stoul/
        return true;
    }
    else
    {
        return false;
    }
}


int main(int argc, char* argv[])
{
    /*
    main function to check for all invalid combination of arguments -N and -I
    and performs multi-threaded laplace computations and write the coordinate values to csv file
    and outputs total computation time using cuda kernel
    */

    if (argc == 5)
    {
        long dim{ 0 };
        long iter{ 0 };
        string strInput1(argv[1]);    //-N
        string strInput2(argv[2]);    //positive integer
        string strInput3(argv[3]);    //-I
        string strInput4(argv[4]);    //positive integer

        if ((strInput1 != "-N") || (strInput3 != "-I"))
        {
            cout << "Invalid parameters, please check your values" << endl;
            return EXIT_SUCCESS;
        }

        bool bIsValid1 = isNumeric(strInput2, dim);
        bool bIsValid2 = isNumeric(strInput4, iter);

        if ((bIsValid1) && (bIsValid2) && (!strInput2.empty()) && (!strInput4.empty()) && (dim > 0) && (dim <= 256) && (iter >= 1) && (iter <= 10000))
        {
            long X = dim + 2; //assigning N+2 to X
            long Y = dim + 2; //assigning N+2 to Y
            long ITER = iter; //assigning iteration number

            double *T = new double[X*Y];   //allocating host memory
            double *_T1, *_T2;  //pointers to device (GPU) memory

            //initialize array on the host
            init(T, X, Y);

            //allocate storage space on the GPU
            cudaMalloc((void **)&_T1,X*Y*sizeof(double));
            cudaMalloc((void **)&_T2,X*Y*sizeof(double));

            //copy (initialized) host arrays to the GPU memory from CPU memory
            cudaMemcpy(_T1, T, X*Y*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(_T2, T, X*Y*sizeof(double), cudaMemcpyHostToDevice);

            //assign a 2D distribution of CUDA "threads" within each CUDA "block"
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);
            int ThreadsPerBlock = 16;//deviceProp.maxThreadsPerBlock;

            dim3 dimBlock(ThreadsPerBlock, ThreadsPerBlock);

            //calculate number of blocks along X and Y in a 2D CUDA "grid"
            dim3 dimGrid(ceil(double(X)/double(dimBlock.x)), ceil(double(Y)/double(dimBlock.y)), 1);

            float time;              //for measuring time
            cudaEvent_t start, stop;

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            //begin Jacobi iteration
            int k = 0;
            while(k < ITER)
            {
                Laplace<<<dimGrid, dimBlock>>>(_T1, _T2, X, Y);   //update T1 using data stored in T2
                Laplace<<<dimGrid, dimBlock>>>(_T2, _T1, X, Y);   //update T2 using data stored in T1
                k += 2;
            }
            cudaEventRecord(stop);

            //copy final array to the CPU from the GPU
            cudaMemcpy(T, _T2, X*Y*sizeof(double), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);

            float mul = pow(10.0, 2);         //rounding off to 2 decimal places
            float tm = ceil(time * mul) / mul;
            cout << tm << endl;

            //writing to csv file
            ofstream myfile ("finalTemperatures.csv");
            if (myfile.is_open())
            {
                for (int j = 0; j < Y; j++)
                {
                    string line;
                    for (int i = 0; i < X; i++)
                    {
                        int idx = i + j*X;

                        //rounding off to 10 decimal places
                        double multiplier = pow(10.0, 10);
                        double val = ceil(T[idx] * multiplier) / multiplier;
                        string elem = to_string(val) + ",";
                        line.append(elem);
                    }
                    myfile << line << "\n"; //comma separated values and new line after every row
                }
                myfile.close();
            }
            else
            {
                cout << "Unable to open file";
                return EXIT_SUCCESS;
            }

            // release memory on the host
            delete T;

            // release memory on the device
            cudaFree(_T1);
            cudaFree(_T2);

            //outputs execution time to console
            return 0;
        }
        else
        {
            cout << "Invalid parameters, please check your values" << endl;
            return EXIT_SUCCESS;
        }
    }
    else
    {
        cout << "Invalid parameters, please check your values" << endl;
        return EXIT_SUCCESS;
    }
}






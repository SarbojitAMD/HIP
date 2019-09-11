#include <stdio.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

#define CHECK_HIP_ERROR(err)                                            \
{                                                                       \
    if(err != hipSuccess)                                               \
    {                                                                   \
        fprintf(stderr, "HIP_ERROR %d in line %d\n", err, __LINE__);    \
        exit(1);                                                        \
    }                                                                   \
}

struct UserData
{
    size_t size;
    int* ptr;
};

__device__ int gData = 0;

__global__ void Update()
{
    gData = gData + 5;
}

// void HIPRT_CB myCallback(hipStream_t stream, hipError_t status, void* user_data) // this throws some compiler warnings
void myCallback(hipStream_t stream, hipError_t status, void* user_data)
{
    UserData* data = reinterpret_cast<UserData*>(user_data);
    printf("Callback called with arg.size = %lu ; arg.ptr = %p\nsleeping for 1 sec...\n", data->size, data->ptr);

    sleep(1);

    printf("Callback ending.\n");
}

int main(int argc, char* argv[])
{
    // Stream
    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // Array size
    size_t size = 10000;
    if(argc > 1)
    {
        size = atol(argv[1]);
    }

    // Device array
    int *data = NULL;
    CHECK_HIP_ERROR(hipMalloc((void**)&data, sizeof(int) * size));

    // Initialize device array to -1
    CHECK_HIP_ERROR(hipMemset(data, -1, sizeof(int) * size));

    // Host array
    int *host = NULL;
    CHECK_HIP_ERROR(hipHostMalloc((void**)&host, sizeof(int) * size));

    // Print host ptr address
    printf("Host ptr address = %p\n", host);

    // Initialize user_data for callback
    UserData arg;
    arg.size = size;
    arg.ptr  = host;

    // Synchronize device
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    // Asynchronous copy from device to host
    CHECK_HIP_ERROR(hipMemcpyAsync(host, data, sizeof(int) * size, hipMemcpyDeviceToHost, stream));

    // Asynchronous memset on device
    CHECK_HIP_ERROR(hipMemsetAsync(data, 0, sizeof(int) * size, stream));

    // Launch Kernel 
    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    int *hData = new int(0);

    hipLaunchKernel((const void*)Update,grid,block,NULL,0,stream);

    CHECK_HIP_ERROR(hipMemcpyAsync(hData, &gData, sizeof(int), hipMemcpyDeviceToHost, stream));

    // Add callback - should happen after hipMemsetAsync()
    CHECK_HIP_ERROR(hipStreamAddCallback(stream, myCallback, &arg, 0));

    // This should happen before the callback actually gets called
    // when the size is large enough
    printf("This should happen before the callback (assuming sufficiently large size).\n");

    //hipLaunchKernel((const void *)Update,grid,block,NULL,0,stream);
    //CHECK_HIP_ERROR(hipMemcpyAsync(hData, &gData, sizeof(int), hipMemcpyDeviceToHost, stream));

    //This should synchronize the stream (including the callback)
    //CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    printf("This should happen after the callback, since we synchronized stream before.\n");

    // Print some host data that just got copied
    printf("Pseudo host data printing (should be -1): %d\n", host[size/2]);

    CHECK_HIP_ERROR(hipMemcpy(host, data, sizeof(int)*size, hipMemcpyDeviceToHost));
    printf("Pseudo host data printing (should be 0): %d\n", host[size-1]);

    //CHECK_HIP_ERROR(hipFree(data));
    // CHECK_HIP_ERROR(hipHostFree(host));
    //CHECK_HIP_ERROR(hipStreamDestroy(stream));

    printf("Program ends\n");
    return 0;
}

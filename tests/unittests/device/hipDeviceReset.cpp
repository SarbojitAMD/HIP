/*
Copyright (c) 2020-Present Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"

/*
 * Testing : hipDeviceReset should make all previous allocations invalid
 * Test strategy :
 *      i)  Allocate memory on host and device
 *      ii) hipDeviceReset()
 *      iii)Try hipMemcopy using already allocated pointers
 *      Expected behavior : API should return error as invalid pointer
 * Note: On AMD platform test can't be executed because of AMD current implementation.
 *       Hence disabling it on AMD.
 */
bool PositiveTest1(){
#ifdef __HIP_PLATFORM_NVCC__
  int *h_A;
  int *d_A;

  // Allocation
  HIPCHECK(hipHostMalloc((void**)&h_A, sizeof(int)));
  HIPCHECK(hipMalloc((void**)&d_A, sizeof(int)));

  // dummy copy to make sure pointers are valid
  HIPCHECK(hipMemcpy(d_A, h_A, sizeof(int), hipMemcpyHostToDevice));

  HIPCHECK(hipDeviceReset());

  // copy operation to verify previous pointers are no longer valid 
  HIPASSERT(hipMemcpy(d_A, h_A, sizeof(int), hipMemcpyHostToDevice) == hipErrorInvalidValue);
#else
  skipped()
#endif
  return true;
}

/*
 * Testing : hipDeviceReset on one device should not impact other device operation
 * Test strategy :
 *      i)  Set curernt device + enqueue some tasks ( allocate memory, initiate copy, etc.)
 *      ii) Switch to other device and call hipDeviceReset()
 *      Expected behavior : Any operation going on non-reseted device should be work fine
 */
#define SIZE 1024*1024
bool PositiveTest2(){
  int devCount = 0;
  HIPCHECK(hipGetDeviceCount(&devCount));

  if (devCount > 1){
    HIPCHECK(hipSetDevice(0));
    int *h_A;
    int *d_A;

    HIPCHECK(hipHostMalloc((void**)&h_A, sizeof(int)*SIZE));
    HIPCHECK(hipMalloc((void**)&d_A, sizeof(int)*SIZE));
    HIPCHECK(hipMemcpyAsync(d_A, h_A, sizeof(int)*SIZE, hipMemcpyHostToDevice));

    hipSetDevice(1);
    hipDeviceReset();

    HIPCHECK(hipMemcpyAsync(d_A, h_A, sizeof(int)*SIZE, hipMemcpyHostToDevice));
  }
  return true;
}

bool PositiveTests(){
  bool status = true;

  status &= PositiveTest1();
  status &= PositiveTest2();

  return status;
}

int main(){
  if (PositiveTests()){
    passed();
  }
  return 0;
}

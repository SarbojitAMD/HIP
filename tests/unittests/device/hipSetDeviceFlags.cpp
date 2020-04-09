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
 * BUILD: %t %s ../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"

bool NegativeTests(){
/*
 * Test fails on HIP platform hence skipped the test for HIP platforms
 */
#ifdef __HIP_PLATFORM_NVCC__
  // To initialize current device
  int *d_A = nullptr;
  HIPCHECK(hipMalloc((void**)&d_A, sizeof(int)));

  // Without device reset hence call should fail
  int flag = 1;
  HIPASSERT(hipSetDeviceFlags(flag) == hipErrorSetOnActiveProcess);

  // Invalid flag value
  flag = 0xF;
  hipDeviceReset();
  HIPASSERT(hipSetDeviceFlags(flag) == hipErrorInvalidValue);
#else
  skipped();
#endif
  return true;
}

bool PositiveTests(){
  unsigned flag = 0;
  HIPCHECK(hipDeviceReset());

  int deviceCount = 0;
  HIPCHECK(hipGetDeviceCount(&deviceCount));

  for (int j = 0; j < deviceCount; j++) {
    HIPCHECK(hipSetDevice(j));

    for (int i = 0; i < 4; i++) {
      flag = 1 << i;
      HIPCHECK(hipSetDeviceFlags(flag));
    }
  }
  return true;
}

int main() {
  bool status = true;

  status &= NegativeTests();
  status &= PositiveTests();
  if(status){
      passed();
  }
  return 0;
}

**Requirements:**

- NVIDIA CUDA SDK
- CUDA 4.0
- gcc 4.4 and g++ 4.8

**Buildng and Running:**

1) Modify CUDAHOME and NVIDIA_CUDA_SDK_LOCATION in v3.x/setup_environment to the location where NVIDIA CUDA SDK and CUDA 4.0 are installed. 

2) Change directory to v3.x 

3) Source the setup_environment file by changing to running `source setup_environment`

2) Make the simulator by running make in the v3.x folder

3) Change directory to pthread_benchmark. Then, `make` in the pthread_benchmark folder.

4) run `./gpgpu-sim [benchmark name list]`. The list [benchmark name list] should contain a list of benchmarks that will be run, seperate by space. For example, running `./gpgpu-sim BLK MM ` will run launch the simulation with two concurrently executing benchmarks.

**Changing the configuration:**

In the pthread_benchmark directory, there exists the configuration file called `gpgpusim.config `. This file contains the configurations paramters that are used in the simulator. 

The paramter `vm_config` decides which configuration to run the simulation in. For example, setting `vm_config` as 0 runs baseline, and setting `vm_config` as 6 runs DWS.

| vm_config | configuration |
| ------ | ------ |
| 0 | baseline |
| 1 | ideal tlb |
| 3 | per application tlb |
| 5 | per application tlb + page walkers|
| 6 | DWS |
| 7 | DWS ++ |


**Parsing the outupt:**

While the simulator does generate statistics files, one should redirect the stdout to a file. When multiple kernels are run, the kernels may have different finish times. Some kernels may finish multiple runs before the another can finish one run. We dump the generated statistics at kernel boundaries to stdout. By capturing stdout, one can also see study the statistics at different points of the simulation run.

**Copyright**

Copyright (c) 2020 Indian Institute of Science

All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution.
Neither the names of Computer Systems Lab, Indian Institute of Science, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.


//Copyright ETH Zurich, IWF

//This file is part of iwf_mfree_gpu_3d.

//iwf_mfree_gpu_3d is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

//iwf_mfree_gpu_3d is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with mfree_iwf.  If not, see <http://www.gnu.org/licenses/>.

//NOTE: this file is adapted code from the ``particles'' CUDA programming sample, license is stated below
// This is one of two available implementations of the fast neighbor search method (spatial hashing)
// (implements abstract class defined grid_base.h)
//  - the cell lists are generated using a custom kernel that exploits shared memory to quickly find the (memory) locations
//    in the particle array where hashes differ between adjecent particles in memory
//	- a device buffer is allocated holding a complete copy of the particle data
//  - this buffer is re-ordered using the particle hashes
//  - the buffer is then either copied back or the pointers are switched
//  	NOTE: the latter option is always preferable, the first one is there for clarity and debugging purposes
//            simply leave m_buffer_method set to buffer_method::swap

//Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions
//are met:
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of NVIDIA CORPORATION nor the names of its
//   contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
//EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
//PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
//OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//For additional information on the license terms, see the CUDA EULA at
//https://docs.nvidia.com/cuda/eula/index.html

#ifndef GRID_GPU_GREEN_H_
#define GRID_GPU_GREEN_H_

#include "particle_gpu.h"

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/partition.h>

#include "grid_gpu_base.h"
#include "tool_3d.h"
#include "types.h"

class grid_gpu_green : public grid_base {
public:

	//buffering
	struct device_buffer {
		float4_t *pos        = 0;
		float4_t *vel        = 0;
		float4_t *vel_bc     = 0;
		float3_t *fc         = 0;
		float3_t *ft         = 0;
		float_t  *h          = 0;
		float_t  *rho        = 0;
		mat3x3_t *S          = 0;
		float_t  *eps_pl     = 0;
		float_t  *eps_pl_dot = 0;
		float_t  *T          = 0;
		float_t  *fixed      = 0;
		float_t  *blanked    = 0;
		float_t  *tool_particle = 0;

		float3_t *pos_t      = 0;
		float3_t *vel_t      = 0;
		float_t  *rho_t      = 0;
		mat3x3_t *S_t        = 0;
		float_t  *T_t        = 0;
	};

private:
	enum buffer_method {copy, swap};

	buffer_method m_buffer_method = buffer_method::swap;
	device_buffer *m_buffer;

	int *m_cell_start = 0;
	int *m_cell_end   = 0;

	void alloc_buffer(int num_cell, int num_part);

public:

	void sort(particle_gpu *particles) const override;
	void get_cells(particle_gpu *particles, int *cell_start, int *cell_end) override;

	grid_gpu_green(unsigned int max_cell, unsigned int N);
	grid_gpu_green(int num_part, float3_t bbmin, float3_t bbmax, float_t h);
};

#endif /* GRID_GPU_GREEN_H_ */

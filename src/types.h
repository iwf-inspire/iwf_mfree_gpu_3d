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

//contains global defines
//	- single or double precision
//  - thermal transport algorithm
//  - kernel correction to linear completeness (CSPM)

#ifndef TYPES_H_
#define TYPES_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <glm/glm.hpp>

//#define USE_DOUBLE

#ifdef USE_DOUBLE
//built in vector types
#define float_t double
#define float2_t double2
#define float3_t double3
#define float4_t double4

// texture types and texture fetching
#define float_tex_t  int2
#define float2_tex_t int4
#define float4_tex_t int4

#define make_float2_t make_double2
#define make_float3_t make_double3
#define make_float4_t make_double4

#define texfetch1 fetch_double
#define texfetch2 fetch_double2
#define texfetch4 fetch_double2

// glm types
#define mat3x3_t glm::dmat3x3
#define vec2_t glm::dvec2
#define vec3_t glm::dvec3

// mathematic functions
#define sqrt_t sqrt
#define exp_t exp
#define log_t log

#else

//-------------------------------------------------------------

// built in vector types
#define float_t float
#define float2_t float2
#define float3_t float3
#define float4_t float4

#define make_float2_t make_float2
#define make_float3_t make_float3
#define make_float4_t make_float4

// texture types
#define float_tex_t  float
#define float2_tex_t float2
#define float4_tex_t float4

#define texfetch1 tex1Dfetch
#define texfetch2 tex1Dfetch
#define texfetch4 tex1Dfetch

// glm types
#define mat3x3_t glm::mat3x3
#define vec2_t glm::vec2
#define vec3_t glm::vec3

// mathematic functions
#define sqrt_t sqrtf
#define exp_t expf
#define log_t logf

#endif

bool check_cuda_error();
bool check_cuda_error(const char *marker);

//#define CSPM

#define Thermal_Conduction_Brookshaw
//#define Thermal_Conduction_PSE

// Customization of CUDA- Blocksizes
#define BLOCK_SIZE 256

#endif /* TYPES_H_ */

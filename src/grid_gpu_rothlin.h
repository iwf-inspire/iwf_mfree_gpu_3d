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
//	- based on the masters thesis by m.roethlin
//  - allocates a buffer per particle attribute _type_ (e.g., float, float2, int, ...)
//  - uses thrust primitives to reorder the particle data and generate the cell lists
//  - slower than greens approach but saves memory

#ifndef GRID_GPU_H_
#define GRID_GPU_H_

//cuda
#include <cuda.h>
#include <cstdio>

#include <assert.h>

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/partition.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>
#include <time.h>
#include <thrust/extrema.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>

#include "grid_gpu_base.h"
#include "particle_gpu.h"
#include "types.h"

class grid_gpu_rothlin : public grid_base {

private:
	//temp array to work around that gather works out of place
	int      *m_tempi   = 0;
	float_t  *m_temp    = 0;
	float2_t *m_tempf2  = 0;
	float3_t *m_tempf3  = 0;
	float4_t *m_tempf4  = 0;
	mat3x3_t *m_tempmat = 0;

	int   *m_seq = 0;

	//cell computation stuff
	int *m_cell_indices = 0;
	int *m_cell_map     = 0;
	int *m_cell_stencil = 0;
	int *m_cell_scatter = 0;

	//offsets to the first particle in a box
	int *m_cell_offsets = 0;

	thrust::device_vector<int> do_get_cells(int *hashes, int num_part);

	void alloc_arrays(int num_cell, int num_part);

public:

	void print() const;

	//additional interface compared to cpu version
	//		sorting and restoring a large number of arrays is not straight forward on gpu
	void sort(particle_gpu *particles) const override;

	//debug
	void assign_random_hashes(particle_gpu *particles) const;

	void get_cells(particle_gpu      *particles, int *cell_start, int *cell_end) override;

	grid_gpu_rothlin(int max_cell, int N);
	grid_gpu_rothlin(int num_part, float3_t bbmin, float3_t bbmax, float_t h);

};

#endif /* GRID_GPU_H_ */

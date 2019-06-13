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

#include "grid_gpu_green.h"

//NOTE: its not worth it to bind particle arrays to textures, no runtime improvement measurable
//      	=> keep this version, code is more readable & compact
// http://developer.download.nvidia.com/assets/cuda/files/particles.pdf
//
__global__ static void find_cell_start(
		const particle_gpu particles,
        int   *__restrict__ cell_start,
        int   *__restrict__ cell_end,
        int    num_cell,
        int    num_part) {

    extern __shared__ int shared_hash[];    // blockSize + 1 elements
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int hash = 0;

    // handle case when no. of particles not multiple of block size
    if (idx < num_part) {
        hash = particles.hash[idx];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        shared_hash[threadIdx.x+1] = hash;

        if (idx > 0 && threadIdx.x == 0) {
            // first thread in block must load neighbor particle hash
            shared_hash[0] = particles.hash[idx-1];
        }
    }

    __syncthreads();

    if (idx < num_part) {
        if (idx == 0 || hash != shared_hash[threadIdx.x]) {

        	if (hash < 0 || hash >= num_cell) {
        		printf("ALARM, particle out of grid but not blanked!: %d\n", hash);
        	}

            cell_start[hash] = idx;
            if (idx > 0) {
                cell_end[shared_hash[threadIdx.x]] = idx;
            }
        }

        if (idx == num_part - 1) {
            cell_end[hash] = idx + 1;
        }
    }
}

__global__ static void reorder_data(
        const particle_gpu particles,
        grid_gpu_green::device_buffer buffer,
        int    num_part) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_part) {

		int sorted_index = particles.idx[idx];

		float4_t pos        = particles.pos[sorted_index];
		float4_t vel        = particles.vel[sorted_index];
		float4_t vel_bc     = particles.vel_bc[sorted_index];
		float3_t fc         = particles.fc[sorted_index];
		float3_t ft         = particles.ft[sorted_index];
		float_t  h          = particles.h[sorted_index];
		float_t  rho        = particles.rho[sorted_index];
		mat3x3_t S          = particles.S[sorted_index];
		float_t  eps_pl     = particles.eps_pl[sorted_index];
		float_t  eps_pl_dot = particles.eps_pl_dot[sorted_index];
		float_t  T          = particles.T[sorted_index];
		float_t  fixed      = particles.fixed[sorted_index];
		float_t  blanked    = particles.blanked[sorted_index];
		float_t  tool_particle = particles.tool_particle[sorted_index];

		float3_t pos_t  = particles.pos_t[sorted_index];
		float3_t vel_t  = particles.vel_t[sorted_index];
		float_t  rho_t  = particles.rho_t[sorted_index];
		mat3x3_t S_t    = particles.S_t[sorted_index];
		float_t  T_t    = particles.T_t[sorted_index];

		buffer.pos[idx]        = pos;
		buffer.vel[idx]        = vel;
		buffer.vel_bc[idx]     = vel_bc;
		buffer.fc[idx]         = fc;
		buffer.ft[idx]         = ft;
		buffer.h[idx]          = h;
		buffer.rho[idx]        = rho;
		buffer.S[idx]          = S;
		buffer.eps_pl[idx]     = eps_pl;
		buffer.eps_pl_dot[idx] = eps_pl_dot;
		buffer.T[idx]          = T;
		buffer.fixed[idx]      = fixed;
		buffer.blanked[idx]    = blanked;
		buffer.tool_particle[idx]   = tool_particle;

		buffer.pos_t[idx]  = pos_t;
		buffer.vel_t[idx]  = vel_t;
		buffer.rho_t[idx]  = rho_t;
		buffer.S_t[idx]    = S_t;
		buffer.T_t[idx]    = T_t;
	}

}

__global__ static void reorder_data_and_find_cell_start(
                                  const particle_gpu particles,
                                  grid_gpu_green::device_buffer buffer,
                                  int   *__restrict__ cell_start,
                                  int   *__restrict__ cell_end,
                                  int    num_part) {
    extern __shared__ int shared_hash[];    // blockSize + 1 elements
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int hash = 0;

    // handle case when no. of particles not multiple of block size
    if (idx < num_part) {
        hash = particles.hash[idx];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        shared_hash[threadIdx.x+1] = hash;

        if (idx > 0 && threadIdx.x == 0) {
            // first thread in block must load neighbor particle hash
            shared_hash[0] = particles.hash[idx-1];
        }
    }

    __syncthreads();

    if (idx < num_part && hash != INT_MAX) {
        if (idx == 0 || hash != shared_hash[threadIdx.x]) {
            cell_start[hash] = idx;
            if (idx > 0) {
                cell_end[shared_hash[threadIdx.x]] = idx;
            }
        }

        if (idx == num_part - 1) {
            cell_end[hash] = idx + 1;
        }
    }

    if (idx < num_part) {

        int sorted_index = particles.idx[idx];

        float4_t pos        = particles.pos[sorted_index];
        float4_t vel        = particles.vel[sorted_index];
        float4_t vel_bc     = particles.vel_bc[sorted_index];
        float3_t fc         = particles.fc[sorted_index];
        float3_t ft         = particles.ft[sorted_index];
        float_t  h          = particles.h[sorted_index];
        float_t  rho        = particles.rho[sorted_index];
        mat3x3_t S          = particles.S[sorted_index];
        float_t  eps_pl     = particles.eps_pl[sorted_index];
        float_t  eps_pl_dot = particles.eps_pl_dot[sorted_index];
        float_t  T          = particles.T[sorted_index];
        float_t  fixed      = particles.fixed[sorted_index];
        float_t  blanked    = particles.blanked[sorted_index];
        float_t  tool_particle = particles.tool_particle[sorted_index];

        float3_t pos_t  = particles.pos_t[sorted_index];
        float3_t vel_t  = particles.vel_t[sorted_index];
        float_t  rho_t  = particles.rho_t[sorted_index];
        mat3x3_t S_t    = particles.S_t[sorted_index];
        float_t  T_t    = particles.T_t[sorted_index];

        buffer.pos[idx]        = pos;
        buffer.vel[idx]        = vel;
        buffer.vel_bc[idx]     = vel_bc;
        buffer.fc[idx]         = fc;
        buffer.ft[idx]         = ft;
        buffer.h[idx]          = h;
        buffer.rho[idx]        = rho;
        buffer.S[idx]          = S;
        buffer.eps_pl[idx]     = eps_pl;
        buffer.eps_pl_dot[idx] = eps_pl_dot;
        buffer.T[idx]          = T;
        buffer.fixed[idx]      = fixed;
        buffer.blanked[idx]    = blanked;
        buffer.tool_particle[idx]   = tool_particle;

        buffer.pos_t[idx]  = pos_t;
        buffer.vel_t[idx]  = vel_t;
        buffer.rho_t[idx]  = rho_t;
        buffer.S_t[idx]    = S_t;
        buffer.T_t[idx]    = T_t;
    }
}

__global__ static void copy_from_buffer(particle_gpu particles, grid_gpu_green::device_buffer buffer, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

    particles.pos[idx]        = buffer.pos[idx];
    particles.vel[idx]        = buffer.vel[idx];
    particles.vel_bc[idx]     = buffer.vel_bc[idx];
    particles.fc[idx]         = buffer.fc[idx];
    particles.ft[idx]         = buffer.ft[idx];
    particles.h[idx]          = buffer.h[idx];
    particles.rho[idx]        = buffer.rho[idx];
    particles.S[idx]          = buffer.S[idx];
    particles.eps_pl[idx]     = buffer.eps_pl[idx];
    particles.eps_pl_dot[idx] = buffer.eps_pl_dot[idx];
    particles.T[idx]          = buffer.T[idx];
    particles.fixed[idx]      = buffer.fixed[idx];
    particles.blanked[idx]    = buffer.blanked[idx];
    particles.tool_particle[idx]    = buffer.tool_particle[idx];

    particles.pos_t[idx]  = buffer.pos_t[idx];
    particles.vel_t[idx]  = buffer.vel_t[idx];
    particles.rho_t[idx]  = buffer.rho_t[idx];
    particles.S_t[idx]    = buffer.S_t[idx];
    particles.T_t[idx]    = buffer.T_t[idx];
}

static void double_buffer(particle_gpu *particles, grid_gpu_green::device_buffer *buffer) {
	std::swap(particles->pos,buffer->pos);
    std::swap(particles->vel, buffer->vel);
    std::swap(particles->vel_bc, buffer->vel_bc);
    std::swap(particles->fc, buffer->fc);
    std::swap(particles->ft, buffer->ft);
    std::swap(particles->h, buffer->h);
    std::swap(particles->rho, buffer->rho);
    std::swap(particles->S, buffer->S);
    std::swap(particles->eps_pl, buffer->eps_pl);
    std::swap(particles->eps_pl_dot, buffer->eps_pl_dot);
    std::swap(particles->T, buffer->T);
    std::swap(particles->fixed, buffer->fixed);
    std::swap(particles->blanked, buffer->blanked);
    std::swap(particles->tool_particle, buffer->tool_particle);

    std::swap(particles->pos_t, buffer->pos_t);
    std::swap(particles->vel_t, buffer->vel_t);
    std::swap(particles->rho_t, buffer->rho_t);
    std::swap(particles->S_t, buffer->S_t);
    std::swap(particles->T_t, buffer->T_t);
}

template<class T>
static void do_sort(T* particles, grid_gpu_green::device_buffer *buffer, bool use_double_buffer,
		int *cell_start, int *cell_end, long int max_cell, bool hard_blank) {

	//index vec
	thrust::device_ptr<int> t_idx(particles->idx);
	thrust::sequence(t_idx, t_idx+particles->N_init);

	//keys to sort by
	thrust::device_ptr<int> t_hashes(particles->hash);
	thrust::sort_by_key(t_hashes, t_hashes+particles->N_init, t_idx);

	cudaMemset(cell_start, 0xffffffff, max_cell*sizeof(int));
	cudaMemset(cell_end,   0xffffffff, max_cell*sizeof(int));

	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N_init + block_size-1) / block_size);
	dim3 dB(block_size);
	unsigned int shared_mem_size = (block_size+1)*sizeof(int);

	if (!hard_blank) {
		reorder_data_and_find_cell_start<<< dG, dB, shared_mem_size>>>(*particles, *buffer, cell_start, cell_end, particles->N_init);
	} else {
		reorder_data<<< dG, dB>>>(*particles, *buffer, particles->N_init);
		find_cell_start<<< dG, dB, shared_mem_size>>>(*particles, cell_start, cell_end, max_cell, particles->N);
	}

	if (use_double_buffer) {
		double_buffer(particles, buffer);
	} else {
		copy_from_buffer<<<dG, dB>>>(*particles, *buffer, particles->N_init);
	}
}

void grid_gpu_green::sort(particle_gpu *particles) const {
	do_sort(particles, m_buffer, m_buffer_method == buffer_method::swap, m_cell_start, m_cell_end, m_num_cell, m_hard_blank);
}

void grid_gpu_green::get_cells(particle_gpu *particles, int *cell_start, int *cell_end)  {
	cudaMemcpy(cell_start, m_cell_start, sizeof(int)*m_num_cell, cudaMemcpyDeviceToDevice);
	cudaMemcpy(cell_end,   m_cell_end,   sizeof(int)*m_num_cell, cudaMemcpyDeviceToDevice);
}

void grid_gpu_green::alloc_buffer(int num_cell, int num_part) {
	m_buffer = new grid_gpu_green::device_buffer();

	cudaMalloc((void**) &m_buffer->pos, sizeof(float4_t)*num_part);
	cudaMalloc((void**) &m_buffer->vel, sizeof(float4_t)*num_part);
	cudaMalloc((void**) &m_buffer->vel_bc, sizeof(float4_t)*num_part);
	cudaMalloc((void**) &m_buffer->fc, sizeof(float3_t)*num_part);
	cudaMalloc((void**) &m_buffer->ft, sizeof(float3_t)*num_part);
	cudaMalloc((void**) &m_buffer->h, sizeof(float_t)*num_part);
	cudaMalloc((void**) &m_buffer->rho, sizeof(float_t)*num_part);
	cudaMalloc((void**) &m_buffer->S, sizeof(mat3x3_t)*num_part);
	cudaMalloc((void**) &m_buffer->eps_pl, sizeof(float_t)*num_part);
	cudaMalloc((void**) &m_buffer->eps_pl_dot, sizeof(float_t)*num_part);
	cudaMalloc((void**) &m_buffer->T, sizeof(float_t)*num_part);
	cudaMalloc((void**) &m_buffer->fixed, sizeof(float_t)*num_part);
	cudaMalloc((void**) &m_buffer->blanked, sizeof(float_t)*num_part);
	cudaMalloc((void**) &m_buffer->tool_particle, sizeof(float_t)*num_part);

	cudaMalloc((void**) &m_buffer->pos_t, sizeof(float4_t)*num_part);
	cudaMalloc((void**) &m_buffer->vel_t, sizeof(float4_t)*num_part);
	cudaMalloc((void**) &m_buffer->rho_t, sizeof(float_t)*num_part);
	cudaMalloc((void**) &m_buffer->S_t, sizeof(mat3x3_t)*num_part);
	cudaMalloc((void**) &m_buffer->T_t, sizeof(float_t)*num_part);

	cudaMalloc((void**) &m_cell_start, sizeof(int)*num_cell);
	cudaMalloc((void**) &m_cell_end,   sizeof(int)*num_cell);
}

grid_gpu_green::grid_gpu_green(unsigned int max_cell, unsigned int N) :
	grid_base(max_cell, N) {

	alloc_buffer(max_cell, N);
}

grid_gpu_green::grid_gpu_green(int num_part, float3_t bbmin, float3_t bbmax, float_t h) :
	grid_base(num_part, bbmin, bbmax, h) {

	alloc_buffer(m_max_cell, num_part);
}

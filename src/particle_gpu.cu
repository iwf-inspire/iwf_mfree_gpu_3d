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

#include "particle_gpu.h"

particle_gpu::particle_gpu(unsigned int N) {
	cudaMalloc((void **) &pos, sizeof(float4_t)*N);
	cudaMalloc((void **) &vel, sizeof(float4_t)*N);
	cudaMalloc((void **) &vel_bc, sizeof(float4_t)*N);

	cudaMalloc((void **) &h,   sizeof(float_t)*N);
	cudaMalloc((void **) &rho, sizeof(float_t)*N);
	cudaMalloc((void **) &p,   sizeof(float_t)*N);

	cudaMalloc((void **) &S,   sizeof(mat3x3_t)*N);
	cudaMalloc((void **) &R,   sizeof(mat3x3_t)*N);
	cudaMalloc((void **) &fc,  sizeof(float3_t)*N);
	cudaMalloc((void **) &ft,  sizeof(float3_t)*N);
	cudaMalloc((void **) &n,   sizeof(float3_t)*N);

	cudaMalloc((void**) &fixed, sizeof(float_t)*N);
	cudaMalloc((void**) &blanked, sizeof(float_t)*N);
	cudaMalloc((void**) &tool_particle, sizeof(float_t)*N);

	cudaMalloc((void**) &eps_pl, sizeof(float_t)*N);
	cudaMalloc((void**) &eps_pl_dot, sizeof(float_t)*N);
	cudaMalloc((void**) &T, sizeof(float_t)*N);

	cudaMalloc((void **) &pos_t, sizeof(float4_t)*N);
	cudaMalloc((void **) &vel_t, sizeof(float4_t)*N);
	cudaMalloc((void **) &rho_t, sizeof(float_t)*N);
	cudaMalloc((void **) &S_t,   sizeof(mat3x3_t)*N);
	cudaMalloc((void **) &T_t,   sizeof(float_t)*N);

	cudaMalloc((void **) &v_der, sizeof(mat3x3_t)*N);
	cudaMalloc((void **) &S_der, sizeof(mat3x3_t)*N);

	cudaMalloc((void **) &idx, sizeof(int)*N);
	cudaMalloc((void **) &hash, sizeof(int)*N);

	thrust::device_ptr<int> t_idx(this->idx);
	thrust::sequence(t_idx, t_idx+N);

	cudaMalloc((void**) &num_nbh, sizeof(int)*N);

	cudaMemset(pos, 0, sizeof(float4_t)*N);
	cudaMemset(vel,0, sizeof(float4_t)*N);
	cudaMemset(vel_bc,0, sizeof(float4_t)*N);

	cudaMemset(h,0,   sizeof(float_t)*N);
	cudaMemset(rho,0, sizeof(float_t)*N);
	cudaMemset(p,0,   sizeof(float_t)*N);

	cudaMemset(S,0,   sizeof(mat3x3_t)*N);
	cudaMemset(R,0,   sizeof(mat3x3_t)*N);
	cudaMemset(fc,0,  sizeof(float3_t)*N);
	cudaMemset(ft,0,  sizeof(float3_t)*N);
	cudaMemset(n,0,   sizeof(float3_t)*N);

	cudaMemset(fixed,0,  sizeof(float_t)*N);
	cudaMemset(blanked,0,  sizeof(float_t)*N);
	cudaMemset(tool_particle, 0,  sizeof(float_t)*N);

	cudaMemset(eps_pl, 0, sizeof(float_t)*N);
	cudaMemset(eps_pl_dot, 0, sizeof(float_t)*N);
	cudaMemset(T, 0, sizeof(float_t)*N);

	cudaMemset(pos_t,0, sizeof(float3_t)*N);
	cudaMemset(vel_t,0, sizeof(float3_t)*N);
	cudaMemset(rho_t,0, sizeof(float_t)*N);
	cudaMemset(S_t,0,   sizeof(mat3x3_t)*N);
	cudaMemset(T_t,0,   sizeof(float_t)*N);

	cudaMemset(v_der, 0, sizeof(mat3x3_t)*N);
	cudaMemset(S_der, 0, sizeof(mat3x3_t)*N);

	cudaMemset(num_nbh, 0, sizeof(int)*N);

	this->N = N;
	this->N_init = N;
}

particle_gpu::particle_gpu(float4_t *pos, float4_t *vel_init, float_t *rho, float_t *h, unsigned int N) : particle_gpu(N) {
	cudaMemcpy(this->pos, pos,      sizeof(float4_t)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(this->vel, vel_init, sizeof(float4_t)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(this->rho, rho,      sizeof(float_t)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(this->h,   h,        sizeof(float_t)*N, cudaMemcpyHostToDevice);
}

particle_gpu::particle_gpu(float4_t *pos, float4_t *vel_init, float_t *rho, float_t *h, float_t *fixed, unsigned int N)
: particle_gpu(pos, vel_init, rho, h, N) {
	cudaMemcpy(this->fixed, fixed, sizeof(float_t)*N, cudaMemcpyHostToDevice);
}

particle_gpu::particle_gpu(float4_t *pos, float4_t *vel_init, float_t *rho, float_t *T_init, float_t *h, float_t *fixed, unsigned int N)
: particle_gpu(pos, vel_init, rho, h, fixed, N) {
	cudaMemcpy(this->T, T_init, sizeof(float_t)*N, cudaMemcpyHostToDevice);
}

particle_gpu::particle_gpu(float4_t *pos, float4_t *vel_init, float_t *rho, float_t *T_init, float_t *h, float_t *fixed, float_t *tool_p, unsigned int N)
: particle_gpu(pos, vel_init, rho, T_init, h, fixed, N) {
	cudaMemcpy(this->tool_particle, tool_p, sizeof(float_t)*N, cudaMemcpyHostToDevice);
}

particle_gpu::particle_gpu(float4_t *pos, float4_t *vel_init, float_t *rho, float_t *h, mat3x3_t *S, unsigned int N) : particle_gpu(N) {
	cudaMemcpy(this->pos, pos,      sizeof(float4_t)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(this->vel, vel_init, sizeof(float4_t)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(this->rho, rho,      sizeof(float_t)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(this->h,   h,        sizeof(float_t)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(this->S,   S,        sizeof(mat3x3_t)*N, cudaMemcpyHostToDevice);
}

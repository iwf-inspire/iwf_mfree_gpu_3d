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

#include "grid_gpu_base.h"

struct compare_float4_x {
  __host__ __device__
  bool operator()(float4_t lhs, float4_t rhs) {
    return lhs.x < rhs.x;
  }
};

struct compare_float4_y {
  __host__ __device__
  bool operator()(float4_t lhs, float4_t rhs) {
    return lhs.y < rhs.y;
  }
};

struct compare_float4_z {
  __host__ __device__
  bool operator()(float4_t lhs, float4_t rhs) {
    return lhs.z < rhs.z;
  }
};

__global__ static void compute_hashes(const float4_t * pos, float_t *blanked, float_t *in_tool, int *hashes, int count,
								float_t dx, long int nx, long int ny, long int nz,
								float_t bbmin_x, float_t bbmin_y, float_t bbmin_z, long int max_cell, bool hard_blank) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < count) {

		if (blanked[idx] == 1. && hard_blank) {
			hashes[idx] = INT_MAX;
			return;
		}

		float_t _px = pos[idx].x;
		float_t _py = pos[idx].y;
		float_t _pz = pos[idx].z;

		int ix = (_px - bbmin_x)/dx;
		int iy = (_py - bbmin_y)/dx;
		int iz = (_pz - bbmin_z)/dx;

		hashes[idx] = ix*ny*nz + iy*nz + iz;
	}
}

long int grid_base::nx() const {
	return m_nx;
}

long int grid_base::ny() const{
	return m_ny;
}

long int grid_base::nz() const{
	return m_nz;
}

float_t grid_base::bbmin_x() const {
	return m_bbmin_x;
}

float_t grid_base::bbmin_y() const {
	return m_bbmin_y;
}

float_t grid_base::bbmin_z() const {
	return m_bbmin_z;
}

float_t grid_base::bbmax_x() const {
	return m_bbmax_x;
}

float_t grid_base::bbmax_y() const {
	return m_bbmax_y;
}

float_t grid_base::bbmax_z() const {
	return m_bbmax_z;
}

float_t grid_base::dx() const {
	return m_dx;
}

bool grid_base::is_locked() const {
	return m_geometry_locked;
}

long int grid_base::num_cell() const {
	return m_num_cell;
}

void grid_base::set_bbox_vel(float3_t bbox_vel) {
	m_vel_bbox = bbox_vel;
}

void grid_base::set_max_fixed(bool fixed) {
	m_max_fixed = fixed;
}

void grid_base::set_min_fixed(bool fixed) {
	m_min_fixed = fixed;
}

void grid_base::set_hard_blank(bool hard_blank) {
	m_hard_blank = hard_blank;
}

void grid_base::assign_hashes(particle_gpu *particles) const {
	const unsigned int block_size = BLOCK_SIZE;

	dim3 dG((particles->N_init + block_size-1) / block_size);
	dim3 dB(block_size);

	compute_hashes<<<dG,dB>>>(particles->pos, particles->blanked, particles->tool_particle, particles->hash, particles->N_init,
			m_dx, m_nx, m_ny, m_nz, m_bbmin_x, m_bbmin_y, m_bbmin_z, m_max_cell, m_hard_blank);
	cudaThreadSynchronize();
	check_cuda_error("After Hash Computation\n");
}

void grid_base::adapt_particle_number(particle_gpu *particles) const {
	if (!m_hard_blank) {
		return;
	}

	thrust::device_ptr<int> hashes_begin(particles->hash);
	thrust::device_ptr<int> hashes_end(particles->hash+particles->N_init);

	int num_out = thrust::count(hashes_begin, hashes_begin+particles->N_init, INT_MAX);
    int particle_number_old = particles->N;
	particles->N = particles->N_init - num_out;
    int particle_number_new = particles->N;
	if (particle_number_old != particle_number_new) {
		printf("adapted particle number. New number of particles: %d, blanked %d!\n", particle_number_new, num_out);
	}

	check_cuda_error("After adapting particle number\n");

	if (particles->N == 0) {
		printf("all particles left bbox due to blanking!\n");
		printf("aborting simulation\n");
		exit(0);
	}
}

void grid_base::update_geometry(particle_gpu *particles, float_t kernel_width) {
	if (m_geometry_locked) {
		if (fabs(m_vel_bbox.x) > 0 || fabs(m_vel_bbox.y) > 0 || fabs(m_vel_bbox.z) > 0) {
			if (!m_min_fixed) {
				m_bbmin_x = m_bbmin_x_init + global_time_current*m_vel_bbox.x;
				m_bbmin_y = m_bbmin_y_init + global_time_current*m_vel_bbox.y;
				m_bbmin_z = m_bbmin_z_init + global_time_current*m_vel_bbox.z;
			}

			if (!m_max_fixed) {
				m_bbmax_x = m_bbmax_x_init + global_time_current*m_vel_bbox.x;
				m_bbmax_y = m_bbmax_y_init + global_time_current*m_vel_bbox.y;
				m_bbmax_z = m_bbmax_z_init + global_time_current*m_vel_bbox.z;
			}
		}

		return;
	}

	unsigned int N = particles->N;

	thrust::device_ptr<float4_t> t_pos(particles->pos);
	thrust::device_ptr<float_t> t_h(particles->h);

	thrust::device_ptr<float4_t> minx = thrust::min_element(t_pos, t_pos+N, compare_float4_x());
	thrust::device_ptr<float4_t> miny = thrust::min_element(t_pos, t_pos+N, compare_float4_y());
	thrust::device_ptr<float4_t> minz = thrust::min_element(t_pos, t_pos+N, compare_float4_z());

	thrust::device_ptr<float4_t> maxx = thrust::max_element(t_pos, t_pos+N, compare_float4_x());
	thrust::device_ptr<float4_t> maxy = thrust::max_element(t_pos, t_pos+N, compare_float4_y());
	thrust::device_ptr<float4_t> maxz = thrust::max_element(t_pos, t_pos+N, compare_float4_z());

	thrust::device_ptr<float_t> maxh = thrust::max_element(t_h, t_h+N);

	//copy 4 floats individually back to host here
	//		could be optimized since values are not really needed on host and are sent back to gpu via argument passing
	float4_t f2_bbmin_x = minx[0];
	float4_t f2_bbmin_y = miny[0];
	float4_t f2_bbmin_z = minz[0];

	float4_t f2_bbmax_x = maxx[0];
	float4_t f2_bbmax_y = maxy[0];
	float4_t f2_bbmax_z = maxz[0];

	m_bbmin_x = f2_bbmin_x.x - 1e-6;
	m_bbmin_y = f2_bbmin_y.y - 1e-6;
	m_bbmin_z = f2_bbmin_z.z - 1e-6;

	m_bbmax_x = f2_bbmax_x.x + 1e-6;
	m_bbmax_y = f2_bbmax_y.y + 1e-6;
	m_bbmax_z = f2_bbmax_z.z + 1e-6;

	m_dx = maxh[0]*kernel_width/2;

	m_lx = m_bbmax_x - m_bbmin_x;
	m_ly = m_bbmax_y - m_bbmin_y;
	m_lz = m_bbmax_z - m_bbmin_z;

	m_nx = ceil(m_lx/m_dx);
	m_ny = ceil(m_ly/m_dx);
	m_nz = ceil(m_lz/m_dx);

	m_num_cell = m_nx*m_ny*m_nz;

	if (m_num_cell >= m_max_cell) {
		printf("num cell %ld max cell %ld\n", m_num_cell, m_max_cell);
	}
	assert(m_num_cell < m_max_cell);
}


void grid_base::set_geometry(float3_t bbmin, float3_t bbmax, float_t h) {
	m_bbmin_x = bbmin.x - 1e-6;
	m_bbmin_y = bbmin.y - 1e-6;
	m_bbmin_z = bbmin.z - 1e-6;

	m_bbmax_x = bbmax.x + 1e-6;
	m_bbmax_y = bbmax.y + 1e-6;
	m_bbmax_z = bbmax.z + 1e-6;

	m_bbmin_x_init = bbmin.x - 1e-6;
	m_bbmin_y_init = bbmin.y - 1e-6;
	m_bbmin_z_init = bbmin.z - 1e-6;

	m_bbmax_x_init = bbmax.x + 1e-6;
	m_bbmax_y_init = bbmax.y + 1e-6;
	m_bbmax_z_init = bbmax.z + 1e-6;

	float_t kernel_width = 2.;
	m_dx = h*kernel_width/2.;

	m_lx = m_bbmax_x - m_bbmin_x;
	m_ly = m_bbmax_y - m_bbmin_y;
	m_lz = m_bbmax_z - m_bbmin_z;

	m_nx = ceil(m_lx/m_dx);
	m_ny = ceil(m_ly/m_dx);
	m_nz = ceil(m_lz/m_dx);

	m_num_cell = m_nx*m_ny*m_nz;
	m_max_cell = m_num_cell;

	m_geometry_locked = true;

	if (m_geometry_locked) {
		printf("cells at locked geometry: %ld %ld %ld for a total of %ld\n", m_nx, m_ny, m_nz, m_num_cell);
	}
}

grid_base::grid_base(int max_cell, int num_part) :
		m_max_cell(max_cell), m_num_cell(max_cell), m_num_part(num_part){
	m_vel_bbox = make_float3_t(0.,0.,0.);
}

grid_base::grid_base(int num_part, float3_t bbmin, float3_t bbmax, float_t h) {
	m_num_part = num_part;
	set_geometry(bbmin, bbmax, h);
	m_vel_bbox = make_float3_t(0.,0.,0.);
}

grid_base::~grid_base() {
}

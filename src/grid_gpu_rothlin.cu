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

#include "grid_gpu_rothlin.h"


struct abs_functor {
	__host__ __device__
	void operator()(int &x) {
		x = abs(x);
	}
};

void grid_gpu_rothlin::print() const {
	printf("bbmin: %f %f\n", m_bbmin_x, m_bbmin_y);
	printf("bbmax: %f %f\n", m_bbmax_x, m_bbmax_y);
	printf("lx: %f ly: %f\n", m_lx, m_ly);
	printf("nx: %d ny: %d\n", m_nx, m_ny);
	printf("---------------\n");
}

thrust::device_vector<int> grid_gpu_rothlin::do_get_cells(int *hash, int num_part) {
	//todo finally figure out which of those are needed
	cudaMemset((void*) m_cell_offsets, 0, sizeof(int)*m_max_cell);
	cudaMemset((void*) m_cell_indices, 0, sizeof(int)*m_max_cell);
	cudaMemset((void*) m_cell_map, 0, sizeof(int)*m_max_cell);
	cudaMemset((void*) m_cell_stencil, 0, sizeof(int)*m_max_cell);
	cudaMemset((void*) m_cell_scatter, 0, sizeof(int)*m_max_cell);

	unsigned int cell_count = m_num_cell;

	//device pointers
	thrust::device_ptr<int> t_cell_indices	(m_cell_indices);
	thrust::device_ptr<int> t_cell_map		(m_cell_map);
	thrust::device_ptr<int> t_cell_stencil	(m_cell_stencil);
	thrust::device_ptr<int> t_cell_scatter	(m_cell_scatter);
	thrust::device_ptr<int>	t_keys			(hash);	//needs to be sorted!
	thrust::device_ptr<int> t_cell_offsets	(m_cell_offsets);

	//first order of business: invalidate all the indices, scatter array and stencil
	thrust::fill(t_cell_indices,   t_cell_indices+m_max_cell, -1);
	thrust::fill(t_cell_scatter,   t_cell_scatter+m_max_cell,  0);
	thrust::fill(t_cell_stencil,   t_cell_stencil+m_max_cell,  0);

	//next: some magic
	thrust::unique_by_key_copy(	t_keys, t_keys + num_part,
			thrust::make_counting_iterator(0),
			t_cell_map,
			t_cell_indices);

	//lets find the number of occupied celles
	int unoccupied_cells	= thrust::count(t_cell_indices,t_cell_indices+cell_count,-1);
	int occupied_cells		= cell_count - unoccupied_cells;

	//take differences of the indices to mark cell boundaries (and to get how many empty celles are in between two occupied celles)
	thrust::transform(t_cell_indices, t_cell_indices + occupied_cells-1, t_cell_indices + 1, t_cell_stencil, thrust::minus<int>());
	thrust::for_each(t_cell_stencil, t_cell_stencil + occupied_cells, abs_functor());

	//some scattering
	thrust::scatter(t_cell_stencil, t_cell_stencil + occupied_cells, t_cell_map, t_cell_scatter);

	//prefix sum for good measure
	thrust::inclusive_scan(t_cell_scatter,t_cell_scatter+cell_count,t_cell_offsets+1);

	//fix last
	//		if you figure out how to have the above do this cleverly in an implicit way i'll buy you a beer
	//		performance impact not measurable but ugly af
	int last_hash;
	int zero = 0;
	cudaMemcpy(&last_hash, &hash[num_part-1], sizeof(int), cudaMemcpyDeviceToHost);
	//	cudaMemcpy(&m_cell_offsets[last_hash+1], &num_part, sizeof(int), cudaMemcpyHostToDevice);
	thrust::fill(t_cell_offsets+last_hash+1, t_cell_offsets+cell_count+1, num_part);
	cudaMemcpy(&m_cell_offsets[0], &zero, sizeof(int), cudaMemcpyHostToDevice);

	check_cuda_error("rothlin cells");

	//and we are done here
	return thrust::device_vector<int>(t_cell_offsets, t_cell_offsets+cell_count+1);


	//so, what kind of black magic is this?

	//---------------------------------*\
	//
	//	EXAMPLE:
	//
	//
	// [0] Input:
	//
	//	keys of the particles: [ 1 1 2 2 2 3 3 6 6 ]
	//
	//	say we have 12 boxes and boxes with a higher index than 6 are empy (so is 0, 4, 5)
	//
	// [1] After unique by key copy:
	//
	//	indices: [ 0 2 5 7 9 ]
	//	map: [ 1 2 3 6 ]
	//
	//	the indices are pointers into the key vector elements which are not equal to their left neighbor (i.e. the start of boxes)
	//	by invalidating the indices and counting the number of valid elements after the unique by key copy we get the number of
	//	occupied boxes
	//
	// [2] After transforming the indices:
	//
	//	stencil: [ 2 3 2 2 ]
	//
	//	the stencil gives the number of particles in the occupied boxes
	//
	// [3] After scatter:
	//
	//	boxes before scan: [ 0 2 3 2 0 0 2 0 0 0 0 0 ]
	//
	//	the map above can be used to scatter those counts to their appropriate positions
	//
	// [4] After inclusive scan:
	//
	//	boxes after scan: [ 0 0 2 5 7 7 7 9 9 9 9 9 9 ]
	//
	//	with the counts in place (zeros padded for empty boxes), an inclusive scan yields the offsets wanted
	//
	//---------------------------------*/
}

void grid_gpu_rothlin::get_cells(particle_gpu *particles, int *cell_start, int *cell_end) {
	//bit wasteful but makes interface consistent with grid_green
	//		runtime impact below 2%
	thrust::device_vector<int> cells = do_get_cells(particles->hash, particles->N);

	thrust::copy(cells.begin(),   cells.begin() + cells.size()-1, cell_start);
	thrust::copy(cells.begin()+1, cells.end(), cell_end);
}

void grid_gpu_rothlin::assign_random_hashes(particle_gpu *particles) const {
	unsigned int num_part = particles->N;

	//keys to sort by
	int *random_hashes = new int[num_part];
	for (unsigned int i = 0; i < num_part; i++) {
		random_hashes[i] = rand()/((double) RAND_MAX)*num_part;
	}
	cudaMemcpy(particles->hash, random_hashes, sizeof(int)*num_part, cudaMemcpyHostToDevice);
	delete[] random_hashes;
}

void grid_gpu_rothlin::sort(particle_gpu *particles) const {
	unsigned int num_part = particles->N;

	thrust::device_vector<int> t_idx(particles->N_init);
	thrust::sequence(t_idx.begin(), t_idx.end());

	//keys to sort by
	thrust::device_ptr<int> t_hashes(particles->hash);

	//device pointers to data to be sorted
	thrust::device_ptr<float4_t> t_pos(particles->pos);
	thrust::device_ptr<float4_t> t_vel(particles->vel);
#ifdef TVF
	thrust::device_ptr<float4_t> t_vel_adv(particles->vel_adv);
#endif
	thrust::device_ptr<float4_t> t_vel_bc(particles->vel);
	thrust::device_ptr<float3_t> t_fc(particles->fc);
	thrust::device_ptr<float3_t> t_ft(particles->fc);
	thrust::device_ptr<float_t>  t_h(particles->h);
	thrust::device_ptr<float_t>  t_rho(particles->rho);
	thrust::device_ptr<mat3x3_t> t_S(particles->S);
	thrust::device_ptr<float_t>  t_eps_pl(particles->eps_pl);
	thrust::device_ptr<float_t>  t_eps_pl_dot(particles->eps_pl_dot);
	thrust::device_ptr<float_t>  t_T(particles->T);
	thrust::device_ptr<float_t>  t_fixed(particles->fixed);
	thrust::device_ptr<float_t>  t_blanked(particles->blanked);
	thrust::device_ptr<float_t>  t_tool_particle(particles->tool_particle);
	thrust::device_ptr<int>      t_index(particles->idx);

	thrust::device_ptr<float3_t> t_pos_t(particles->pos_t);
	thrust::device_ptr<float3_t> t_vel_t(particles->vel_t);
#ifdef TVF
	thrust::device_ptr<float3_t> t_vel_adv_t(particles->vel_adv_t);
#endif
	thrust::device_ptr<float_t>  t_rho_t(particles->rho_t);
	thrust::device_ptr<mat3x3_t> t_S_t(particles->S_t);
	thrust::device_ptr<float_t>  t_T_t(particles->T_t);

	//temp vector because gather works out of place
	thrust::device_ptr<int>      tempi(m_tempi);
	thrust::device_ptr<float_t>  temp(m_temp);
	thrust::device_ptr<float2_t> temp2f(m_tempf2);
	thrust::device_ptr<float3_t> temp3f(m_tempf3);
	thrust::device_ptr<float4_t> temp4f(m_tempf4);
	thrust::device_ptr<mat3x3_t> tempmat(m_tempmat);

	//sort index keys according to keys to be used by gather
	thrust::sort_by_key(t_hashes,t_hashes+num_part,t_idx.begin());

	thrust::gather(t_idx.begin(), t_idx.end(), t_index, tempi);
	thrust::copy(tempi,tempi+num_part,t_index);

	thrust::gather(t_idx.begin(), t_idx.end(), t_pos, temp4f);
	thrust::copy(temp4f,temp4f+num_part,t_pos);

	thrust::gather(t_idx.begin(), t_idx.end(), t_vel, temp4f);
	thrust::copy(temp4f,temp4f+num_part,t_vel);

	thrust::gather(t_idx.begin(), t_idx.end(), t_vel_bc, temp4f);
	thrust::copy(temp4f,temp4f+num_part,t_vel_bc);

#ifdef TVF
	thrust::gather(t_idx.begin(), t_idx.end(), t_vel_adv, temp4f);
	thrust::copy(temp4f,temp4f+num_part,t_vel_adv);
#endif

	thrust::gather(t_idx.begin(), t_idx.end(), t_fc, temp3f);
	thrust::copy(temp3f,temp3f+num_part,t_fc);

	thrust::gather(t_idx.begin(), t_idx.end(), t_ft, temp3f);
	thrust::copy(temp3f,temp3f+num_part,t_ft);

	thrust::gather(t_idx.begin(), t_idx.end(), t_h, temp);
	thrust::copy(temp,temp+num_part,t_h);

	thrust::gather(t_idx.begin(), t_idx.end(), t_rho, temp);
	thrust::copy(temp,temp+num_part,t_rho);

	thrust::gather(t_idx.begin(), t_idx.end(), t_S, tempmat);
	thrust::copy(tempmat,tempmat+num_part,t_S);

	thrust::gather(t_idx.begin(), t_idx.end(), t_eps_pl, temp);
	thrust::copy(temp,temp+num_part,t_eps_pl);

	thrust::gather(t_idx.begin(), t_idx.end(), t_eps_pl_dot, temp);
	thrust::copy(temp,temp+num_part,t_eps_pl_dot);

	thrust::gather(t_idx.begin(), t_idx.end(), t_fixed, temp);
	thrust::copy(temp,temp+num_part,t_fixed);

	thrust::gather(t_idx.begin(), t_idx.end(), t_blanked, temp);
	thrust::copy(temp,temp+num_part,t_blanked);

	thrust::gather(t_idx.begin(), t_idx.end(), t_tool_particle, temp);
	thrust::copy(temp,temp+num_part,t_tool_particle);

	thrust::gather(t_idx.begin(), t_idx.end(), t_T, temp);
	thrust::copy(temp,temp+num_part,t_T);

	//---------------------------------------------------------

	thrust::gather(t_idx.begin(), t_idx.end(), t_pos_t, temp3f);
	thrust::copy(temp3f,temp3f+num_part,t_pos_t);

	thrust::gather(t_idx.begin(), t_idx.end(), t_vel_t, temp3f);
	thrust::copy(temp3f,temp3f+num_part,t_vel_t);

#ifdef TVF
	thrust::gather(t_idx.begin(), t_idx.end(), t_vel_adv_t, temp3f);
	thrust::copy(temp3f,temp3f+num_part,t_vel_adv_t);
#endif

	thrust::gather(t_idx.begin(), t_idx.end(), t_rho_t, temp);
	thrust::copy(temp,temp+num_part,t_rho_t);

	thrust::gather(t_idx.begin(), t_idx.end(), t_S_t, tempmat);
	thrust::copy(tempmat,tempmat+num_part,t_S_t);

	thrust::gather(t_idx.begin(), t_idx.end(), t_T_t, temp);
	thrust::copy(temp,temp+num_part,t_T_t);

	//---------------------------------------------------------

	check_cuda_error("after copy");
}

void grid_gpu_rothlin::alloc_arrays(int num_cell, int num_part) {
	cudaMalloc((void**) &m_tempi,sizeof(int)*num_part);
	cudaMalloc((void**) &m_temp,sizeof(float_t)*num_part);
	cudaMalloc((void**) &m_tempf2,sizeof(float2_t)*num_part);
	cudaMalloc((void**) &m_tempf3,sizeof(float3_t)*num_part);
	cudaMalloc((void**) &m_tempf4,sizeof(float4_t)*num_part);
	cudaMalloc((void**) &m_tempmat,sizeof(mat3x3_t)*num_part);
	cudaMalloc((void**) &m_seq,sizeof(int)*num_part);

	cudaMalloc((void**) &m_cell_offsets,sizeof(int)*(num_cell+1));
	cudaMalloc((void**) &m_cell_indices,sizeof(int)*num_cell);
	cudaMalloc((void**) &m_cell_map,sizeof(int)*num_cell);
	cudaMalloc((void**) &m_cell_stencil,sizeof(int)*num_cell);
	cudaMalloc((void**) &m_cell_scatter,sizeof(int)*num_cell);

	cudaMemset((void*) m_tempi, 0, sizeof(int)*num_part);
	cudaMemset((void*) m_temp, 0, sizeof(float)*num_part);
	cudaMemset((void*) m_tempf2, 0, sizeof(float2_t)*num_part);
	cudaMemset((void*) m_tempf4, 0, sizeof(float4_t)*num_part);
	cudaMemset((void*) m_seq, 0, sizeof(float)*num_part);

	cudaMemset((void*) m_cell_offsets, 0, sizeof(int)*(num_cell+1));
	cudaMemset((void*) m_cell_indices, 0, sizeof(int)*num_cell);
	cudaMemset((void*) m_cell_map, 0, sizeof(int)*num_cell);
	cudaMemset((void*) m_cell_stencil, 0, sizeof(int)*num_cell);
	cudaMemset((void*) m_cell_scatter, 0, sizeof(int)*num_cell);

	check_cuda_error();
}

grid_gpu_rothlin::grid_gpu_rothlin(int max_cell, int N) :
			grid_base(max_cell, N) {
	alloc_arrays(max_cell, N);
}

grid_gpu_rothlin::grid_gpu_rothlin(int num_part, float3_t bbmin, float3_t bbmax, float_t h) :
		grid_base(num_part, bbmin, bbmax, h) {
	alloc_arrays(m_num_cell, num_part);
}

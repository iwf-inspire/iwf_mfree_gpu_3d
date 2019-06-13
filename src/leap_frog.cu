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

#include "leap_frog.h"

#include "grid_gpu_green.h"
struct inistate_struct {
	float4_t *pos_init;
	float4_t *vel_init;
	mat3x3_t *S_init;
	float_t  *rho_init;
	float_t  *T_init;
	float_t  *T_init_tool;
};

__global__ void init(particle_gpu particles, inistate_struct inistate) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N_init) return;
	if (particles.blanked[pidx] == 1.) return;

	inistate.pos_init[pidx] = particles.pos[pidx];
	inistate.vel_init[pidx] = particles.vel[pidx];
	inistate.S_init[pidx]   = particles.S[pidx];
	inistate.rho_init[pidx] = particles.rho[pidx];
	inistate.T_init[pidx]   = particles.T[pidx];
}

__global__ void predict(particle_gpu particles, inistate_struct inistate, float_t dt) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N_init) return;
	if (particles.blanked[pidx] == 1.) return;

	particles.pos[pidx].x = inistate.pos_init[pidx].x + 0.5*dt*particles.pos_t[pidx].x;
	particles.pos[pidx].y = inistate.pos_init[pidx].y + 0.5*dt*particles.pos_t[pidx].y;
	particles.pos[pidx].z = inistate.pos_init[pidx].z + 0.5*dt*particles.pos_t[pidx].z;

	particles.vel[pidx].x = inistate.vel_init[pidx].x + 0.5*dt*particles.vel_t[pidx].x;
	particles.vel[pidx].y = inistate.vel_init[pidx].y + 0.5*dt*particles.vel_t[pidx].y;
	particles.vel[pidx].z = inistate.vel_init[pidx].z + 0.5*dt*particles.vel_t[pidx].z;

	particles.S[pidx][0][0]  = inistate.S_init[pidx][0][0] + 0.5*dt*particles.S_t[pidx][0][0];
	particles.S[pidx][0][1]  = inistate.S_init[pidx][0][1] + 0.5*dt*particles.S_t[pidx][0][1];
	particles.S[pidx][0][2]  = inistate.S_init[pidx][0][2] + 0.5*dt*particles.S_t[pidx][0][2];

	particles.S[pidx][1][0]  = inistate.S_init[pidx][1][0] + 0.5*dt*particles.S_t[pidx][1][0];
	particles.S[pidx][1][1]  = inistate.S_init[pidx][1][1] + 0.5*dt*particles.S_t[pidx][1][1];
	particles.S[pidx][1][2]  = inistate.S_init[pidx][1][2] + 0.5*dt*particles.S_t[pidx][1][2];

	particles.S[pidx][2][0]  = inistate.S_init[pidx][2][0] + 0.5*dt*particles.S_t[pidx][2][0];
	particles.S[pidx][2][1]  = inistate.S_init[pidx][2][1] + 0.5*dt*particles.S_t[pidx][2][1];
	particles.S[pidx][2][2]  = inistate.S_init[pidx][2][2] + 0.5*dt*particles.S_t[pidx][2][2];

	particles.rho[pidx]   = inistate.rho_init[pidx] + 0.5*dt*particles.rho_t[pidx];

	particles.T[pidx]     = inistate.T_init[pidx]   + 0.5*dt*particles.T_t[pidx];
}

__global__ void correct(particle_gpu particles, inistate_struct inistate, float_t dt) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N_init) return;
	if (particles.blanked[pidx] == 1.) return;

	particles.pos[pidx].x = inistate.pos_init[pidx].x + dt*particles.pos_t[pidx].x;
	particles.pos[pidx].y = inistate.pos_init[pidx].y + dt*particles.pos_t[pidx].y;
	particles.pos[pidx].z = inistate.pos_init[pidx].z + dt*particles.pos_t[pidx].z;

	particles.vel[pidx].x = inistate.vel_init[pidx].x + dt*particles.vel_t[pidx].x;
	particles.vel[pidx].y = inistate.vel_init[pidx].y + dt*particles.vel_t[pidx].y;
	particles.vel[pidx].z = inistate.vel_init[pidx].z + dt*particles.vel_t[pidx].z;

	particles.S[pidx][0][0]  = inistate.S_init[pidx][0][0] + dt*particles.S_t[pidx][0][0];
	particles.S[pidx][0][1]  = inistate.S_init[pidx][0][1] + dt*particles.S_t[pidx][0][1];
	particles.S[pidx][0][2]  = inistate.S_init[pidx][0][2] + dt*particles.S_t[pidx][0][2];

	particles.S[pidx][1][0]  = inistate.S_init[pidx][1][0] + dt*particles.S_t[pidx][1][0];
	particles.S[pidx][1][1]  = inistate.S_init[pidx][1][1] + dt*particles.S_t[pidx][1][1];
	particles.S[pidx][1][2]  = inistate.S_init[pidx][1][2] + dt*particles.S_t[pidx][1][2];

	particles.S[pidx][2][0]  = inistate.S_init[pidx][2][0] + dt*particles.S_t[pidx][2][0];
	particles.S[pidx][2][1]  = inistate.S_init[pidx][2][1] + dt*particles.S_t[pidx][2][1];
	particles.S[pidx][2][2]  = inistate.S_init[pidx][2][2] + dt*particles.S_t[pidx][2][2];

	particles.rho[pidx]   = inistate.rho_init[pidx] + dt*particles.rho_t[pidx];

	particles.T[pidx]     = inistate.T_init[pidx]   + dt*particles.T_t[pidx];
}

__global__ void do_reset_contact_forces(particle_gpu particles) {
	const unsigned int N = particles.N;

	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;

	//reset contact forces
	particles.fc[pidx].x = 0.;
	particles.fc[pidx].y = 0.;
	particles.fc[pidx].z = 0.;
	particles.ft[pidx].x = 0.;
	particles.ft[pidx].y = 0.;
	particles.ft[pidx].z = 0.;
	particles.n[pidx].x  = 0.;
	particles.n[pidx].y  = 0.;
	particles.n[pidx].z  = 0.;

}

void leap_frog::step(particle_gpu *particles, grid_base *g, bool record_forces) {
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N_init + block_size-1) / block_size);
	dim3 dB(block_size);

	inistate_struct inistate;
	inistate.pos_init = pos_init;
	inistate.vel_init = vel_init;
	inistate.S_init   = S_init;
	inistate.rho_init = rho_init;
	inistate.T_init   = T_init;

	if (global_blanking) {
		global_blanking->update();
		perform_blanking(particles, global_blanking);
	}

	//spatial sorting
	g->update_geometry(particles);
	g->assign_hashes(particles);
	if (global_blanking) {
		g->adapt_particle_number(particles);
	}
	g->sort(particles);
	g->get_cells(particles, cell_start, cell_end);

	init<<<dG,dB>>>(*particles, inistate);
	predict<<<dG,dB>>>(*particles, inistate, global_time_dt);

	material_eos(particles);
	corrector_artificial_stress(particles);

	interactions_setup_geometry_constants(g);
	interactions_monaghan(particles, cell_start, cell_end, g->num_cell());

#ifdef Thermal_Conduction_PSE
	interactions_heat_pse(particles, cell_start, cell_end, g->num_cell());
#endif

	material_stress_rate_jaumann(particles);
	contmech_continuity(particles);
	contmech_momentum(particles);
	contmech_advection(particles);

#ifndef NDEBUG
	debug_invalidate(particles);
#endif

	correct<<<dG,dB>>>(*particles, inistate, global_time_dt);

	plasticity_johnson_cook(particles);

	do_reset_contact_forces<<<dG,dB>>>(*particles);

	if (record_forces) {
		global_tool_forces->reset();
	}

	for (auto globaltool : global_tool) {	// Iterate over contacting tool(s)
		globaltool->update_tool();
		globaltool->compute_contact_force(particles, record_forces);
	}
	material_fric_heat_gen(particles, global_tool[0]->get_vel());

	perform_boundary_conditions(particles);
	perform_boundary_conditions_thermal(particles);

	actions_move_tool_particles(particles, global_tool[0]);

#ifndef NDEBUG
	debug_check_valid_full(particles);
#endif
	check_cuda_error();

}

leap_frog::leap_frog(unsigned int num_part, unsigned int num_cell) {
	cudaMalloc((void **) &pos_init, sizeof(float4_t)*num_part);
	cudaMalloc((void **) &vel_init, sizeof(float4_t)*num_part);
	cudaMalloc((void **) &S_init,   sizeof(mat3x3_t)*num_part);
	cudaMalloc((void **) &rho_init,   sizeof(float_t)*num_part);
	cudaMalloc((void **) &T_init,   sizeof(float_t)*num_part);

	cudaMalloc((void **) &cell_start, sizeof(int)*num_cell);
	cudaMalloc((void **) &cell_end,   sizeof(int)*num_cell);
}

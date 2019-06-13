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

//this file contains all particle ``actions'', i.e. functions that can be expressed with a single loop over the particles, like so
//for i=1:N
//		do_stuff(particles[i]);
//end

#ifndef ACTIONS_GPU_H_
#define ACTIONS_GPU_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "blanking.h"
#include "particle_gpu.h"
#include "constants_structs.h"
#include "types.h"

//global state for timekeeping
extern float_t global_time_dt;
extern float_t global_time_final;
extern float_t global_time_current;
extern int global_time_step;

//equation of state for hydrostatic part of the stress (i.e. pressure)
void material_eos(particle_gpu *particles);

//artificial stress due to monaghan, gray, 2001
void corrector_artificial_stress(particle_gpu *particles);
void material_stress_rate_jaumann(particle_gpu *particles);

//turn frictional work into heat
void material_fric_heat_gen(particle_gpu *particles, vec3_t vel);

//balance equations
void contmech_continuity(particle_gpu *particles);
void contmech_momentum(particle_gpu *particles);
void contmech_advection(particle_gpu *particles);

//blank (i.e. deactivate particles) due to various criteria (see blanking class)
//	this allows for local divergence of the simulation
void perform_blanking(particle_gpu *particles, blanking *global_blanking);
void perform_blanking_dbg(particle_gpu *particles, blanking *global_blanking);

//johnson cook flow stress model using radial return
void plasticity_johnson_cook(particle_gpu *particles);

//simple dirichlet boundaries (i.e. fixed values for tempreature and displacement)
void perform_boundary_conditions_thermal(particle_gpu *particles);
void perform_boundary_conditions(particle_gpu *particles);

//move particle contained in tool (subject to thermal solver only), along with the tool
void actions_move_tool_particles(particle_gpu *particles, tool_3d_gpu *tool);

//set up various constants for the correct working of the actions contained in this header file
void actions_setup_johnson_cook_constants(joco_constants joco);
void actions_setup_physical_constants(phys_constants phys);
void actions_setup_corrector_constants(corr_constants corr);
void actions_setup_thermal_constants_wp(trml_constants thrm);
void actions_setup_thermal_constants_tool(trml_constants thrm);

//check for nan values, either due to divergence or bugs
void debug_check_valid(particle_gpu *particles);
void debug_check_valid_full(particle_gpu *particles);
void debug_invalidate(particle_gpu *particles);

#endif /* ACTIONS_GPU_H_ */

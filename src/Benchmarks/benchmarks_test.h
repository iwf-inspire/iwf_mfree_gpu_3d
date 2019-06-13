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

//contains various preliminary benchmarks that verify the basic workings of the solver, mostly impact computations
//
// Basic Tests:
//	- setup_rings: two colliding spheres, actually. 3d version of the famous rubber ring impact, c.f. e.g. gray, monaghan 2001
//	- setup_disk: a spinning disk. use CSPM correction to appreciate that linear complete kernels conserve angular momentum while
//                the standard SPH does not
//
// Contact Algorithm & Plasticity Tests
//  - setup_ring_contact: either plastic or rubber ring impacts a rigid wall
//  - setup_solid_ball_plastic_wall_impact: solid ball shoots true a metal plate
//  - setup_plastic_ball_plastic_wall_impact: metal ball shoots a metal plate.
//
// Corrector Tests:
// - setup_cylinder_impact: sometimes its quite hard to see the working of the artificial stress in 3d simulations since they seem less
//                          prone to numerical fracture, especially at larger hdx. thin walled cylinders seem to be an exception to this case;
//                          fraction can be observed if use_art_stress is set to false but not if set to true. this is true for a wide range
//                          of speeds and hdx

#ifndef BENCHMARKS_H_
#define BENCHMARKS_H_

#include <thrust/device_vector.h>
#include "../constants_structs.h"
#include "../particle_gpu.h"
#include "../types.h"
#include "../actions_gpu.h"
#include "../interactions_gpu.h"
#include "../types.h"
#include "../grid_gpu_base.h"
#include "../grid_gpu_rothlin.h"
#include "../json_reader.h"
#include "../surface_tri_mesh.h"
#include "../geometry_utils.h"
#include "../ldynak_reader.h"

extern std::vector<tool_3d_gpu *> global_tool;
extern blanking *global_blanking;

extern float_t global_time_dt;
extern float_t global_time_final;

particle_gpu *setup_rings(int nbox, grid_base **grid);			//[X] re-tested 11.06.2019, MR
particle_gpu *setup_disk(int nbox, grid_base **grid);			//[X] re-tested 11.06.2019, MR
																//    NOTES: - CSPM rotates correctly, SPH wobbles back an forth, as expected
																//           - Quite a bit of particle disarray at the end both for single and double precision
particle_gpu *setup_ring_contact(int nbox, grid_base **grid);   //[X] tested 11.06.2019, MR

particle_gpu *setup_solid_ball_plastic_wall_impact(int nbox, grid_base **grid);   //[X] tested 11.06.2019, MR
particle_gpu *setup_plastic_ball_plastic_wall_impact(int nbox, grid_base **grid); //[X] tested 11.06.2019, MR

particle_gpu *setup_cylinder_impact(int nbox, grid_base **grid, bool use_art_stress = false); //[X] tested 11.06.2019, MR
#endif /* BENCHMARKS_H_ */

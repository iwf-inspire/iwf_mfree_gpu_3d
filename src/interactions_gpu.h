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

//the functions in this header implement all the function of the algorithms which can be expressed as a particle ``interaction'', i.e.
// for i=1:N
//	  for j in nbh(i)
//		interact(i,j)
//    end
// end
//
// - the nbh(i) lookup is performed by spatial hashing / cell lists, using one of the two derived grid classes
// - the functions contained herein are by far the computationally most expensive ones and have been quite extensively optimized
// - read only values are bound to textures, write to values are restricted
// - all of these kernels spill to local memory, but splitting them up into smaller kernels to save registers decreases performance
// - thus all interaction described in monaghans / gray 2001 paper are collapsed into one kernel
//		- this kernel also computes thermal conduction using the brookshaw approximation if the symbol in types.h is defined
//      - this kernel is enhanced using CSPM / randles libersky corrections if the symbol in types.h is defined
// - alternatively, thermals can also be computed using particle strength exchange (PSE) using the appropriate kernel
//		- this is quite a bit more expensive since an exponential needs to be computed per kernel, and an additional iteration over the
//        cell lists is required

#ifndef INTERACTIONS_GPU_H_
#define INTERACTIONS_GPU_H_

#include <stdio.h>

#include "particle_gpu.h"
#include "grid_gpu_green.h"
#include "constants_structs.h"
#include "tool_3d_gpu.h"

extern float_t global_time_dt;
extern float_t global_time_final;
extern float_t global_time_current;
extern int global_time_step;

void interactions_setup_geometry_constants(grid_base *g);

void interactions_monaghan(particle_gpu *particles, const int *cell_start, const int *cell_end, int num_cell);
void interactions_heat_pse(particle_gpu *particles, const int *cell_start, const int *cell_end, int num_cell);

void interactions_setup_physical_constants(phys_constants phys);
void interactions_setup_corrector_constants(corr_constants corr);
void interactions_setup_thermal_constants_workpiece(trml_constants trml);
void interactions_setup_thermal_constants_tool(trml_constants trml, tool_3d_gpu *tool);

#endif /* INTERACTIONS_GPU_H_ */

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

#ifndef BENCHMARKS_MR_H_
#define BENCHMARKS_MR_H_

#include <thrust/device_vector.h>

#include <algorithm>

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
#include "../vtk_reader.h"
#include "interp_utils.h"

//globals holding the tools and the blanking mechanism
extern std::vector<tool_3d_gpu *> global_tool;
extern blanking *global_blanking;

//globals holding time step and termination time
extern float_t global_time_dt;
extern float_t global_time_final;

//generally, there are two tool files delivered with the package at hand
//	pin25_reduziert_20062018 is a tilted diamond with minor and major cutting edge that was scaled for optimal runtime
//  Diamant_von_Mansur_04072018_SDB1125-2025-D851 is an unscaled diamond, courtesy of Dr. Akbari

//furthermore, two sets of material parameters for Ti6Al4V are available
//		The set published by Khan et al., 2004, "Quasi-static and dynamic loading responses and constitutive modeling of titanium alloys"
//			also used by Dr. Akbari
//		The set published by Johnson et al., 1985, "Strength and fracture characteristics of a titanium alloy (. 06al,. 04v) subjected to various strains, strain rates, temperatures and pressures"
//			also used by Dr. Rüttimann

//multi grain cutting simulation using 5 tools, this function is used to start the simulations in section 5.2 of the IJMMS paper
//	some grains can be commented out to achieve different patterns
//  temperature field can be sampled from a result file (interp_steady_temp boolean)
//  a result file can be read to initialize positions, useful to simulate a pre-cut but cooled down and de-stressed work-piece
//		see commented out lines below particle generation code
particle_gpu *setup_single_grain_5tool(grid_base **grid);

//single grain simulation using the scaled diamond, not present in the paper
particle_gpu *setup_single_grain_1tool(grid_base **grid);

//single grain simulation using the unscaled diamond. use this simulation to appreciate the chip curling in corrected simulations
//		corrected refers to corrected kernel, i.e. Randles Libersky / CSPM correction
particle_gpu *setup_single_grain_1tool_realscale(grid_base **grid);

//single grain simulation using the scaled diamond on a very long work piece. This used as a sentinel simulation to retrieve
//the initial temperature field for the 5tool simulation. This simulation uses a moving spatial hashing grid and blanking structure
//to only ever activate a small portion of the whole work piece length
particle_gpu *setup_single_grain_1tool_trml_steady(grid_base **grid);

#endif /* BENCHMARKS_MR_H_ */

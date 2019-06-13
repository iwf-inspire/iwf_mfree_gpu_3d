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

// leap frog time integration. This file contains the basic simulation loop

#ifndef LEAP_FROG_H_
#define LEAP_FROG_H_

#include "grid_gpu_base.h"
#include "particle_gpu.h"

#include "actions_gpu.h"
#include "blanking.h"
#include "interactions_gpu.h"
#include "tool_3d_gpu.h"

#include "types.h"

extern std::vector<tool_3d_gpu *> global_tool;
extern tool_forces *global_tool_forces;
extern blanking *global_blanking;

class leap_frog{
private:
	float4_t *pos_init;
	float4_t *vel_init;
	mat3x3_t *S_init;
	float_t  *rho_init;
	float_t  *T_init;

	int *cell_start;
	int *cell_end;
public:
	void step(particle_gpu *particles, grid_base *g, bool record_forces);
	leap_frog(unsigned int num_part, unsigned int num_cell);
};

#endif /* LEAP_FROG_H_ */

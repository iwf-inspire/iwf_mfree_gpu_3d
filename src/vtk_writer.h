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

//write simulation state to legacy vtk format
//	- if a vector of tools is passed, a separate file is opened for each tool
//  - if a single file containing all the tools is desired, use _write_unified

#ifndef VTK_WRITER_H_
#define VTK_WRITER_H_

#include <stdio.h>
#include <math.h>
#include <numeric>

#include "blanking.h"
#include "types.h"
#include "particle_gpu.h"
#include "tool_3d_gpu.h"

extern blanking *global_blanking;
extern int global_time_step;

void vtk_writer_write(particle_gpu *particles);
void vtk_writer_write(tool_3d_gpu *tool);
void vtk_writer_write(particle_gpu *particles, tool_3d_gpu *tool);
void vtk_writer_write(tool_3d_gpu *tool, particle_gpu *particles);
void vtk_writer_write(std::vector<tool_3d_gpu *> tool, particle_gpu *particles);
void vtk_writer_write(particle_gpu *particles, std::vector<tool_3d_gpu *> tool);

void vtk_writer_write_unified(std::vector<tool_3d_gpu *> tools, int step, char filename[256]);
void vtk_writer_write(tool_3d_gpu *tool, char filename[256]);

#endif /* VTK_WRITER_H_ */

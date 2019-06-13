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

//tool on the cpu. Basically the same functionality as on the gpu
//	- designed as an intermediary construct to eventually build a tool_gpu only
//  - flips triangle normals to outside if necessary
//		- this is achieved by constructing a point which is outside for sure, then check if the inside/outside test using the triangle normals
//		  gives the correct result. otherwise the normals are flipped
//		- this relaxes the requirements on the input files (*.k / *.json) on a specific ordering of the triangle verticies
//	- the contact methods are used to sample the tool on the gpu only, they are not used during the simulation run
//	- as for the contact algorithms, see the notes in tool_3d_gpu.h

#ifndef TOOL_3D_H_
#define TOOL_3D_H_

#include <vector>
#include <glm/glm.hpp>
#include <assert.h>

#include "grid_cpu_tri.h"
#include "surface_tri_mesh.h"
#include "types.h"
#include "tool_3d_gpu.h"

class tool_3d {

	friend class tool_3d_gpu;

public:
	void dbg_print_bbox() const;

	void get_bbox(vec3_t &bbmin, vec3_t &bbmax) const;

	void dbg_dump_tris() const;

	const std::vector<mesh_triangle> &get_triangles() const;

	const std::vector<vec3_t> &get_positions() const;

	bool contact(vec3_t qp, vec3_t &cp, vec3_t &dir) const;
	bool contact_n2(vec3_t qp, vec3_t &cp, vec3_t &dir) const;
	bool contact_safe(vec3_t qp) const;
	bool contact_safe_n2(vec3_t qp) const;

	const std::vector<vec3_t> sample(float_t dx) const;

	//construct tool, (angle averaged) edge and face normals need to be present!
	//	h is a characteristic length of the mesh (e.g. average edge length)
	tool_3d(std::vector<mesh_triangle> triangles, std::vector<vec3_t> positions, float_t h_avg);

private:
	//grid needed to construct cell lists
	grid_cpu_tri m_grid;

	//cell lists
	mesh_triangle *m_boxed_tris;
	unsigned int *m_cells;
	unsigned int m_num_cells;
	unsigned int m_num_boxed_tris;

	//triangles and positions in original order, needed for output only
	std::vector<mesh_triangle> m_triangles;
	std::vector<vec3_t> m_positions;
};

#endif /* TOOL_3D_H_ */

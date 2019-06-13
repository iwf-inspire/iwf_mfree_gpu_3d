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

//this class enables the construction of a hashing structure for triangles
//	- the construction is a bit different when compared to the same principle for particles, since triangles
//    have an extension in 3-space, hence a triangle may be contained in more than one box
//  - this class only _constructs_ the hashing structure, it is then used by the tool class (both CPU and GPU)
//	- construction is done using a slow, but very readable algorithm. This is ok since the structure is constructed only once, in the begining
//    of the simulation run
//  - since the tool does not deform, i.e. performs rigid body modes only, the spatial hashing structure never needs to be recomputed
//    (theoretically it could be pre-processed once per tool mesh and then stored on disk along with the mesh itself)

#ifndef GRID_CPU_TRI_H_
#define GRID_CPU_TRI_H_

#include "surface_tri_mesh.h"

class grid_cpu_tri {
public:
	void update_geometry_mesh(std::vector<mesh_triangle> tris, float_t h_avg);
	void get_cells(const std::vector<mesh_triangle> &tris, unsigned int *&cells, mesh_triangle *&out_tris, unsigned int &num_boxes, unsigned int &num_boxed_tris) const;
	void get_bbox(vec3_t &bbmin, vec3_t &bbmax) const;
	unsigned int unhash_pos(vec3_t pos, unsigned int &ix, unsigned int &iy, unsigned int &iz) const;

	unsigned int nx() const;
	unsigned int ny() const;
	unsigned int nz() const;

	float_t bbmin_x() const;
	float_t bbmin_y() const;
	float_t bbmin_z() const;

	float_t dx() const;

private:

	float_t m_dx;
	float_t m_lx, m_ly, m_lz;
	unsigned int m_nx, m_ny, m_nz;
	unsigned int m_num_cell;

	float_t m_bbmin_x, m_bbmax_x;
	float_t m_bbmin_y, m_bbmax_y;
	float_t m_bbmin_z, m_bbmax_z;
};

#endif /* GRID_CPU_TRI_H_ */

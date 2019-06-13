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

#include "grid_cpu_tri.h"

unsigned int grid_cpu_tri::nx() const {
	return m_nx;
}

unsigned int grid_cpu_tri::ny() const {
	return m_ny;
}

unsigned int grid_cpu_tri::nz() const {
	return m_nz;
}

float_t grid_cpu_tri::bbmin_x() const {
	return m_bbmin_x;
}

float_t grid_cpu_tri::bbmin_y() const {
	return m_bbmin_y;
}

float_t grid_cpu_tri::bbmin_z() const {
	return m_bbmin_z;
}

float_t grid_cpu_tri::dx() const {
	return m_dx;
}

void grid_cpu_tri::update_geometry_mesh(std::vector<mesh_triangle> tris, float_t h_avg) {
	m_bbmin_x = DBL_MAX; m_bbmax_x = -DBL_MAX;
	m_bbmin_y = DBL_MAX; m_bbmax_y = -DBL_MAX;
	m_bbmin_z = DBL_MAX; m_bbmax_z = -DBL_MAX;

	for (auto it = tris.begin(); it != tris.end(); ++it) {

		vec3_t bbmin, bbmax;
		it->bbox(bbmin, bbmax);

		m_bbmin_x = fmin(bbmin.x, m_bbmin_x);
		m_bbmin_y = fmin(bbmin.y, m_bbmin_y);
		m_bbmin_z = fmin(bbmin.z, m_bbmin_z);

		m_bbmax_x = fmax(bbmax.x, m_bbmax_x);
		m_bbmax_y = fmax(bbmax.y, m_bbmax_y);
		m_bbmax_z = fmax(bbmax.z, m_bbmax_z);
	}

	//some nudging to prevent round off errors

	m_bbmin_x -= 1e-4;
	m_bbmin_y -= 1e-4;
	m_bbmin_z -= 1e-4;
	m_bbmax_x += 1e-4;
	m_bbmax_y += 1e-4;
	m_bbmax_z += 1e-4;

	m_dx = h_avg;

	m_lx = m_bbmax_x - m_bbmin_x;
	m_ly = m_bbmax_y - m_bbmin_y;
	m_lz = m_bbmax_z - m_bbmin_z;

	m_nx = ceil(m_lx/m_dx);
	m_ny = ceil(m_ly/m_dx);
	m_nz = ceil(m_lz/m_dx);
	m_num_cell = m_nx*m_ny*m_nz;
}

unsigned int grid_cpu_tri::unhash_pos(vec3_t pos, unsigned int &ix, unsigned int &iy, unsigned int &iz) const {
	ix = (unsigned int) ((pos.x - m_bbmin_x)/m_dx);
	iy = (unsigned int) ((pos.y - m_bbmin_y)/m_dx);
	iz = (unsigned int) ((pos.z - m_bbmin_z)/m_dx);

	return ix*(m_ny)*(m_nz) + iy*(m_nz) + iz;
}

void grid_cpu_tri::get_cells(const std::vector<mesh_triangle> &tris, unsigned int *&cells, mesh_triangle *&out_tris, unsigned int &num_boxes, unsigned int &num_boxed_tris) const {

	std::vector<std::vector<mesh_triangle>> m_grid(m_nx*m_ny*m_nz);

	for (auto it = tris.begin(); it != tris.end(); ++it) {

		vec3_t min, max;
		it->bbox(min, max);

		unsigned int ixlo, iylo, izlo;
		unsigned int ixhi, iyhi, izhi;

		unhash_pos(min, ixlo, iylo, izlo);
		unhash_pos(max, ixhi, iyhi, izhi);

		for (unsigned int ii = ixlo; ii <= ixhi; ii++) {
			for (unsigned int jj = iylo; jj <= iyhi; jj++) {
				for (unsigned int kk = izlo; kk <= izhi; kk++) {

					unsigned int linidx = ii*(m_ny)*(m_nz) + jj*(m_nz) + kk;

					vec3_t cube_lo;
					vec3_t cube_hi;

					cube_lo.x = m_bbmin_x + ii*m_dx;
					cube_lo.y = m_bbmin_y + jj*m_dx;
					cube_lo.z = m_bbmin_z + kk*m_dx;

					cube_hi.x = m_bbmin_x + (ii+1)*m_dx;
					cube_hi.y = m_bbmin_y + (jj+1)*m_dx;
					cube_hi.z = m_bbmin_z + (kk+1)*m_dx;

					if (it->cover_aabb(cube_lo, cube_hi)) {
						m_grid[linidx].push_back(*it);
					}
				}
			}
		}
	}

	unsigned int flat_size = 0;
	for (auto it = m_grid.begin(); it != m_grid.end(); ++it) {
		flat_size += it->size();
	}

	unsigned int *boxes = new unsigned int[m_grid.size()+1];
	mesh_triangle *boxed_tris = new mesh_triangle[flat_size];

	unsigned int cur_offset = 0;
	for (unsigned int i = 0; i < m_grid.size(); i++) {
		boxes[i] = cur_offset;
		cur_offset += m_grid[i].size();
	}
	boxes[m_grid.size()] = cur_offset;

	unsigned int iter = 0;
	for (unsigned int i = 0; i < m_grid.size(); i++) {
		for (auto it = m_grid[i].begin(); it != m_grid[i].end(); ++it) {
			boxed_tris[iter] = *it;
			iter++;
		}
	}

	cells = boxes;
	out_tris = boxed_tris;
	num_boxes = m_grid.size();
	num_boxed_tris = flat_size;

//	{
//		FILE *fp = fopen("tri_grid.txt", "w+");
//		for (int i = 0; i < m_nx; i++) {
//			for (int j = 0; j < m_ny; j++) {
//				for (int k = 0; k < m_nz; k++) {
//					fprintf(fp, "%d %d %d %d\n", i, j, k, (int) m_grid[i*m_ny*m_nz + j *m_nz + k].size());
//				}
//			}
//		}
//		fclose(fp);
//	}
}

void grid_cpu_tri::get_bbox(vec3_t &bbmin, vec3_t &bbmax) const {
	bbmin.x = m_bbmin_x;
	bbmin.y = m_bbmin_y;
	bbmin.z = m_bbmin_z;

	bbmax.x = m_bbmax_x;
	bbmax.y = m_bbmax_y;
	bbmax.z = m_bbmax_z;
}

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

#include "tool_3d.h"

static bool valid(vec3_t normal) {
	return !(normal.x == 0. && normal.y == 0. && normal.z==0.);
}

static float_t dist_squared(vec3_t p1, vec3_t p2) {
	float_t dx = p1.x - p2.x;
	float_t dy = p1.y - p2.y;
	float_t dz = p1.z - p2.z;

	return dx*dx + dy*dy + dz*dz;
}

void tool_3d::get_bbox(vec3_t &bbmin, vec3_t &bbmax) const {
	m_grid.get_bbox(bbmin, bbmax);
}

void tool_3d::dbg_dump_tris() const {
	FILE *fp = fopen("tridump.txt", "w+");
	for (auto it = m_triangles.begin(); it != m_triangles.end(); ++it) {
		fprintf(fp, "%f %f %f %f\n", it->p1.x, it->p2.x, it->p3.x, it->p1.x);
		fprintf(fp, "%f %f %f %f\n", it->p1.y, it->p2.y, it->p3.y, it->p1.y);
		fprintf(fp, "%f %f %f %f\n", it->p1.z, it->p2.z, it->p3.z, it->p1.z);
	}
	fclose(fp);
	printf("dumped triangles!\n");
}

const std::vector<mesh_triangle> &tool_3d::get_triangles() const {
	return m_triangles;
}

const std::vector<vec3_t> &tool_3d::get_positions() const {
	return m_positions;
}

bool tool_3d::contact(vec3_t qp, vec3_t &cp, vec3_t &dir) const {
	vec3_t closest_point;
	vec3_t normal;

	unsigned int ix,iy,iz;
	unsigned int box_idx = m_grid.unhash_pos(qp, ix, iy, iz);

	float_t min_dist = DBL_MAX;

	bool first_hit_empty = (m_cells[box_idx] == m_cells[box_idx+1]);

	if (m_cells[box_idx] != m_cells[box_idx+1]) {
		//test nodes in box

		for (unsigned int pidx = m_cells[box_idx]; pidx < m_cells[box_idx+1]; pidx++) {
			vec3_t cp_tri, np_tri;
			m_boxed_tris[pidx].closest_point(qp, cp_tri, np_tri);
			float_t cur_dist = dist_squared(cp_tri, qp);

			if (cur_dist < min_dist) {
				min_dist = cur_dist;
				closest_point = cp_tri;
				normal = np_tri;
			}
		}
	}

	//look at neighbors until closest point is found
	int offset = 1;
	bool all_done = false;
	bool all_empty_so_far = first_hit_empty;

	unsigned int m_nx = m_grid.nx();
	unsigned int m_ny = m_grid.ny();
	unsigned int m_nz = m_grid.nz();

	while (true) {

		int low_i  = (int) ix-offset < 0 ? 0 : (int) ix-offset;
		int low_j  = (int) iy-offset < 0 ? 0 : (int) iy-offset;
		int low_k  = (int) iz-offset < 0 ? 0 : (int) iz-offset;

		int high_i = ix+offset+1 > m_nx ? m_nx : ix+offset+1;
		int high_j = iy+offset+1 > m_ny ? m_ny : iy+offset+1;
		int high_k = iz+offset+1 > m_nz ? m_nz : iz+offset+1;

		for (int ni = low_i; ni < high_i; ni++) {
			for (int nj = low_j; nj < high_j; nj++) {
				for (int nk = low_k; nk < high_k; nk++) {

					bool i_on_boundary = ni == low_i || ni == high_i-1;
					bool j_on_boundary = nj == low_j || nj == high_j-1;
					bool k_on_boundary = nk == low_k || nk == high_k-1;

					// only look at boundaries of cube (do not re check boxes)
					if (!(i_on_boundary || j_on_boundary || k_on_boundary)) {
						continue;
					}

					unsigned int box_idx = ni*(m_ny)*(m_nz) + nj*(m_nz) + nk;

					if (m_cells[box_idx] != m_cells[box_idx+1]) {

						for (unsigned int pidx = m_cells[box_idx]; pidx < m_cells[box_idx+1]; pidx++) {
							all_empty_so_far = false;

							vec3_t cp_tri, np_tri;
							m_boxed_tris[pidx].closest_point(qp, cp_tri, np_tri);
							float_t cur_dist = dist_squared(cp_tri, qp);

							if (cur_dist < min_dist) {
								min_dist = cur_dist;
								closest_point = cp_tri;
								normal = np_tri;
							}
						}
					}

				}
			}
		}

		if (all_done) {
			break;
		}

		if (!all_empty_so_far) {
			offset += 1;
			all_done = true;
			continue;
		}

		offset += 1;
	}

	vec3_t diff = closest_point - qp;
	float_t sign = (glm::dot(diff,normal) > 0.) ? 1. : -1.;

	if (sign > 0.) {
		return false;
	}

	cp = closest_point;
	dir = sign*normal;

	return true;
}

bool tool_3d::contact_n2(vec3_t qp, vec3_t &cp, vec3_t &dir) const {
	vec3_t closest_point;
	vec3_t normal;

	float_t min_dist = DBL_MAX;

	for (unsigned int pidx = 0; pidx < m_triangles.size(); pidx++) {
		vec3_t cp_tri, np_tri;
		m_triangles[pidx].closest_point(qp, cp_tri, np_tri);
		float_t cur_dist = dist_squared(cp_tri, qp);

		if (cur_dist < min_dist) {
			min_dist = cur_dist;
			closest_point = cp_tri;
			normal = np_tri;
		}
	}

	vec3_t diff = closest_point - qp;
	float_t sign = (glm::dot(diff,normal) > 0.) ? 1. : -1.;

	if (sign > 0.) {
		return false;
	}

	cp = closest_point;
	dir = sign*normal;

	return true;
}

bool tool_3d::contact_safe_n2(vec3_t qp) const {

	vec3_t ray_o = qp;			//ray origin
	vec3_t ray_dir(0.,0.,1.);	//arbitrary direction

	unsigned int num_inter = 0;
	for (unsigned int pidx = 0; pidx < m_triangles.size(); pidx++) {
		bool inter = m_triangles[pidx].ray_intersect(ray_o, ray_dir);

		if (inter) num_inter++;
	}

	return ((num_inter % 2) != 0);
}

bool tool_3d::contact_safe(vec3_t qp) const {
	unsigned int ix,iy,iz;
	unsigned int box_idx = m_grid.unhash_pos(qp, ix, iy, iz);

	vec3_t ray_o = qp;			//ray origin
	vec3_t ray_dir(0.,0.,1.);	//arbitrary direction

	unsigned int m_nx = m_grid.nx();
	unsigned int m_ny = m_grid.ny();
	unsigned int m_nz = m_grid.nz();

	unsigned int num_inter = 0;
	float_t t = 0.;
	for (; iz < m_nz; ++iz) {
		unsigned int box_idx = ix*(m_ny)*(m_nz) + iy*(m_nz) + iz;
		for (unsigned int pidx = m_cells[box_idx]; pidx < m_cells[box_idx+1]; pidx++) {
			bool inter = m_boxed_tris[pidx].ray_intersect(ray_o, ray_dir, t);

			if (inter) {	//triangle may be in more than one box. make sure that intersection is counted only once

				vec3_t inter_p = ray_o + t*ray_dir;

				float_t lo_x = m_grid.bbmin_x() + ix*m_grid.dx();
				float_t lo_y = m_grid.bbmin_y() + iy*m_grid.dx();
				float_t lo_z = m_grid.bbmin_z() + iz*m_grid.dx();

				float_t hi_x = m_grid.bbmin_x() + (ix+1)*m_grid.dx();
				float_t hi_y = m_grid.bbmin_y() + (iy+1)*m_grid.dx();
				float_t hi_z = m_grid.bbmin_z() + (iz+1)*m_grid.dx();

				bool in_x = lo_x <= inter_p.x && inter_p.x < hi_x;
				bool in_y = lo_y <= inter_p.y && inter_p.y < hi_y;
				bool in_z = lo_z <= inter_p.z && inter_p.z < hi_z;

				if (in_x && in_y && in_z) {
					num_inter++;
				}
			}
		}
	}

	return ((num_inter % 2) != 0);
}

const std::vector<vec3_t> tool_3d::sample(float_t dx) const {

	vec3_t bbmin, bbmax;
	get_bbox(bbmin, bbmax);

	float_t lx = bbmax.x - bbmin.x;
	float_t ly = bbmax.y - bbmin.y;
	float_t lz = bbmax.z - bbmin.z;

	unsigned int nx = ceil(lx/dx);
	unsigned int ny = ceil(ly/dx);
	unsigned int nz = ceil(lz/dx);

	std::vector<vec3_t> samples;

	FILE *fp = fopen("samples.txt", "w+");

	for (unsigned int i = 0; i < nx; i++) {
		for (unsigned int j = 0; j < ny; j++) {
			for (unsigned int k = 0; k < nz; k++) {
				float_t px = bbmin.x + i*dx;
				float_t py = bbmin.y + j*dx;
				float_t pz = bbmin.z + k*dx;

				vec3_t pos(px, py, pz);
				if (contact_safe(pos)) {
					fprintf(fp, "%f %f %f\n", px, py, pz);
					samples.push_back(pos);
				}
			}
		}
	}

	fclose(fp);

	return samples;

}

//construct tool, (angle averaged) edge and face normals need to be present!
//	h is a characteristic length of the mesh (e.g. average edge length)
tool_3d::tool_3d(std::vector<mesh_triangle> triangles, std::vector<vec3_t> positions, float_t h_avg) {
	for (auto it = triangles.begin(); it != triangles.end(); ++it) {
		assert(valid(it->normal));

		assert(valid(it->ne1));
		assert(valid(it->ne2));
		assert(valid(it->ne3));

		assert(valid(it->np1));
		assert(valid(it->np2));
		assert(valid(it->np3));
	}

	m_grid.update_geometry_mesh(triangles, h_avg);
	m_grid.get_cells(triangles, m_cells, m_boxed_tris, m_num_cells, m_num_boxed_tris);

	m_triangles = triangles;
	for (auto it : m_triangles) {
		it.p1_init = it.p1;
		it.p2_init = it.p2;
		it.p3_init = it.p3;
	}
	m_positions = positions;

	//lets determine if normals point inside or outside for this model
	//		it is assumed all normals point on same side, if your models features mixed triangles get a better model

	//construct point that is outside for sure
	vec3_t bbmin,bbmax;
	m_grid.get_bbox(bbmin, bbmax);
	vec3_t outside = bbmax;

	outside.x += 1e-6;
	outside.y += 1e-6;
	outside.z += 1e-6;

	vec3_t cp, n;
	bool is_inside = contact(outside, cp, n);

	// if point that is construced to be outside isnt outside, normals need to be flipped
	if (is_inside) {
		for (auto &it : m_triangles) {
			it.ne1 = -it.ne1;
			it.ne2 = -it.ne2;
			it.ne3 = -it.ne3;

			it.np1 = -it.np1;
			it.np2 = -it.np2;
			it.np3 = -it.np3;

			it.normal = -it.normal;
		}

		for (unsigned int i = 0; i < m_num_boxed_tris; i++) {
			m_boxed_tris[i].ne1 = -m_boxed_tris[i].ne1;
			m_boxed_tris[i].ne2 = -m_boxed_tris[i].ne2;
			m_boxed_tris[i].ne3 = -m_boxed_tris[i].ne3;

			m_boxed_tris[i].np1 = -m_boxed_tris[i].np1;
			m_boxed_tris[i].np2 = -m_boxed_tris[i].np2;
			m_boxed_tris[i].np3 = -m_boxed_tris[i].np3;

			m_boxed_tris[i].normal = -m_boxed_tris[i].normal;
		}
	}

}

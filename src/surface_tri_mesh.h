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

//file holding the classes for a triangle on a surface mesh, both for GPU and CPU
//	- important to note is that each triangle holds several normals
//		- face normals
//		- a normal at each corner vertex
//		- a normal on each edge
//	- these normals are computed by weighted angle averaging, see "Computing Vertex Normals from Polygonal Facets" - Thürmer // Wüthrich
//  - the member functions for various geometric queries on such triangles implemented in mesh_trianlge (cpu) are reimplemented for the gpu
//    in tool_3d_gpu.cu
//	- these geometrical queries are used in conjunction with grid_cpu_tri.* to perform inside / outisde queries on the tool, i.e. in tool_3d_(gpu|cpu).*

#include <stdio.h>
#include <vector>
#include <unordered_map>

#include <glm/glm.hpp>

#include "types.h"

#ifndef SURFACE_TRI_MESH_H_
#define SURFACE_TRI_MESH_H_

struct mesh_triangle {
private:
	vec3_t closest_on_segment(vec3_t e1, vec3_t e2, vec3_t qp) const;
public:

	unsigned int idx;
	int i1, i2, i3;
	vec3_t p1, p2, p3;
	vec3_t p1_init, p2_init, p3_init;

	vec3_t normal;			//face normal
	vec3_t np1, np2, np3;	//angle weighted vertex normals
	vec3_t ne1, ne2, ne3;	//angle weighted edge normals

	float_t max_h() const;
	vec3_t center() const;

	void bbox(vec3_t &min, vec3_t &max) const;
	bool cover_aabb(vec3_t min, vec3_t max) const;

	void closest_vertex(vec3_t p, vec3_t &cp, unsigned int &idx) const;
	void closest_point(vec3_t qp, vec3_t &cp, vec3_t &np) const;
	bool ray_intersect(vec3_t o, vec3_t dir) const;
	bool ray_intersect(vec3_t o, vec3_t dir, float_t &t) const;
};

struct mesh_triangle_gpu {
private:
	void construct(const mesh_triangle *triangles_cpu, unsigned int num_tri);

public:

	int *idx;
	vec3_t *p1, *p2, *p3;
	vec3_t *p1_init, *p2_init, *p3_init;

	vec3_t *n;	//face normal
	vec3_t *np1, *np2, *np3;		//node normals
	vec3_t *ne1, *ne2, *ne3;		//edge normals

	// construction by both raw arrays of mesh_triangle and std::vector of mesh_triangle
	mesh_triangle_gpu(const mesh_triangle *triangles_cpu, unsigned int num_tri);
	mesh_triangle_gpu(const std::vector<mesh_triangle> &triangles_cpu);
};

//compute angle weighted vertex normals according to
//Computing Vertex Normals from Polygonal Facets - Thürmer // Wüthrich
void mesh_compute_vertex_normals(std::vector<mesh_triangle> &triangles, const std::vector<vec3_t> &points);

//report average edge length
float_t mesh_average_edge_length(const std::vector<mesh_triangle> &triangles);

#endif /* SURFACE_TRI_MESH_H_ */

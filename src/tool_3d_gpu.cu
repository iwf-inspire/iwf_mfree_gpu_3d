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

#include "tool_3d_gpu.h"
#include "tool_3d.h"

__constant__ static geom_constants geometry;

__constant__ float_t  friction_mu;
__constant__ float_t  contact_alpha;
__constant__ float_t  slave_mass;
__constant__ float3_t tool_vel;
__constant__ float_t usui_K;
__constant__ float_t usui_xi;

float3 *tool_forces::get_forces_device() const {
	return m_forces;
}

const std::vector<float3> tool_forces::get_forces_host() const {
	std::vector<float3> h_forces(m_num_tool);
	cudaMemcpy((void*) h_forces.data(), m_forces, sizeof(float3)*m_num_tool, cudaMemcpyDeviceToHost);
	return h_forces;
}

void tool_forces::reset() {
	cudaMemset(m_forces,0,sizeof(float3)*m_num_tool);
}

tool_forces::tool_forces(unsigned int num_tool) : m_num_tool(num_tool) {
	cudaMalloc((void**) &m_forces, sizeof(float3)*num_tool);
	m_fp = fopen("results/forces_new.txt", "w+");
}

tool_forces::~tool_forces() {
	fclose(m_fp);
}

void tool_forces::report(unsigned int step) const {
	auto h_forces = get_forces_host();
	if (m_verbose) {
		printf("Tool Forces:\n");
	}
	for (auto &h_force : h_forces) {
		fprintf(m_fp, "%.9g %.9g %.9g ", h_force.x, h_force.y, h_force.z);
		if (m_verbose) {
			printf("%f %f %f ", h_force.x*1e7, h_force.y*1e7, h_force.z*1e7);
		}
	}
	fprintf(m_fp, "\n");
	if (m_verbose) {
		printf("\n");
	}
	fflush(m_fp);
}


//--------------------------------------------------------

__global__ void do_update_tool(mesh_triangle_gpu triangles, int N, float_t cur_t, vec3_t vel) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;

	vec3_t p1i = triangles.p1_init[pidx];
	vec3_t p2i = triangles.p2_init[pidx];
	vec3_t p3i = triangles.p3_init[pidx];

	vec3_t p1, p2, p3;

	p1.x = p1i.x + vel.x*cur_t;
	p1.y = p1i.y + vel.y*cur_t;
	p1.z = p1i.z + vel.z*cur_t;

	p2.x = p2i.x + vel.x*cur_t;
	p2.y = p2i.y + vel.y*cur_t;
	p2.z = p2i.z + vel.z*cur_t;

	p3.x = p3i.x + vel.x*cur_t;
	p3.y = p3i.y + vel.y*cur_t;
	p3.z = p3i.z + vel.z*cur_t;

	triangles.p1[pidx] = p1;
	triangles.p2[pidx] = p2;
	triangles.p3[pidx] = p3;
}

static __device__ float_t unhash_pos(vec3_t pos, int &ix, int &iy, int &iz) {
	ix = (unsigned int) ((pos.x - geometry.bbmin_x)/geometry.dx);
	iy = (unsigned int) ((pos.y - geometry.bbmin_y)/geometry.dx);
	iz = (unsigned int) ((pos.z - geometry.bbmin_z)/geometry.dx);

	return ix*(geometry.ny)*(geometry.nz) + iy*(geometry.nz) + iz;
}

static __device__ vec3_t closest_on_segment(vec3_t p1, vec3_t p2, vec3_t p) {
	float_t dx = p1.x - p2.x;
	float_t dy = p1.y - p2.y;
	float_t dz = p1.z - p2.z;

	float_t ns2 = dx*dx + dy*dy + dz*dz;

	if (ns2 == 0.) {
		return p1;
	}

	float_t t = ((p2.x - p1.x)*(p.x - p1.x) +
			(p2.y - p1.y)*(p.y - p1.y) +
			(p2.z - p1.z)*(p.z - p1.z))/ns2;

	if (t > 1.) {
		return p2;
	} else if (t < 0.) {
		return p1;
	} else {
		return (1-t)*p1 + t*p2;
	}
}

static __device__ float_t distsquared(vec3_t p1, vec3_t p2) {
	float_t dx = p1.x - p2.x;
	float_t dy = p1.y - p2.y;
	float_t dz = p1.z - p2.z;

	return dx*dx + dy*dy + dz*dz;
}

static __device__ void comp_closest_point(mesh_triangle_gpu triangles, int idx, vec3_t pos, vec3_t &cp, vec3_t &np) {
	//algorithm taken from gts

	// cp - closest point
	// np - normal at closest point
	// qp - query point

	vec3_t qp = pos;					/*!< qp - query point */

	vec3_t p1 = triangles.p1[idx];
	vec3_t p2 = triangles.p2[idx];
	vec3_t p3 = triangles.p3[idx];

	vec3_t p1p2 = p2 - p1;
	vec3_t p1p3 = p3 - p1;
	vec3_t pp1  = p1  - qp;

	float_t B = glm::dot(p1p3, p1p2);
	float_t E = glm::dot(p1p2, p1p2);
	float_t C = glm::dot(p1p3, p1p3);

	//collinear case
	float_t det = B*B - E*C;

	if (det == 0.) {
		printf("DEGENERATE TRIANGLE: %d: (%f %f %f) | (%f %f %f) | (%f %f %f) !\n", idx, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z);
	}

	float_t A = glm::dot(p1p3, pp1);
	float_t D = glm::dot(p1p2, pp1);

	float_t t1 = (D*C - A*B)/det;
	float_t t2 = (A*E - D*B)/det;

	if (t1 < 0.) {
		cp = closest_on_segment(p1, p3, qp);

		if (cp == p1) {
			np = triangles.np1[idx];
		} else if (cp == p3) {
			np = triangles.np3[idx];
		} else {
			np = triangles.ne3[idx];
		}

	} else if (t2 < 0.) {
		cp = closest_on_segment(p1, p2, qp);

		if (cp == p1) {
			np = triangles.np1[idx];
		} else if (cp == p2) {
			np = triangles.np2[idx];
		} else {
			np = triangles.ne1[idx];
		}

	} else if (t1 + t2  > 1.) {
		cp = closest_on_segment(p2, p3, qp);

		if (cp == p2) {
			np = triangles.np2[idx];
		} else if (cp == p3) {
			np = triangles.np3[idx];
		} else {
			np = triangles.ne2[idx];
		}

	} else {
		cp = p1 + t1*p1p2 + t2*p1p3;
		np = triangles.n[idx];
	}
}

static __device__ bool triangle_ray_intersect(mesh_triangle_gpu triangles, int idx, vec3_t o, vec3_t dir) {
	//Möller Trombore intersection algorithm from
	//https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

    const float_t EPSILON = 0.0000001;
	vec3_t vertex0 = triangles.p1[idx];
	vec3_t vertex1 = triangles.p2[idx];
	vec3_t vertex2 = triangles.p3[idx];
    vec3_t edge1, edge2, h, s, q;
    float_t a,f,u,v;
    edge1 = vertex1 - vertex0;
    edge2 = vertex2 - vertex0;

    h = glm::cross(dir, edge2);
    a = glm::dot(edge1,h);
    if (a > -EPSILON && a < EPSILON)
        return false;

    f = float_t(1.)/a;
    s = o - vertex0;
    u = f * (glm::dot(s,h));
    if (u < 0.0 || u > 1.0)
        return false;

    q = glm::cross(s,edge1);
    v = f * glm::dot(dir,q);
    if (v < 0.0 || u + v > 1.0)
        return false;

    // At this stage we can compute t to find out where the intersection point is on the line.
    float_t t = f * glm::dot(edge2, q);
    return (t > EPSILON);
}

static __device__ bool triangle_ray_intersect(mesh_triangle_gpu triangles, int idx, vec3_t o, vec3_t dir, float_t &t) {
	//Möller Trombore intersection algorithm from
	//https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

    const float_t EPSILON = 0.0000001;
	vec3_t vertex0 = triangles.p1[idx];
	vec3_t vertex1 = triangles.p2[idx];
	vec3_t vertex2 = triangles.p3[idx];
    vec3_t edge1, edge2, h, s, q;
    float_t a,f,u,v;
    edge1 = vertex1 - vertex0;
    edge2 = vertex2 - vertex0;

    h = glm::cross(dir, edge2);
    a = glm::dot(edge1,h);
    if (a > -EPSILON && a < EPSILON)
        return false;

    f = float_t(1.)/a;
    s = o - vertex0;
    u = f * (glm::dot(s,h));
    if (u < 0.0 || u > 1.0)
        return false;

    q = glm::cross(s,edge1);
    v = f * glm::dot(dir,q);
    if (v < 0.0 || u + v > 1.0)
        return false;

    // At this stage we can compute t to find out where the intersection point is on the line.
    t = f * glm::dot(edge2, q);
    return (t > EPSILON);
}

__device__ bool establish_contact_by_normal(mesh_triangle_gpu triangles, const int  *__restrict__ cells, vec3_t qp, vec3_t &cp, vec3_t &n, int num_tri, int num_cell) {
	int ix,iy,iz;
	int box_idx = unhash_pos(qp, ix, iy, iz);

	float_t min_dist = DBL_MAX;
	vec3_t closest_point, normal;

	bool first_hit_empty = (cells[box_idx] == cells[box_idx+1]);

	if (cells[box_idx] != cells[box_idx+1]) {
		//test nodes in box

		for (int pidx = cells[box_idx]; pidx < cells[box_idx+1]; pidx++) {
			vec3_t cp_tri, np_tri;

			comp_closest_point(triangles, pidx, qp, cp_tri, np_tri);
			float_t cur_dist = distsquared(cp_tri, qp);

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

	while (true) {

		int low_i  =  ix-offset < 0 ? 0 :  ix-offset;
		int low_j  =  iy-offset < 0 ? 0 :  iy-offset;
		int low_k  =  iz-offset < 0 ? 0 :  iz-offset;

		int high_i = ix+offset+1 > geometry.nx ? geometry.nx : ix+offset+1;
		int high_j = iy+offset+1 > geometry.ny ? geometry.ny : iy+offset+1;
		int high_k = iz+offset+1 > geometry.nz ? geometry.nz : iz+offset+1;

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

					int box_idx = ni*(geometry.ny)*(geometry.nz) + nj*(geometry.nz) + nk;

					if (cells[box_idx] != cells[box_idx+1]) {


						for (int pidx = cells[box_idx]; pidx < cells[box_idx+1]; pidx++) {
							all_empty_so_far = false;

							vec3_t cp_tri, np_tri;
							comp_closest_point(triangles, pidx, qp, cp_tri, np_tri);
							float_t cur_dist = distsquared(cp_tri, qp);

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

	cp = closest_point;
	n  = normal;

	return sign < 0.;
}

__device__ bool establish_contact_by_ray(mesh_triangle_gpu triangles, const int  *__restrict__ cells, vec3_t qp, bool print = false) {

	int ix,iy,iz;
	 unhash_pos(qp, ix, iy, iz);

	vec3_t ray_dir(0.,0.,1.);	//arbitrary direction

	int m_ny = geometry.ny;
	int m_nz = geometry.nz;

	if (print) {
		printf("%d %d %d | %d %d %d\n", ix, iy, iz, geometry.nx, geometry.ny, geometry.nz);
	}

	unsigned int num_inter = 0;
	float_t t;
	for (; iz < m_nz; ++iz) {
		unsigned int box_idx = ix*(m_ny)*(m_nz) + iy*(m_nz) + iz;

		if (print) {
			printf("tris in box: %d %d\n", cells[box_idx], cells[box_idx+1]);
		}

		for (unsigned int pidx = cells[box_idx]; pidx < cells[box_idx+1]; pidx++) {

			bool inter = triangle_ray_intersect(triangles, pidx, qp, ray_dir, t);

			if (print) {
				printf("inter: %d\n", inter ? 1 : 0);
			}

			if (inter) { //triangle may be in more than one box. make sure that intersection is counted only once

				if (inter && print) {
					printf("%f %f %f %f\n", triangles.p1[pidx].x, triangles.p2[pidx].x, triangles.p3[pidx].x, triangles.p1[pidx].x);
					printf("%f %f %f %f\n", triangles.p1[pidx].y, triangles.p2[pidx].y, triangles.p3[pidx].y, triangles.p1[pidx].y);
					printf("%f %f %f %f\n", triangles.p1[pidx].z, triangles.p2[pidx].z, triangles.p3[pidx].z, triangles.p1[pidx].z);
				}

				vec3_t inter_p = qp + t*ray_dir;

				float_t lo_x = geometry.bbmin_x + ix*geometry.dx;
				float_t lo_y = geometry.bbmin_y + iy*geometry.dx;
				float_t lo_z = geometry.bbmin_z + iz*geometry.dx;

				float_t hi_x = geometry.bbmin_x + (ix+1)*geometry.dx;
				float_t hi_y = geometry.bbmin_y + (iy+1)*geometry.dx;
				float_t hi_z = geometry.bbmin_z + (iz+1)*geometry.dx;

				bool in_x = lo_x <= inter_p.x && inter_p.x < hi_x;
				bool in_y = lo_y <= inter_p.y && inter_p.y < hi_y;
				bool in_z = lo_z <= inter_p.z && inter_p.z < hi_z;

				if (in_x && in_y && in_z) {
					num_inter++;
				}
			}
		}
	}

	//if the ray crossed the boundary an uneven number of times the particle resides outside
	return ((num_inter % 2) != 0);
}

__global__ void do_dbg(particle_gpu particles) {
	const unsigned int N = particles.N;

	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;

	particles.T[pidx] = 300.;
}

__global__ void do_compute_contact_forces_n2(int tool_idx, particle_gpu particles, mesh_triangle_gpu triangles,
		float_t dt, int num_tri,
		float3 *forces, bool record_forces, int *num_cntct, bool safe = false) {	// exhaustive contact algorithm (n^2)
	const unsigned int N = particles.N;

	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (particles.blanked[pidx] == 1.) return;
	if (particles.tool_particle[pidx] == 1.) return;

	float4_t pos = particles.pos[pidx];

	vec3_t qp;		// qp -query point
	qp.x = pos.x;
	qp.y = pos.y;
	qp.z = pos.z;

	//-----------------------------------------------

	vec3_t closest_point;
	vec3_t normal;

	//early reject with bb
	bool in_x = qp.x > geometry.bbmin_x && qp.x < geometry.bbmax_x;
	bool in_y = qp.y > geometry.bbmin_y && qp.y < geometry.bbmax_y;
	bool in_z = qp.z > geometry.bbmin_z && qp.z < geometry.bbmax_z;

	if (!(in_x && in_y && in_z)) {
		return;
	}

	//make sure there are no false positives
	//	should only matter for points far from the tool
	if (safe) {
		int num_inter = 0;
		const vec3_t dir(0., 0., 1.);
		for (int i = 0; i < num_tri; i++) {
			bool inter = triangle_ray_intersect(triangles, i, qp, dir);
			if (inter) {
				num_inter++;
			}
		}

		if (num_inter % 2 == 0) return;
	}

	float_t min_dist = FLT_MAX;

	for (int i = 0; i < num_tri; i++) {
		vec3_t cp;
		vec3_t np;
		comp_closest_point(triangles, i, qp, cp, np);
		float_t dist = distsquared(cp, qp);
		if (dist < min_dist) {
			min_dist = dist;
			closest_point = cp;
			normal = np;
		}
	}

	vec3_t diff = closest_point - qp;

	float_t sign = (glm::dot(diff,normal) > 0.) ? 1. : -1.;

	if (!safe && sign > 0.) { //if safe is activated whether or not the point is inside has already been decided
		return;
	}

	float_t dt2 = dt*dt;
	float_t gN  = glm::dot((qp-closest_point), normal);
	vec3_t  fN  = -slave_mass*gN*normal/dt2*contact_alpha;
	vec3_t  fT(0.,0., 0.);

	vec3_t vm = vec3_t(tool_vel.x, tool_vel.y, tool_vel.z);
	if (friction_mu != 0.) {
		vec3_t vs;

		float4_t vel = particles.vel[pidx];
		vs.x = vel.x;
		vs.y = vel.y;
		vs.z = vel.z;

		float3_t fric;
		fric.x =  particles.ft[pidx].x;
		fric.y =  particles.ft[pidx].y;
		fric.z =  particles.ft[pidx].z;

		vec3_t v = vs-vm;
		vec3_t vr = v - v*normal;
		vec3_t fricold(fric.x , fric.y, fric.z);

		vec3_t kdeltae = contact_alpha*slave_mass*vr/dt;
		float_t fy = friction_mu*glm::length(fN);

		vec3_t fstar = fricold - kdeltae;

		if (glm::length(fstar) > fy) {
			fT  = fy*fstar/glm::length(fstar);
		} else {
			fT = fstar;
		}
	}

	particles.fc[pidx].x = fN.x;
	particles.fc[pidx].y = fN.y;
	particles.fc[pidx].z = fN.z;

	particles.ft[pidx].x = fT.x;
	particles.ft[pidx].y = fT.y;
	particles.ft[pidx].z = fT.z;

	particles.n[pidx].x  = normal.x;
	particles.n[pidx].y  = normal.y;
	particles.n[pidx].z  = normal.z;

	if (record_forces) {
		float_t fx = float(fN.x+fT.x);
		float_t fy = float(fN.y+fT.y);
		float_t fz = float(fN.z+fT.z);

		fx = (isnan(fx)) ? 0. : fx;
		fy = (isnan(fy)) ? 0. : fy;
		fz = (isnan(fz)) ? 0. : fz;

		atomicAdd(&(forces[tool_idx].x), fx);
		atomicAdd(&(forces[tool_idx].y), fy);
		atomicAdd(&(forces[tool_idx].z), fz);
	}
}

__global__ void do_compute_contact_forces(int tool_idx, particle_gpu particles, mesh_triangle_gpu triangles,
		const int *__restrict__ cells, float_t dt, int num_tri, int num_cell,
		float3 *forces, bool record_forces, int *num_contact, bool safe = false) {	// Spatial Hashing Contact Algorithm
	const unsigned int N = particles.N;

	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (particles.blanked[pidx] == 1.) return;
	if (particles.tool_particle[pidx] == 1.) return;

	float4_t pos = particles.pos[pidx];

	vec3_t qp;		// qp -query point
	qp.x = pos.x;
	qp.y = pos.y;
	qp.z = pos.z;

	//-----------------------------------------------

	vec3_t closest_point;
	vec3_t normal;

	//early reject with bb
	bool in_x = qp.x > geometry.bbmin_x && qp.x < geometry.bbmax_x;
	bool in_y = qp.y > geometry.bbmin_y && qp.y < geometry.bbmax_y;
	bool in_z = qp.z > geometry.bbmin_z && qp.z < geometry.bbmax_z;

	if (!(in_x && in_y && in_z)) {
		return;
	}

	if (safe) {
		bool inside_safe = establish_contact_by_ray(triangles, cells, qp);

		if (!inside_safe) {
			return;
		}
		establish_contact_by_normal(triangles, cells, qp, closest_point, normal, num_tri, num_cell);
	} else {
		bool inside = establish_contact_by_normal(triangles, cells, qp, closest_point, normal, num_tri, num_cell);
		if (!inside) {
			return;
		}
	}

	float_t dt2 = dt*dt;
	float_t gN  = glm::dot((qp-closest_point), normal);
	vec3_t  fN  = -slave_mass*gN*normal/dt2*contact_alpha;
	vec3_t  fT(0.,0., 0.);

	vec3_t vm = vec3_t(tool_vel.x, tool_vel.y, tool_vel.z);
	if (friction_mu != 0.) {
		vec3_t vs;

		float4_t vel = particles.vel[pidx];
		vs.x = vel.x;
		vs.y = vel.y;
		vs.z = vel.z;

		float3_t fric;
		fric.x =  particles.ft[pidx].x;
		fric.y =  particles.ft[pidx].y;
		fric.z =  particles.ft[pidx].z;

		vec3_t v = vs-vm;
		vec3_t vr = v - v*normal;
		vec3_t fricold(fric.x , fric.y, fric.z);

		vec3_t kdeltae = contact_alpha*slave_mass*vr/dt;
		float_t fy = friction_mu*glm::length(fN);
		vec3_t fstar = fricold - kdeltae;

		if (glm::length(fstar) > fy) {
			fT  = fy*fstar/glm::length(fstar);
		} else {
			fT = fstar;
		}
	}

	particles.fc[pidx].x = fN.x;
	particles.fc[pidx].y = fN.y;
	particles.fc[pidx].z = fN.z;

	particles.ft[pidx].x = fT.x;
	particles.ft[pidx].y = fT.y;
	particles.ft[pidx].z = fT.z;

	particles.n[pidx].x  = normal.x;
	particles.n[pidx].y  = normal.y;
	particles.n[pidx].z  = normal.z;

	if (record_forces) {
		atomicAdd(&forces[tool_idx].x,float(fN.x+fT.x));
		atomicAdd(&forces[tool_idx].y,float(fN.y+fT.y));
		atomicAdd(&forces[tool_idx].z,float(fN.z+fT.z));
	}
}

void tool_3d_gpu::set_vel(vec3_t vel) {
	m_vel = vel;
	cudaMemcpyToSymbol(tool_vel, &m_vel, sizeof(float3_t), 0, cudaMemcpyHostToDevice);
}

void tool_3d_gpu::set_mu(float_t mu) {
	assert(mu > 0.);
	m_mu = mu;
	cudaMemcpyToSymbol(friction_mu, &m_mu, sizeof(float_t), 0, cudaMemcpyHostToDevice);
}
void tool_3d_gpu::set_contact_alpha(float_t alpha) {
	assert(alpha > 0.);
	cudaMemcpyToSymbol(contact_alpha, &alpha, sizeof(float_t), 0);
}

void tool_3d_gpu::set_birth_death(float_t birth, float_t death) {
	m_birth = birth;
	m_death = death;
}

vec3_t tool_3d_gpu::get_vel() {
	return m_vel;
}

const std::vector<mesh_triangle> &tool_3d_gpu::get_cpu_tris() const {
	return m_cpu_mesh;
}

const std::vector<vec3_t> &tool_3d_gpu::get_cpu_pos() const {
	return m_cpu_pos;
}

float_t tool_3d_gpu::mu() const {
	return m_mu;
}

void tool_3d_gpu::set_thermal(bool is_thermal) {
	m_is_thermal = is_thermal;
}

bool tool_3d_gpu::thermal() const {
	return m_is_thermal;
}

void tool_3d_gpu::get_bbox(vec3_t &min, vec3_t &max) const {
	min.x = m_geometry_constants.bbmin_x;
	min.y = m_geometry_constants.bbmin_y;
	min.z = m_geometry_constants.bbmin_z;

	max.x = m_geometry_constants.bbmax_x;
	max.y = m_geometry_constants.bbmax_y;
	max.z = m_geometry_constants.bbmax_z;
}

bool tool_3d_gpu::is_alive() const {
	return global_time_current >= m_birth && global_time_current <= m_death;
}

void tool_3d_gpu::set_algorithm_type(tool_3d_gpu::contact_algorithm algo_type) {
	m_algo_type = algo_type;

	switch (m_algo_type) {
	case tool_3d_gpu::contact_algorithm::spatial_hashing:
		printf("Using spatial hashing for contact search (n*log(n)).\n");
		break;
	case tool_3d_gpu::contact_algorithm::exhaustive:
		printf("Using exhaustive algorithm for contact search (n^2).\n");
		break;
	}

}

void tool_3d_gpu::set_physics(phys_constants phys) {
	if (phys.mass <= 0.) {
		printf("alarm! tool was given a particle mass of zero. no contact forces will be computed!\n");
	}

	cudaMemcpyToSymbol(slave_mass,    &phys.mass, sizeof(float_t), 0);
}

tool_3d_gpu::tool_3d_gpu(const tool_3d* tool, vec3_t vel, phys_constants phys, float_t alpha) {

	{
		static int idx = 0;
		this->idx = idx;
		idx++;
	}

	m_num_cell = tool->m_num_cells;
	m_num_boxed_tris = tool->m_num_boxed_tris;
	m_num_tris = tool->m_triangles.size();

	cudaMalloc((void**) &m_cells, sizeof(int)*(m_num_cell+1));
	cudaMemcpy(m_cells, tool->m_cells, sizeof(int)*(m_num_cell+1), cudaMemcpyHostToDevice);

	m_boxed_triangles = new mesh_triangle_gpu(tool->m_boxed_tris, tool->m_num_boxed_tris);
	m_triangles = new mesh_triangle_gpu(tool->m_triangles);

	if (phys.mass <= 0.) {
		printf("alarm!, tool was given a particle mass of zero. no contact forces will be computed!\n");
	}

	cudaMemcpyToSymbol(contact_alpha, &alpha,     sizeof(float_t), 0);
	cudaMemcpyToSymbol(slave_mass,    &phys.mass, sizeof(float_t), 0);

	m_geometry_constants.nx = tool->m_grid.nx();
	m_geometry_constants.ny = tool->m_grid.ny();
	m_geometry_constants.nz = tool->m_grid.nz();

	vec3_t bbmin, bbmax;
	tool->get_bbox(bbmin, bbmax);

	m_geometry_constants.bbmin_x = bbmin.x;
	m_geometry_constants.bbmin_y = bbmin.y;
	m_geometry_constants.bbmin_z = bbmin.z;

	m_geometry_constants.bbmax_x = bbmax.x;
	m_geometry_constants.bbmax_y = bbmax.y;
	m_geometry_constants.bbmax_z = bbmax.z;

	m_geometry_constants.dx = tool->m_grid.dx();

	cudaMemcpyToSymbol(geometry, &m_geometry_constants, sizeof(geom_constants), 0, cudaMemcpyHostToDevice);

	m_vel = vel;

	m_cpu_mesh = tool->m_triangles;
	m_cpu_pos = tool->m_positions;
	m_cpu_pos_init = tool->m_positions;
}

tool_3d_gpu::tool_3d_gpu(const tool_3d* tool, float_t alpha) {

	{
		static int idx = 0;
		this->idx = idx;
		idx++;
	}

	m_num_cell = tool->m_num_cells;
	m_num_boxed_tris = tool->m_num_boxed_tris;
	m_num_tris = tool->m_triangles.size();

	cudaMalloc((void**) &m_cells, sizeof(int)*(m_num_cell+1));
	cudaMemcpy(m_cells, tool->m_cells, sizeof(int)*(m_num_cell+1), cudaMemcpyHostToDevice);

	m_boxed_triangles = new mesh_triangle_gpu(tool->m_boxed_tris, tool->m_num_boxed_tris);
	m_triangles = new mesh_triangle_gpu(tool->m_triangles);

	cudaMemcpyToSymbol(contact_alpha, &alpha,     sizeof(float_t), 0);

	m_geometry_constants.nx = tool->m_grid.nx();
	m_geometry_constants.ny = tool->m_grid.ny();
	m_geometry_constants.nz = tool->m_grid.nz();

	vec3_t bbmin, bbmax;
	tool->get_bbox(bbmin, bbmax);

	m_geometry_constants.bbmin_x = bbmin.x;
	m_geometry_constants.bbmin_y = bbmin.y;
	m_geometry_constants.bbmin_z = bbmin.z;

	m_geometry_constants.bbmax_x = bbmax.x;
	m_geometry_constants.bbmax_y = bbmax.y;
	m_geometry_constants.bbmax_z = bbmax.z;

	m_geometry_constants.dx = tool->m_grid.dx();

	cudaMemcpyToSymbol(geometry, &m_geometry_constants, sizeof(geom_constants), 0, cudaMemcpyHostToDevice);

	m_cpu_mesh = tool->m_triangles;
	m_cpu_pos = tool->m_positions;
	m_cpu_pos_init = tool->m_positions;
}

tool_3d_gpu::tool_3d_gpu() {
}

void tool_3d_gpu::sample(const tool_3d* tool, std::vector<float4_t> &samples, float_t dx) {
	const std::vector<vec3_t> vec3_samples = tool->sample(dx);

	for (auto it : vec3_samples) {
		float4_t pos;
		pos.x = it.x;
		pos.y = it.y;
		pos.z = it.z;
		samples.push_back(pos);
	}
}

void tool_3d_gpu::compute_contact_force(particle_gpu *particles, bool record_forces) {
	if (m_num_boxed_tris == 0) {
		return;
	}

	if (global_time_current < m_birth || global_time_current > m_death) {
		return;
	}

	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size-1) / block_size);
	dim3 dB(block_size);

	float3 *forces_device = 0;
	if (record_forces) {
		forces_device = global_tool_forces->get_forces_device();
	}

	int *num_cntc_device = 0;

	switch (m_algo_type) {
	case tool_3d_gpu::contact_algorithm::spatial_hashing:
		do_compute_contact_forces<<<dG,dB>>>(this->idx, *particles, *m_boxed_triangles, m_cells, global_time_dt, m_num_boxed_tris, m_num_cell,
				forces_device, record_forces, num_cntc_device, true);
		break;
	case tool_3d_gpu::contact_algorithm::exhaustive:
		do_compute_contact_forces_n2<<<dG,dB>>>(this->idx, *particles, *m_triangles, global_time_dt, m_num_tris,
				forces_device, record_forces, num_cntc_device, false);
		break;
	}

	check_cuda_error("compute contact force!\n");
}

void tool_3d_gpu::update_tool() {
	if (m_num_boxed_tris == 0) {
		return;
	}

	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((m_num_boxed_tris + block_size-1) / block_size);
	dim3 dB(block_size);

	switch (m_algo_type) {
	case tool_3d_gpu::contact_algorithm::spatial_hashing:
		do_update_tool<<<dG,dB>>>(*m_boxed_triangles, m_num_boxed_tris, global_time_current, m_vel);
		break;
	case tool_3d_gpu::contact_algorithm::exhaustive:
		do_update_tool<<<dG,dB>>>(*m_triangles, m_num_tris, global_time_current, m_vel);
		break;
	}

	//update bounding box
	m_geometry_constants.bbmin_x += global_time_dt*m_vel.x;
	m_geometry_constants.bbmin_y += global_time_dt*m_vel.y;
	m_geometry_constants.bbmin_z += global_time_dt*m_vel.z;

	m_geometry_constants.bbmax_x += global_time_dt*m_vel.x;
	m_geometry_constants.bbmax_y += global_time_dt*m_vel.y;
	m_geometry_constants.bbmax_z += global_time_dt*m_vel.z;

	cudaMemcpyToSymbol(geometry, &m_geometry_constants, sizeof(geom_constants), 0, cudaMemcpyHostToDevice);

	check_cuda_error("update tool!\n");

	//update cpu mesh for output
	for (auto &it : m_cpu_mesh) {
		it.p1 = it.p1_init + global_time_current*m_vel;
		it.p2 = it.p2_init + global_time_current*m_vel;
		it.p3 = it.p3_init + global_time_current*m_vel;
	}

	for (int i = 0; i < m_cpu_pos.size(); i++) {
		m_cpu_pos[i] = m_cpu_pos_init[i] + global_time_current*m_vel;
	}
}

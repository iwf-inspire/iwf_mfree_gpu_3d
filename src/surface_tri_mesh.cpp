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

#include "surface_tri_mesh.h"

float_t mesh_triangle::max_h() const {
	float_t l1 = glm::length(p1 - p2);
	float_t l2 = glm::length(p2 - p3);
	float_t l3 = glm::length(p3 - p1);

	return fmax(fmax(l1,l2),l3);
}

vec3_t mesh_triangle::center() const {
	return float_t(1./3.)*(vec3_t(p1) + vec3_t(p2) + vec3_t(p3));
}

void mesh_triangle::bbox(vec3_t &min, vec3_t &max) const {
	min.x = fmin(fmin(p1.x, p2.x),p3.x);
	min.y = fmin(fmin(p1.y, p2.y),p3.y);
	min.z = fmin(fmin(p1.z, p2.z),p3.z);

	max.x = fmax(fmax(p1.x, p2.x),p3.x);
	max.y = fmax(fmax(p1.y, p2.y),p3.y);
	max.z = fmax(fmax(p1.z, p2.z),p3.z);
}

void mesh_triangle::closest_point(vec3_t qp, vec3_t &cp, vec3_t &np) const {
	//algorithm taken from gts

	vec3_t p1p2 = p2 - p1;
	vec3_t p1p3 = p3 - p1;
	vec3_t pp1  = vec3_t(p1)  - qp;

	float_t B = glm::dot(p1p3, p1p2);
	float_t E = glm::dot(p1p2, p1p2);
	float_t C = glm::dot(p1p3, p1p3);

	//collinear case
	float_t det = B*B - E*C;

	if (det == 0.) {
		printf("DEGENERATE TRIANGLE!\n");
		assert(false);
	}

	float_t A = glm::dot(p1p3, pp1);
	float_t D = glm::dot(p1p2, pp1);

	float_t t1 = (D*C - A*B)/det;
	float_t t2 = (A*E - D*B)/det;

	if (t1 < 0.) {
		cp = closest_on_segment(p1, p3, qp);

		if (vec3_t(cp) == vec3_t(p1)) {
			np = np1;
		} else if (vec3_t(cp) == vec3_t(p3)) {
			np = np3;
		} else {
			np = ne3;
		}

	} else if (t2 < 0.) {
		cp = closest_on_segment(p1, p2, qp);

		if (vec3_t(cp) == vec3_t(p1)) {
			np = np1;
		} else if (vec3_t(cp) == vec3_t(p2)) {
			np = np2;
		} else {
			np = ne1;
		}

	} else if (t1 + t2  > 1.) {
		cp = closest_on_segment(p2, p3, qp);

		if (vec3_t(cp) == vec3_t(p2)) {
			np = np2;
		} else if (vec3_t(cp) == vec3_t(p3)) {
			np = np3;
		} else {
			np = ne2;
		}

	} else {
		cp = vec3_t(p1) + t1*p1p2 + t2*p1p3;
		np = normal;
	}
}

vec3_t mesh_triangle::closest_on_segment(vec3_t p1, vec3_t p2, vec3_t p) const {
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

void mesh_triangle::closest_vertex(vec3_t p, vec3_t &cp, unsigned int &idx) const {
	float_t d1 = glm::length(p - vec3_t(p1));
	float_t d2 = glm::length(p - vec3_t(p2));
	float_t d3 = glm::length(p - vec3_t(p3));

	if (d1 < d2 && d1 < d3) {
		cp  = p1;
		idx = i1;
	}

	if (d2 < d1 && d2 < d3) {
		cp  = p2;
		idx = i2;
	}

	if (d3 < d1 && d3 < d1) {
		cp  = p3;
		idx = i3;
	}
}

//defines from
//	http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt

#define FINDMINMAX(x0,x1,x2,min,max) \
		min = max = x0;   \
		if(x1<min) min=x1;\
		if(x1>max) max=x1;\
		if(x2<min) min=x2;\
		if(x2>max) max=x2;

/*======================== X-tests ========================*/
#define AXISTEST_X01(a, b, fa, fb)			   \
		p0 = a*v0.y - b*v0.z;			       	   \
		p2 = a*v2.y - b*v2.z;			       	   \
		if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
		rad = fa * boxhalfsize.y + fb * boxhalfsize.z;   \
		if(min>rad || max<-rad) return false;

#define AXISTEST_X2(a, b, fa, fb)			   \
		p0 = a*v0.y - b*v0.z;			           \
		p1 = a*v1.y - b*v1.z;			       	   \
		if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
		rad = fa * boxhalfsize.y + fb * boxhalfsize.z;   \
		if(min>rad || max<-rad) return false;

/*======================== Y-tests ========================*/
#define AXISTEST_Y02(a, b, fa, fb)			   \
		p0 = -a*v0.x + b*v0.z;		      	   \
		p2 = -a*v2.x + b*v2.z;	       	       	   \
		if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
		rad = fa * boxhalfsize.x + fb * boxhalfsize.z;   \
		if(min>rad || max<-rad) return false;

#define AXISTEST_Y1(a, b, fa, fb)			   \
		p0 = -a*v0.x + b*v0.z;		      	   \
		p1 = -a*v1.x + b*v1.z;	     	       	   \
		if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
		rad = fa * boxhalfsize.x + fb * boxhalfsize.z;   \
		if(min>rad || max<-rad) return false;

/*======================== Z-tests ========================*/
#define AXISTEST_Z12(a, b, fa, fb)			   \
		p1 = a*v1.x - b*v1.y;			           \
		p2 = a*v2.x - b*v2.y;			       	   \
		if(p2<p1) {min=p2; max=p1;} else {min=p1; max=p2;} \
		rad = fa * boxhalfsize.x + fb * boxhalfsize.y;   \
		if(min>rad || max<-rad) return false;

#define AXISTEST_Z0(a, b, fa, fb)			   \
		p0 = a*v0.x - b*v0.y;				   \
		p1 = a*v1.x - b*v1.y;			           \
		if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
		rad = fa * boxhalfsize.x + fb * boxhalfsize.y;   \
		if(min>rad || max<-rad) return false;

bool plane_box_overlap(vec3_t normal, vec3_t vert, vec3_t maxbox){
	vec3_t vmin,vmax;

	for(unsigned int q=0; q < 3; q++) {
		float_t v=vert[q];
		if(normal[q] > 0.0) {
			vmin[q]=-maxbox[q] - v;
			vmax[q]= maxbox[q] - v;
		} else {
			vmin[q]= maxbox[q] - v;
			vmax[q]=-maxbox[q] - v;
		}
	}

	if(glm::dot(normal,vmin) >  0.0) return false;
	if(glm::dot(normal,vmax) >= 0.0) return true;

	return false;
}


bool mesh_triangle::cover_aabb(vec3_t bbmin, vec3_t bbmax) const {
	vec3_t boxcenter   = float_t(0.5)*(bbmin+bbmax);
	vec3_t boxhalfsize = float_t(0.5)*(bbmax-bbmin);

	vec3_t v0 = vec3_t(p1) - boxcenter;
	vec3_t v1 = vec3_t(p2) - boxcenter;
	vec3_t v2 = vec3_t(p3) - boxcenter;

	vec3_t e0 = p2-p1;
	vec3_t e1 = p3-p2;
	vec3_t e2 = p1-p3;

	float_t min,max,p0,p1,p2,rad;

	float_t fex = fabs(e0.x);
	float_t fey = fabs(e0.y);
	float_t fez = fabs(e0.z);

	AXISTEST_X01(e0.z, e0.y, fez, fey);
	AXISTEST_Y02(e0.z, e0.x, fez, fex);
	AXISTEST_Z12(e0.y, e0.x, fey, fex);

	fex = fabsf(e1.x);
	fey = fabsf(e1.y);
	fez = fabsf(e1.z);

	AXISTEST_X01(e1.z, e1.y, fez, fey);
	AXISTEST_Y02(e1.z, e1.x, fez, fex);
	AXISTEST_Z0(e1.y, e1.x, fey, fex);

	fex = fabsf(e2.x);
	fey = fabsf(e2.y);
	fez = fabsf(e2.z);

	AXISTEST_X2(e2.z, e2.y, fez, fey);
	AXISTEST_Y1(e2.z, e2.x, fez, fex);
	AXISTEST_Z12(e2.y, e2.x, fey, fex);

	FINDMINMAX(v0.x,v1.x,v2.x,min,max);
	if(min>boxhalfsize.x || max<-boxhalfsize.x) return false;

	FINDMINMAX(v0.y,v1.y,v2.y,min,max);
	if(min>boxhalfsize.y || max<-boxhalfsize.y) return false;

	FINDMINMAX(v0.z,v1.z,v2.z,min,max);
	if(min>boxhalfsize.z || max<-boxhalfsize.z) return false;

	vec3_t normal = glm::cross(e0,e1);
	if (!plane_box_overlap(normal,v0,boxhalfsize)) return false;

	return true;
}

bool mesh_triangle::ray_intersect(vec3_t o, vec3_t dir) const {

	//Möller Trombore intersection algorithm from
	//https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

    const float EPSILON = 0.0000001;
    vec3_t vertex0 = p1;
    vec3_t vertex1 = p2;
    vec3_t vertex2 = p3;
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

bool mesh_triangle::ray_intersect(vec3_t o, vec3_t dir, float_t &t) const {

	//Möller Trombore intersection algorithm from
	//https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

    const float EPSILON = 0.0000001;
    vec3_t vertex0 = p1;
    vec3_t vertex1 = p2;
    vec3_t vertex2 = p3;
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

static void pick_other_vertex(mesh_triangle tri, vec3_t qp, vec3_t &far1, vec3_t &far2) {
	assert(glm::length(vec3_t(tri.p1) - qp) < 1e-12 || glm::length(vec3_t(tri.p2) - qp) < 1e-12 || glm::length(vec3_t(tri.p3) - qp) < 1e-12);

	if (glm::length(vec3_t(tri.p1) - qp) < 1e-12) {
		far1 = tri.p2;
		far2 = tri.p3;
	}

	if (glm::length(vec3_t(tri.p2) - qp) < 1e-12) {
		far1 = tri.p1;
		far2 = tri.p3;
	}

	if (glm::length(vec3_t(tri.p3) - qp) < 1e-12) {
		far1 = tri.p1;
		far2 = tri.p2;
	}
}

static vec3_t angle_average_vertex_normal(const std::vector<unsigned int> &triangle_idx, const std::vector<mesh_triangle> &triangles, vec3_t pos) {
	vec3_t np(0.);
	for (auto it = triangle_idx.begin(); it != triangle_idx.end(); it++) {
		mesh_triangle cur_tri = triangles[*it];
		vec3_t f1, f2;

		pick_other_vertex(cur_tri, pos, f1, f2);

		vec3_t f1p = f1-pos;
		vec3_t f2p = f2-pos;

		float_t angle = acos(fabs(glm::dot(f1p, f2p)/(glm::length(f1p)*glm::length(f2p))));
		np += angle*cur_tri.normal;
	}
	return np;
}

static void print_triangle(mesh_triangle t1) {
	printf("%f %f %f %f\n", t1.p1.x, t1.p2.x, t1.p3.x, t1.p1.x);
	printf("%f %f %f %f\n", t1.p1.y, t1.p2.y, t1.p3.y, t1.p1.y);
	printf("%f %f %f %f\n", t1.p1.z, t1.p2.z, t1.p3.z, t1.p1.z);
}

static void print_triangle_normals(FILE* fp, mesh_triangle t) {
	fprintf(fp, "%f %f %f %f %f %f\n", t.p1.x, t.p1.y, t.p1.z, t.ne1.x, t.ne1.y, t.ne1.z);
	fprintf(fp, "%f %f %f %f %f %f\n", t.p2.x, t.p2.y, t.p2.z, t.ne2.x, t.ne2.y, t.ne2.z);
	fprintf(fp, "%f %f %f %f %f %f\n", t.p3.x, t.p3.y, t.p3.z, t.ne3.x, t.ne3.y, t.ne3.z);

	vec3_t e1 = float_t(0.5)*(t.p1 + t.p2);
	vec3_t e2 = float_t(0.5)*(t.p2 + t.p3);
	vec3_t e3 = float_t(0.5)*(t.p3 + t.p1);

	fprintf(fp, "%f %f %f %f %f %f\n", e1.x, e1.y, e1.z, t.ne1.x, t.ne1.y, t.ne1.z);
	fprintf(fp, "%f %f %f %f %f %f\n", e2.x, e2.y, e2.z, t.ne2.x, t.ne2.y, t.ne2.z);
	fprintf(fp, "%f %f %f %f %f %f\n", e3.x, e3.y, e3.z, t.ne3.x, t.ne3.y, t.ne3.z);

	vec3_t f = float_t(1./3.)*(t.p1 + t.p2 + t.p3);

	fprintf(fp, "%f %f %f %f %f %f\n", f.x, f.y, f.z, t.normal.x, t.normal.y, t.normal.z);
}

void mesh_compute_vertex_normals(std::vector<mesh_triangle> &triangles, const std::vector<vec3_t> &points) {
	//build dictionaries

	//mapping from points to each attached triangle
	std::vector<unsigned int> *pos_tri_dict = new std::vector<unsigned int>[points.size()];

	struct edge {
		int p1, p2;
		bool operator == (const edge &other) const {
			return ((other.p1 == p1) && (other.p2 == p2)) || ((other.p2 == p1) && (other.p1 == p2));
		}
	};

	struct edge_hasher {
		std::size_t operator()(const edge& e) const {
			return std::hash<unsigned int>()(e.p1) + std::hash<unsigned int>()(e.p2);
		}
	};

	//map each edge to two triangles (indices)
	std::unordered_map<edge, std::vector<unsigned int>, edge_hasher> edge_dict;

	for (unsigned int i = 0; i < triangles.size(); i++) {
		//triangle map
		pos_tri_dict[triangles[i].i1].push_back(i);
		pos_tri_dict[triangles[i].i2].push_back(i);
		pos_tri_dict[triangles[i].i3].push_back(i);

		//egde map
		edge e1, e2, e3;
		e1.p1 = triangles[i].i1;
		e1.p2 = triangles[i].i2;
		e2.p1 = triangles[i].i2;
		e2.p2 = triangles[i].i3;
		e3.p1 = triangles[i].i3;
		e3.p2 = triangles[i].i1;

		auto found_e1 = edge_dict.find(e1);
		auto found_e2 = edge_dict.find(e2);
		auto found_e3 = edge_dict.find(e3);

		if (found_e1 != edge_dict.end()) {
			found_e1->second.push_back(i);
		} else {
			edge_dict.insert({e1, std::vector<unsigned int>({i})});
		}

		if (found_e2 != edge_dict.end()) {
			found_e2->second.push_back(i);
		} else {
			edge_dict.insert({e2, std::vector<unsigned int>({i})});
		}

		if (found_e3 != edge_dict.end()) {
			found_e3->second.push_back(i);
		} else {
			edge_dict.insert({e3, std::vector<unsigned int>({i})});
		}
	}

	FILE *fp = fopen("bugged_tris.txt", "w+");
	for (auto it = edge_dict.begin(); it != edge_dict.end(); ++it) {
		if (it->second.size() != 2) {
			for (int i = 0; i < it->second.size(); i++) {

				mesh_triangle tt = triangles[it->second[i]];

				vec3_t p1 = points[tt.i1];
				vec3_t p2 = points[tt.i2];
				vec3_t p3 = points[tt.i3];

				fprintf(fp, "%f %f %f %f\n", p1.x, p2.x, p3.x, p1.x);
				fprintf(fp, "%f %f %f %f\n", p1.y, p2.y, p3.y, p1.y);
				fprintf(fp, "%f %f %f %f\n", p1.z, p2.z, p3.z, p1.z);
			}
		}
//		assert(it->second.size() == 2);
	}
	fclose(fp);

	//build face normals (straight forward)
	for (auto it = triangles.begin(); it != triangles.end(); it++) {
		unsigned int idx1 = it->i1;
		unsigned int idx2 = it->i2;
		unsigned int idx3 = it->i3;

		assert(idx1 >= 0 && idx1 < points.size());
		assert(idx2 >= 0 && idx2 < points.size());
		assert(idx3 >= 0 && idx3 < points.size());

		it->p1 = points[idx1];
		it->p2 = points[idx2];
		it->p3 = points[idx3];

		it->p1_init = points[idx1];
		it->p2_init = points[idx2];
		it->p3_init = points[idx3];

		vec3_t t1 = it->p2-it->p1;
		vec3_t t2 = it->p3-it->p1;
		it->normal = glm::normalize(glm::cross(t1,t2));
	}

	//compute (angle weighted) edge normals
	for (auto it = triangles.begin(); it != triangles.end(); it++) {
		edge e1, e2, e3;
		e1.p1 = it->i1;
		e1.p2 = it->i2;
		e2.p1 = it->i2;
		e2.p2 = it->i3;
		e3.p1 = it->i3;
		e3.p2 = it->i1;

		if (glm::length(triangles[edge_dict[e1][0]].normal + triangles[edge_dict[e1][1]].normal) < 1e-12) {
			print_triangle(triangles[edge_dict[e1][0]]);
			printf("----------\n");
			print_triangle(triangles[edge_dict[e1][1]]);
			printf("\n");
		}

		if (glm::length(triangles[edge_dict[e2][0]].normal + triangles[edge_dict[e2][1]].normal) < 1e-12) {
			print_triangle(triangles[edge_dict[e2][0]]);
			printf("----------\n");
			print_triangle(triangles[edge_dict[e2][1]]);
			printf("\n");
		}

		if (glm::length(triangles[edge_dict[e3][0]].normal + triangles[edge_dict[e3][1]].normal) < 1e-12) {
			print_triangle(triangles[edge_dict[e3][0]]);
			printf("----------\n");
			print_triangle(triangles[edge_dict[e3][1]]);
			printf("\n");
		}

		//angle weighted edge normal is just average of face normals attached to that edge
		it->ne1 = glm::normalize(triangles[edge_dict[e1][0]].normal + triangles[edge_dict[e1][1]].normal);
		it->ne2 = glm::normalize(triangles[edge_dict[e2][0]].normal + triangles[edge_dict[e2][1]].normal);
		it->ne3 = glm::normalize(triangles[edge_dict[e3][0]].normal + triangles[edge_dict[e3][1]].normal);
	}

	//compute (angle weighted) vertex normals
	for (auto it = triangles.begin(); it != triangles.end(); it++) {
		unsigned int idx1 = it->i1;
		unsigned int idx2 = it->i2;
		unsigned int idx3 = it->i3;

		it->np1 = glm::normalize(angle_average_vertex_normal(pos_tri_dict[idx1], triangles, points[idx1]));
		it->np2 = glm::normalize(angle_average_vertex_normal(pos_tri_dict[idx2], triangles, points[idx2]));
		it->np3 = glm::normalize(angle_average_vertex_normal(pos_tri_dict[idx3], triangles, points[idx3]));
	}

	{
		FILE *fp = fopen("tool_normals.txt", "w+");
		for (auto it = triangles.begin(); it != triangles.end(); it++) {
			print_triangle_normals(fp, *it);
		}
		fclose(fp);
	}
}

float_t mesh_average_edge_length(const std::vector<mesh_triangle> &triangles) {
	float_t lmin = DBL_MAX;
	float_t lmax = 0;

	for (auto it = triangles.begin(); it != triangles.end(); ++it) {
		float_t l1 = glm::length(it->p1-it->p2);
		float_t l2 = glm::length(it->p2-it->p3);
		float_t l3 = glm::length(it->p3-it->p1);

		lmin = fmin(fmin(fmin(lmin, l1),l2),l3);
		lmax = fmax(fmax(fmax(lmax, l1),l2),l3);
	}

	return 0.5*(lmax+lmin);
}


//-----------------------------------------------------------------------------------------------------

void mesh_triangle_gpu::construct(const mesh_triangle *triangles_cpu, unsigned int num_tri) {
	int *h_idx = new int[num_tri];

	vec3_t *h_p1 = new vec3_t[num_tri];
	vec3_t *h_p2 = new vec3_t[num_tri];
	vec3_t *h_p3 = new vec3_t[num_tri];

	vec3_t *h_n = new vec3_t[num_tri];

	vec3_t *h_np1 = new vec3_t[num_tri];
	vec3_t *h_np2 = new vec3_t[num_tri];
	vec3_t *h_np3 = new vec3_t[num_tri];

	vec3_t *h_ne1 = new vec3_t[num_tri];
	vec3_t *h_ne2 = new vec3_t[num_tri];
	vec3_t *h_ne3 = new vec3_t[num_tri];

	for (unsigned int i = 0; i < num_tri; i++) {
		h_idx[i] = triangles_cpu[i].idx;

		h_p1[i] = triangles_cpu[i].p1;
		h_p2[i] = triangles_cpu[i].p2;
		h_p3[i] = triangles_cpu[i].p3;

		h_n[i] = triangles_cpu[i].normal;

		h_np1[i] = triangles_cpu[i].np1;
		h_np2[i] = triangles_cpu[i].np2;
		h_np3[i] = triangles_cpu[i].np3;

		h_ne1[i] = triangles_cpu[i].ne1;
		h_ne2[i] = triangles_cpu[i].ne2;
		h_ne3[i] = triangles_cpu[i].ne3;
	}

	cudaMalloc((void**) &idx, sizeof(int)*num_tri);

	cudaMalloc((void**) &p1, sizeof(vec3_t)*num_tri);
	cudaMalloc((void**) &p2, sizeof(vec3_t)*num_tri);
	cudaMalloc((void**) &p3, sizeof(vec3_t)*num_tri);

	cudaMalloc((void**) &p1_init, sizeof(vec3_t)*num_tri);
	cudaMalloc((void**) &p2_init, sizeof(vec3_t)*num_tri);
	cudaMalloc((void**) &p3_init, sizeof(vec3_t)*num_tri);

	cudaMalloc((void**) &n, sizeof(vec3_t)*num_tri);

	cudaMalloc((void**) &np1, sizeof(vec3_t)*num_tri);
	cudaMalloc((void**) &np2, sizeof(vec3_t)*num_tri);
	cudaMalloc((void**) &np3, sizeof(vec3_t)*num_tri);

	cudaMalloc((void**) &ne1, sizeof(vec3_t)*num_tri);
	cudaMalloc((void**) &ne2, sizeof(vec3_t)*num_tri);
	cudaMalloc((void**) &ne3, sizeof(vec3_t)*num_tri);

	check_cuda_error("surface_tri_mesh: malloc'd");

	cudaMemcpy(idx, h_idx, sizeof(int)*num_tri, cudaMemcpyHostToDevice);

	cudaMemcpy(p1, h_p1, sizeof(vec3_t)*num_tri, cudaMemcpyHostToDevice);
	cudaMemcpy(p2, h_p2, sizeof(vec3_t)*num_tri, cudaMemcpyHostToDevice);
	cudaMemcpy(p3, h_p3, sizeof(vec3_t)*num_tri, cudaMemcpyHostToDevice);

	cudaMemcpy(p1_init, h_p1, sizeof(vec3_t)*num_tri, cudaMemcpyHostToDevice);
	cudaMemcpy(p2_init, h_p2, sizeof(vec3_t)*num_tri, cudaMemcpyHostToDevice);
	cudaMemcpy(p3_init, h_p3, sizeof(vec3_t)*num_tri, cudaMemcpyHostToDevice);

	cudaMemcpy(n, h_n, sizeof(vec3_t)*num_tri, cudaMemcpyHostToDevice);

	cudaMemcpy(np1, h_np1, sizeof(vec3_t)*num_tri, cudaMemcpyHostToDevice);
	cudaMemcpy(np2, h_np2, sizeof(vec3_t)*num_tri, cudaMemcpyHostToDevice);
	cudaMemcpy(np3, h_np3, sizeof(vec3_t)*num_tri, cudaMemcpyHostToDevice);

	cudaMemcpy(ne1, h_ne1, sizeof(vec3_t)*num_tri, cudaMemcpyHostToDevice);
	cudaMemcpy(ne2, h_ne2, sizeof(vec3_t)*num_tri, cudaMemcpyHostToDevice);
	cudaMemcpy(ne3, h_ne3, sizeof(vec3_t)*num_tri, cudaMemcpyHostToDevice);

	check_cuda_error("surface_tri_mesh: copie'd");

	delete[] h_idx;

	delete[] h_p1;
	delete[] h_p2;
	delete[] h_p3;

	delete[] h_n;

	delete[] h_np1;
	delete[] h_np2;
	delete[] h_np3;

	delete[] h_ne1;
	delete[] h_ne2;
	delete[] h_ne3;
}

mesh_triangle_gpu::mesh_triangle_gpu(const mesh_triangle *triangles_cpu, unsigned int num_tri) {
	construct(triangles_cpu, num_tri);
}

mesh_triangle_gpu::mesh_triangle_gpu(const std::vector<mesh_triangle> &triangles_cpu) {
	int num_tri = triangles_cpu.size();

	mesh_triangle *triangles_ray = new mesh_triangle[num_tri];

	for (unsigned int i = 0; i < num_tri; i++) {
		triangles_ray[i] = triangles_cpu[i];
	}

	construct(triangles_ray, num_tri);

	delete[] triangles_ray;
}

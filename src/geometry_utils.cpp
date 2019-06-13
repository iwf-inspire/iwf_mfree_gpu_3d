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

#include "geometry_utils.h"

void geometry_rotate(std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions, rotation_axis axis, float_t angle) {
	mat3x3_t R(0.);
	switch (axis) {
	case rotation_axis::x:
		R[0][0] =  1.;
		R[1][1] =   cos(angle);
		R[1][2] =  -sin(angle);
		R[2][1] =   sin(angle);
		R[2][2] =   cos(angle);
		break;
	case rotation_axis::y:
		R[0][0] =   cos(angle);
		R[0][2] =   sin(angle);
		R[1][1] =  1.;
		R[2][0] =   -sin(angle);
		R[2][2] =    cos(angle);
		break;
	case rotation_axis::z:
		R[0][0] =  cos(angle);
		R[0][1] = -sin(angle);
		R[1][0] =  sin(angle);
		R[1][1] =  cos(angle);
		R[2][2] =  1;
		break;
	}

	for (auto it = positions.begin(); it != positions.end(); ++it) {
		*it = R*(*it);
	}

	for (auto it = triangles.begin(); it != triangles.end(); ++it) {
		it->p1 = R*it->p1;
		it->p2 = R*it->p2;
		it->p3 = R*it->p3;
	}
}

void geometry_scale_to_unity(std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions, double rot_angle_z) {
	mat3x3_t R(0.);
	R[0][0] =  cos(rot_angle_z);
	R[0][1] = -sin(rot_angle_z);
	R[1][0] =  sin(rot_angle_z);
	R[1][1] =  cos(rot_angle_z);
	R[2][2] =  1;

	//rotate
	for (auto it = positions.begin(); it != positions.end(); ++it) {
		*it = R*(*it);
	}

	for (auto it = triangles.begin(); it != triangles.end(); ++it) {
		it->p1 = R*it->p1;
		it->p2 = R*it->p2;
		it->p3 = R*it->p3;
	}

	//measure bbox
	double bbmin_x =  DBL_MAX;
	double bbmin_y =  DBL_MAX;
	double bbmin_z =  DBL_MAX;

	double bbmax_x = -DBL_MAX;
	double bbmax_y = -DBL_MAX;
	double bbmax_z = -DBL_MAX;

	for (auto it = positions.begin(); it != positions.end(); ++it) {
		bbmin_x = fmin(it->x, bbmin_x);
		bbmin_y = fmin(it->y, bbmin_y);
		bbmin_z = fmin(it->z, bbmin_z);

		bbmax_x = fmax(it->x, bbmax_x);
		bbmax_y = fmax(it->y, bbmax_y);
		bbmax_z = fmax(it->z, bbmax_z);
	}

	vec3_t bbmin(bbmin_x, bbmin_y, bbmin_z);
	vec3_t bbmax(bbmax_x, bbmax_y, bbmax_z);

	double lx = bbmax_x - bbmin_x;
	double ly = bbmax_y - bbmin_y;
	double lz = bbmax_z - bbmin_z;

	//move to origin
	geometry_translate(triangles, positions, -bbmin);

	// scale
	geometry_scale(triangles, positions, vec3_t(1./lx, 1./ly, 1./lz));
}

void geometry_print_bb(const std::vector<vec3_t> &positions) {
	//measure bbox
	double bbmin_x =  DBL_MAX;
	double bbmin_y =  DBL_MAX;
	double bbmin_z =  DBL_MAX;

	double bbmax_x = -DBL_MAX;
	double bbmax_y = -DBL_MAX;
	double bbmax_z = -DBL_MAX;

	for (auto it = positions.begin(); it != positions.end(); ++it) {
		bbmin_x = fmin(it->x, bbmin_x);
		bbmin_y = fmin(it->y, bbmin_y);
		bbmin_z = fmin(it->z, bbmin_z);

		bbmax_x = fmax(it->x, bbmax_x);
		bbmax_y = fmax(it->y, bbmax_y);
		bbmax_z = fmax(it->z, bbmax_z);
	}

	printf("measured bbox:\n");
	printf("min: %f %f %f\n", bbmin_x, bbmin_y, bbmin_z);
	printf("max: %f %f %f\n", bbmax_x, bbmax_y, bbmax_z);
	printf("--------------\n");
}

void geometry_scale(std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions, vec3_t scale) {
	double lx = scale.x;
	double ly = scale.y;
	double lz = scale.z;

	for (auto it = positions.begin(); it != positions.end(); ++it) {
		it->x *= lx;
		it->y *= ly;
		it->z *= lz;
	}

	for (auto it = triangles.begin(); it != triangles.end(); ++it) {
		it->p1.x *= lx;
		it->p1.y *= ly;
		it->p1.z *= lz;

		it->p2.x *= lx;
		it->p2.y *= ly;
		it->p2.z *= lz;

		it->p3.x *= lx;
		it->p3.y *= ly;
		it->p3.z *= lz;
	}
}

void geometry_translate(std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions, vec3_t translate) {
	for (auto it = positions.begin(); it != positions.end(); ++it) {
		*it += translate;
	}

	for (auto it = triangles.begin(); it != triangles.end(); ++it) {
		it->p1 += translate;
		it->p2 += translate;
		it->p3 += translate;
	}
}

void geometry_get_bb(const std::vector<vec3_t> &positions, vec3_t &bbmin, vec3_t &bbmax) {

	double bbmin_x =  DBL_MAX;
	double bbmin_y =  DBL_MAX;
	double bbmin_z =  DBL_MAX;

	double bbmax_x = -DBL_MAX;
	double bbmax_y = -DBL_MAX;
	double bbmax_z = -DBL_MAX;

	for (auto &it : positions) {
		bbmin_x = fmin(it.x, bbmin_x);
		bbmin_y = fmin(it.y, bbmin_y);
		bbmin_z = fmin(it.z, bbmin_z);

		bbmax_x = fmax(it.x, bbmax_x);
		bbmax_y = fmax(it.y, bbmax_y);
		bbmax_z = fmax(it.z, bbmax_z);
	}

	bbmin.x = bbmin_x;
	bbmin.y = bbmin_y;
	bbmin.z = bbmin_z;

	bbmax.x = bbmax_x;
	bbmax.y = bbmax_y;
	bbmax.z = bbmax_z;
}

void geometry_get_bb(const float4_t * positions, unsigned int num_part, vec3_t &bbmin, vec3_t &bbmax, float_t safety) {

	double bbmin_x =  DBL_MAX;
	double bbmin_y =  DBL_MAX;
	double bbmin_z =  DBL_MAX;

	double bbmax_x = -DBL_MAX;
	double bbmax_y = -DBL_MAX;
	double bbmax_z = -DBL_MAX;

	for (unsigned int i = 0; i < num_part; i++) {
		bbmin_x = fmin(positions[i].x, bbmin_x);
		bbmin_y = fmin(positions[i].y, bbmin_y);
		bbmin_z = fmin(positions[i].z, bbmin_z);

		bbmax_x = fmax(positions[i].x, bbmax_x);
		bbmax_y = fmax(positions[i].y, bbmax_y);
		bbmax_z = fmax(positions[i].z, bbmax_z);
	}

	bbmin.x = bbmin_x - safety;
	bbmin.y = bbmin_y - safety;
	bbmin.z = bbmin_z - safety;

	bbmax.x = bbmax_x + safety;
	bbmax.y = bbmax_y + safety;
	bbmax.z = bbmax_z + safety;
}

//------------------------------------------

void geometry_rotate(std::vector<vec3_t> &positions, rotation_axis axis, float_t angle) {
	mat3x3_t R(0.);
	switch (axis) {
	case rotation_axis::x:
		R[0][0] =  1.;
		R[1][1] =   cos(angle);
		R[1][2] =  -sin(angle);
		R[2][1] =   sin(angle);
		R[2][2] =   cos(angle);
		break;
	case rotation_axis::y:
		R[0][0] =   cos(angle);
		R[0][2] =   sin(angle);
		R[1][1] =  1.;
		R[2][0] =   -sin(angle);
		R[2][2] =    cos(angle);
		break;
	case rotation_axis::z:
		R[0][0] =  cos(angle);
		R[0][1] = -sin(angle);
		R[1][0] =  sin(angle);
		R[1][1] =  cos(angle);
		R[2][2] =  1;
		break;
	}

	for (auto it = positions.begin(); it != positions.end(); ++it) {
		*it = R*(*it);
	}
}

void geometry_scale(std::vector<vec3_t> &positions, vec3_t scale) {
	for (auto it = positions.begin(); it != positions.end(); ++it) {
		it->x = scale.x*it->x;
		it->y = scale.y*it->y;
		it->z = scale.z*it->z;
	}
}

void geometry_translate(std::vector<vec3_t> &positions, vec3_t trafo) {
	for (auto it = positions.begin(); it != positions.end(); ++it) {
		*it = *it + trafo;
	}
}

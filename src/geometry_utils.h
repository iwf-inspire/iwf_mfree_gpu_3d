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

#ifndef GEOMETRY_UTILS_H_
#define GEOMETRY_UTILS_H_

#include "surface_tri_mesh.h"

#include <vector>
#include <glm/glm.hpp>

enum rotation_axis{x, y, z};

void geometry_rotate(std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions, rotation_axis axis, float_t angle);
void geometry_scale_to_unity(std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions, double rot_angle_z);
void geometry_print_bb(const std::vector<vec3_t> &positions);
void geometry_scale(std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions, vec3_t scale);
void geometry_translate(std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions, vec3_t translate);
void geometry_get_bb(const std::vector<vec3_t> &positions, vec3_t &bbmin, vec3_t &bbmax);
void geometry_get_bb(const float4_t * positions, unsigned int num_part, vec3_t &bbmin, vec3_t &bbmax, float_t safety = 0.);

void geometry_rotate(std::vector<vec3_t> &positions, rotation_axis axis, float_t angle);
void geometry_scale(std::vector<vec3_t> &positions, vec3_t scale);
void geometry_translate(std::vector<vec3_t> &positions, vec3_t trafo);

#endif /* GEOMETRY_UTILS_H_ */

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

// extremely rough functions to read in quad or tet FEM meshes produced by LSPREPOST
//  can also dump a surface mesh to be viewed in LSPREPOST
//	no error handling, no protection against malformed files, very basic parsing and tokenizing
//	USE WITH CAUTION!

#ifndef LDYNAK_READER_H_
#define LDYNAK_READER_H_

#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <set>

#include <glm/glm.hpp>

#include "surface_tri_mesh.h"
#include "types.h"

void ldynak_read_triangles_from_quadmesh(std::string file_name, std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions);
void ldynak_read_triangles_from_tetmesh(std::string file_name, std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions);

void ldynak_dump(std::string file_name, std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions);

#endif /* LDYNAK_READER_H_ */

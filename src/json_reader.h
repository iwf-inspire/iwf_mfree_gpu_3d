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

#ifndef JSON_READER_H_
#define JSON_READER_H_

#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>

#include "surface_tri_mesh.h"

//extremely rough and simple method to read triangles and position from a json file produced by meshlab.
//		no error handling, will fall apart if slightly poked with a stick
void json_read_triangles(std::string file_name, std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions);

#endif /* JSON_READER_H_ */

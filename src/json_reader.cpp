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

#include "json_reader.h"

template<typename Out>
static void split(const std::string &s, char delim, Out &result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        result.push_back(item);
    }
}

static std::vector<vec3_t> read_vertices(std::ifstream &infile) {

	std::vector<vec3_t> points;

	std::string line;
	std::string vert_id("vertices");
	bool vert_sec = false;

	while (std::getline(infile, line)) {
		if (line.find(vert_id) != std::string::npos) {
			vert_sec = true;
		}

		if (line.find(']') != std::string::npos) {
			break;
		}

		if (vert_sec) {
			std::vector<std::string> tokens;
			split(line, ',', tokens);

			if (tokens.size() >= 3) {
				assert(tokens.size() % 3 == 0);

				for (unsigned int i = 0; i < tokens.size(); i+=3) {
					vec3_t p;
					p.x = atof(tokens[i+0].c_str());
					p.y = atof(tokens[i+1].c_str());
					p.z = atof(tokens[i+2].c_str());
					points.push_back(p);
				}
			}
		}
	}

	return points;
}

static std::vector<mesh_triangle> read_triangles(std::ifstream &infile) {

	std::vector<mesh_triangle> triangles;

	std::string line;
	std::string tri_id("connectivity");
	bool tri_sec = false;

	unsigned int iter = 0;
	while (std::getline(infile, line)) {
		if (line.find(tri_id) != std::string::npos) {
			tri_sec = true;
		}

		if (tri_sec && line.find(']') != std::string::npos) {
			break;
		}

		if (tri_sec) {
			std::vector<std::string> tokens;
			split(line, ',', tokens);

			if (tokens.size() >= 3) {
				assert(tokens.size() % 3 == 0);

				for (unsigned int i = 0; i < tokens.size(); i+=3) {
					mesh_triangle t;
					t.idx = iter;
					t.i1 = atoi(tokens[i+0].c_str());
					t.i2 = atoi(tokens[i+1].c_str());
					t.i3 = atoi(tokens[i+2].c_str());
					triangles.push_back(t);
					iter++;
				}
			}
		}
	}

	return triangles;
}

void json_read_triangles(std::string file_name, std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions) {
	std::ifstream infile(file_name, std::ifstream::in);

	positions = read_vertices(infile);
	triangles = read_triangles(infile);

	if (positions.size() == 0 || triangles.size() == 0) {
		printf("WARNING: no triangles read!\n");
	}
}

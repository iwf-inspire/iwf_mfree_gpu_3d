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

#include "vtk_reader.h"

template<typename Out>
static void split(const std::string &s, char delim, Out &result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        result.push_back(item);
    }
}

std::vector<float4_t> vtk_read_init_pos(std::string file_name) {
	std::vector<float4_t> points;

	std::string line;
	std::string point_id("POINTS");
	std::string stop_id("CELLS");

	bool point_sec = false;
	std::ifstream infile(file_name, std::ifstream::in);

	while (std::getline(infile, line)) {
		if (line.find(point_id) != std::string::npos) {
			point_sec = true;
			continue;
		}

		if (line.compare(" ") == 0) {
			continue;
		}

		if (point_sec && line.find(stop_id) != std::string::npos) {
			break;
		}

		if (point_sec) {
			std::vector<std::string> tokens;
			split(line, ' ', tokens);
			if (tokens.size() != 3) {
				continue;	//TODO
			}

			double x = std::stod (tokens[0], NULL);
			double y = std::stod (tokens[1], NULL);
			double z = std::stod (tokens[2], NULL);

			points.push_back(make_float4_t(x,y,z,float_t(0.)));
		}
	}

	return points;
}

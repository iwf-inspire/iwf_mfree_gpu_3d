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

#include "ldynak_reader.h"

#include <iterator>
template<typename Out>
static void split(const std::string &s, char delim, Out &result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        result.push_back(item);
    }
}

std::vector<std::string> split(std::string const &input) {
    std::istringstream buffer(input);
    std::vector<std::string> ret;

    std::copy(std::istream_iterator<std::string>(buffer),
              std::istream_iterator<std::string>(),
              std::back_inserter(ret));
    return ret;
}

struct quad_segment {
	unsigned int i1, i2, i3, i4;

	bool operator == (const quad_segment &other) const {
		std::set<unsigned int> this_set({i1,i2,i3,i4});
		std::set<unsigned int> other_set({other.i1,other.i2,other.i3,other.i4});
		std::set<unsigned int> intersection;
		std::set_intersection(this_set.begin(),this_set.end(),other_set.begin(),other_set.end(),
		                  std::inserter(intersection,intersection.begin()));
		return intersection.size() == 4;
	}
};

struct tet_segment {
	unsigned int i1, i2, i3;

	bool operator == (const tet_segment &other) const {
		std::set<unsigned int> this_set({i1,i2,i3});
		std::set<unsigned int> other_set({other.i1,other.i2,other.i3});
		std::set<unsigned int> intersection;
		std::set_intersection(this_set.begin(),this_set.end(),other_set.begin(),other_set.end(),
		                  std::inserter(intersection,intersection.begin()));
		return intersection.size() == 3;
	}
};

struct quad {
	quad_segment q1,q2,q3,q4,q5,q6;
	quad() {};
	quad(std::vector<quad_segment> segs) {
		q1 = segs[0];
		q2 = segs[1];
		q3 = segs[2];
		q4 = segs[3];
		q5 = segs[4];
		q6 = segs[5];
	}

	std::vector<quad_segment> get_segments() const {
		std::vector<quad_segment> segments({q1, q2, q3, q4, q5, q6});
		return segments;
	}
};

struct tet {
	tet_segment t1,t2,t3,t4;
	tet() {};
	tet(std::vector<tet_segment> segs) {
		t1 = segs[0];
		t2 = segs[1];
		t3 = segs[2];
		t4 = segs[3];
	}

	std::vector<tet_segment> get_segments() const {
		std::vector<tet_segment> segments({t1, t2, t3, t4});
		return segments;
	}
};

static std::vector<quad_segment> ids_to_quad_segment(std::vector<unsigned int> idx) {
	assert(idx.size() == 8);

	quad_segment s1 = {idx[0], idx[1], idx[2], idx[3]};		//top and bottom
	quad_segment s2 = {idx[7], idx[6], idx[5], idx[4]};

	quad_segment s3 = {idx[4], idx[5], idx[1], idx[0]};
	quad_segment s4 = {idx[3], idx[2], idx[6], idx[7]};

	quad_segment s5 = {idx[0], idx[3], idx[7], idx[4]};
	quad_segment s6 = {idx[5], idx[6], idx[2], idx[1]};

	std::vector<quad_segment> segments({s1,s2,s3,s4,s5,s6});
	return segments;
}

static std::vector<tet_segment> ids_to_tet_segment(std::vector<unsigned int> idx) {
	assert(idx.size() == 4);

	tet_segment s1 = {idx[0], idx[1], idx[3]};
	tet_segment s2 = {idx[2], idx[1], idx[0]};

	tet_segment s3 = {idx[1], idx[2], idx[3]};
	tet_segment s4 = {idx[3], idx[2], idx[0]};

	std::vector<tet_segment> segments({s1,s2,s3,s4});
	return segments;
}

static std::vector<quad> read_quads(std::ifstream &infile) {
	std::vector<quad> quads;

	std::string line;
	std::string quad_id("*ELEMENT");
	std::string stop_id("*");
	std::string comment_id("$");
	bool quad_sec = false;

	while (std::getline(infile, line)) {
		if (line.find(comment_id) != std::string::npos) {
			continue;
		}

		if (line.find(quad_id) != std::string::npos) {
			quad_sec = true;
			continue;
		}

		if (quad_sec && line.find(stop_id) != std::string::npos) {
			break;
		}

		if (quad_sec) {
			std::vector<std::string> tokens = split(line);
			assert(tokens.size() == 10);	//eid + pid + 8 node ids

			std::vector<unsigned int> ids;
			for (unsigned int i = 2; i < 10; i++) {
				  unsigned int ul = strtoul (tokens[i].c_str(), NULL, 0);
				  ids.push_back(ul);
			}

			std::vector<unsigned int> last_5_ids = std::vector<unsigned int>(ids.begin()+3, ids.end());
			bool all_equal = std::equal(last_5_ids.begin() + 1, last_5_ids.end(), last_5_ids.begin());

			if (all_equal) {
				printf("WARNING: trying to read tet mesh using quad mesh reader. This is gonna end badly!\n");
			}

			std::vector<quad_segment> quad_segs = ids_to_quad_segment(ids);
			quads.push_back(quad(quad_segs));
		}
	}

	return quads;
}

static std::vector<tet> read_tets(std::ifstream &infile) {
	std::vector<tet> tets;

	std::string line;
	std::string quad_id("*ELEMENT");
	std::string stop_id("*");
	std::string comment_id("$");
	bool tet_sec = false;

	while (std::getline(infile, line)) {
		if (line.find(comment_id) != std::string::npos) {
			continue;
		}

		if (line.find(quad_id) != std::string::npos) {
			tet_sec = true;
			continue;
		}

		if (tet_sec && line.find(stop_id) != std::string::npos) {
			break;
		}

		if (tet_sec) {
			std::vector<std::string> tokens = split(line);
			assert(tokens.size() == 10);	//eid + pid + 8 node ids

			std::vector<unsigned int> ids;
			for (unsigned int i = 2; i < 6; i++) {
				  unsigned int ul = strtoul (tokens[i].c_str(), NULL, 0);
				  ids.push_back(ul);
			}

			std::vector<unsigned int> last_5_ids = std::vector<unsigned int>(ids.begin()+3, ids.end());
			bool all_equal = std::equal(last_5_ids.begin() + 1, last_5_ids.end(), last_5_ids.begin());

			if (!all_equal) {
				printf("WARNING: trying to read quad mesh using tet mesh reader. This is gonna end badly!\n");
			}

			std::vector<tet_segment> tet_segs = ids_to_tet_segment(ids);
			tets.push_back(tet(tet_segs));
		}
	}

	return tets;
}

static std::unordered_map<int, vec3_t> read_vertices(std::ifstream &infile) {
	std::unordered_map<int, vec3_t> vertices;

	std::string line;
	std::string node_id("*NODE");
	std::string stop_id("*");
	std::string comment_id("$");
	bool vert_sec = false;

	while (std::getline(infile, line)) {
		if (line.find(comment_id) != std::string::npos) {
			continue;
		}

		if (line.find(node_id) != std::string::npos) {
			vert_sec = true;
			continue;
		}

		if (vert_sec && line.find(stop_id) != std::string::npos) {
			break;
		}

		if (vert_sec) {
			std::vector<std::string> tokens = split(line);
			assert(tokens.size() == 6 || tokens.size() == 4);	//nid + pos + (tc + rc), tc and rc optional

			unsigned int idx = strtoul (tokens[0].c_str(), NULL, 0);
			double x = std::stod (tokens[1], NULL);
			double y = std::stod (tokens[2], NULL);
			double z = std::stod (tokens[3], NULL);

			vertices.insert({idx, vec3_t(x,y,z)});
		}
	}

	return vertices;
}

static std::vector<mesh_triangle> quads_to_tris(const std::vector<quad_segment> &quads) {
	std::vector<mesh_triangle> tris;

	unsigned int iter = 0;
	for (auto it = quads.begin(); it != quads.end(); ++it) {
		mesh_triangle t1;
		t1.idx = iter;
		t1.i1 = it->i1;
		t1.i2 = it->i2;
		t1.i3 = it->i3;
		iter++;

		mesh_triangle t2;
		t2.idx = iter;
		t2.i1 = it->i3;
		t2.i2 = it->i4;
		t2.i3 = it->i1;
		iter++;

		tris.push_back(t1);
		tris.push_back(t2);
	}

	return tris;
}

static std::vector<mesh_triangle> tets_to_tris(const std::vector<tet_segment> &tets) {
	std::vector<mesh_triangle> tris;

	unsigned int iter = 0;
	for (auto it = tets.begin(); it != tets.end(); ++it) {
		mesh_triangle t;
		t.idx = iter;
		t.i1 = it->i1;
		t.i2 = it->i2;
		t.i3 = it->i3;
		iter++;

		tris.push_back(t);
	}

	return tris;
}

void ldynak_read_triangles_from_quadmesh(std::string file_name, std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions) {
	std::ifstream infile(file_name, std::ifstream::in);
	std::vector<quad> quads = read_quads(infile);
	printf("quads read! %lu\n", quads.size());

	infile.clear();
	infile.seekg(0);

	std::unordered_map<int, vec3_t> verts = read_vertices(infile);
	printf("verts read! %lu\n", quads.size());

	struct quad_seg_hasher {
		std::size_t operator()(const quad_segment& e) const {
			return std::hash<unsigned int>()(e.i1) + std::hash<unsigned int>()(e.i2) + std::hash<unsigned int>()(e.i3) + std::hash<unsigned int>()(e.i4);
		}
	};


	//identify segments wich belong to only one quad (=boundary)
	std::unordered_map<quad_segment, unsigned int, quad_seg_hasher> segment_dict;
	for (auto &it : quads) {
		std::vector<quad_segment> segments = it.get_segments();
		for (auto &jt: segments) {
			auto found_seg = segment_dict.find(jt);
			if (found_seg != segment_dict.end()) {
				found_seg->second++;
			} else {
				segment_dict.insert({jt,1});
			}
		}
	}

	std::vector<quad_segment> boundary_quads;
	for (auto &it : segment_dict) {
		if (it.second == 1) {
			boundary_quads.push_back(it.first);
		}
	}

	printf("found %lu boundary quads!\n", boundary_quads.size());

	//remap to linear indices
	std::unordered_map<unsigned int, unsigned int> idx_dict;
	std::vector<vec3_t> verts_linear;
	unsigned int iter = 0;
	for (auto &it : verts) {
		idx_dict.insert({it.first, iter});
		verts_linear.push_back(it.second);
		iter++;
	}

	for (auto &it: boundary_quads) {
		it.i1 = idx_dict[it.i1];
		it.i2 = idx_dict[it.i2];
		it.i3 = idx_dict[it.i3];
		it.i4 = idx_dict[it.i4];
	}

	//triangulate quads
	triangles = quads_to_tris(boundary_quads);
	positions = verts_linear;
}

void ldynak_read_triangles_from_tetmesh(std::string file_name, std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions) {
	std::ifstream infile(file_name, std::ifstream::in);
	std::vector<tet> tets = read_tets(infile);
	printf("tets read! %lu\n", tets.size());

	if (tets.size() == 0) {
		printf("WARNING: problem reading lsdyna file! zero tests!\n");
	}

	infile.clear();
	infile.seekg(0);

	std::unordered_map<int, vec3_t> verts = read_vertices(infile);
	printf("verts read! %lu\n", verts.size());

	struct tet_seg_hasher {
		std::size_t operator()(const tet_segment& e) const {
			return std::hash<unsigned int>()(e.i1) + std::hash<unsigned int>()(e.i2) + std::hash<unsigned int>()(e.i3);
		}
	};

	//identify segments wich belong to only one tet (=boundary)
	std::unordered_map<tet_segment, unsigned int, tet_seg_hasher> segment_dict;
	for (auto &it : tets) {
		std::vector<tet_segment> segments = it.get_segments();
		for (auto &jt: segments) {
			auto found_seg = segment_dict.find(jt);
			if (found_seg != segment_dict.end()) {
				found_seg->second++;
			} else {
				segment_dict.insert({jt,1});
			}
		}
	}

	std::vector<tet_segment> boundary_tets;
	for (auto &it : segment_dict) {
		if (it.second == 1) {
			boundary_tets.push_back(it.first);
		}
	}

	printf("found %lu boundary tets!\n", boundary_tets.size());

	//remap to linear indices
	std::unordered_map<unsigned int, unsigned int> idx_dict;
	std::vector<vec3_t> verts_linear;
	unsigned int iter = 0;
	for (auto &it : verts) {
		idx_dict.insert({it.first, iter});
		verts_linear.push_back(it.second);
		iter++;
	}

	for (auto &it: boundary_tets) {
		it.i1 = idx_dict[it.i1];
		it.i2 = idx_dict[it.i2];
		it.i3 = idx_dict[it.i3];
	}

	//triangulate quads
	triangles = tets_to_tris(boundary_tets);
	positions = verts_linear;
}

void ldynak_dump(std::string file_name, std::vector<mesh_triangle> &triangles, std::vector<vec3_t> &positions) {
	FILE *fp = fopen(file_name.c_str(), "w+");

	fprintf(fp, "*KEYWORD\n");
	fprintf(fp, "*PART\n");
	fprintf(fp, "tool\n");
	fprintf(fp, "         1         0         0         0         0         0         0         0\n");

	fprintf(fp, "*ELEMENT_SHELL\n");

	int el_iter = 1;
	int pid = 1;
	for (auto &it : triangles) {
		fprintf(fp, "%8d%8d%8d%8d%8d\n", el_iter, pid, it.i1+1, it.i2+1, it.i3+1);
		el_iter++;
	}

	int node_iter = 1;
	fprintf(fp, "*NODE\n");
	for (auto &it : positions) {
		fprintf(fp, "%8d   %13e   %13e   %13e       0       0\n", node_iter, it.x, it.y, it.z);
		node_iter++;
	}

	fprintf(fp, "*END\n");

	fclose(fp);
}

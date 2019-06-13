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

#include "vtk_writer.h"

bool m_individual_tool_files = false;

void vtk_writer_write_blanking() {
	char buf[256];
	sprintf(buf, "results/vtk_bbox_%06d.vtk", global_time_step);
	FILE *fp = fopen(buf, "w+");

	vec3_t bbmin, bbmax;
	global_blanking->get_bb(bbmin, bbmax);

	const int num_face = 6;
	const int num_corner = 8;

	//generate 8 corners of tool bbox
	vec3_t c000(bbmin.x, bbmin.y, bbmin.z);
	vec3_t c100(bbmax.x, bbmin.y, bbmin.z);
	vec3_t c110(bbmax.x, bbmax.y, bbmin.z);
	vec3_t c010(bbmin.x, bbmax.y, bbmin.z);

	vec3_t c001(bbmin.x, bbmin.y, bbmax.z);
	vec3_t c101(bbmax.x, bbmin.y, bbmax.z);
	vec3_t c111(bbmax.x, bbmax.y, bbmax.z);
	vec3_t c011(bbmin.x, bbmax.y, bbmax.z);

	std::vector<vec3_t> corners({c000, c100, c110, c010, c001, c101, c111, c011});

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");

	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");			// Particle positions
	fprintf(fp, "POINTS %d float\n", num_corner);

	for (auto &it : corners) {
		fprintf(fp, "%f %f %f\n", it.x, it.y, it.z);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", num_face);
	for (int i = 0; i < num_face; i++) {
		fprintf(fp, "%d\n", 9);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", num_face, 5*num_face);
	fprintf(fp, "4 %d %d %d %d\n", 0, 1, 2, 3);
	fprintf(fp, "4 %d %d %d %d\n", 0, 1, 5, 4);
	fprintf(fp, "4 %d %d %d %d\n", 1, 2, 6, 5);
	fprintf(fp, "4 %d %d %d %d\n", 0, 3, 7, 4);
	fprintf(fp, "4 %d %d %d %d\n", 3, 2, 6, 7);
	fprintf(fp, "4 %d %d %d %d\n", 4, 5, 6, 7);

	fprintf(fp, "\n");

	fclose(fp);
}

void vtk_writer_write(particle_gpu *particles) {
	static int      *h_idx		= 0;
	static float4_t *h_pos		= 0;
	static float4_t *h_vel		= 0;
	static float3_t *h_vel_t	= 0;
	static float_t  *h_rho		= 0;
	static float_t  *h_h		= 0;
	static float_t  *h_p		= 0;
	static float_t  *h_T		= 0;
	static float_t  *h_eps		= 0;

	static mat3x3_t *h_S		= 0;
	static mat3x3_t *h_S_t		= 0;
	static mat3x3_t *h_S_der	= 0;
	static mat3x3_t *h_v_der	= 0;

	static float_t *h_fixed		= 0;
	static float_t *h_blanked	= 0;
	static float_t *h_tool_p	= 0;
	static float3_t *h_contact  = 0;

	if (h_idx == 0) {
		int n_init = particles->N_init;

		// Memory allocation only upon first call;
		h_idx		= new int[n_init];
		h_pos		= new float4_t[n_init];
		h_vel		= new float4_t[n_init];
		h_vel_t		= new float3_t[n_init];
		h_vel_t		= new float3_t[n_init];
		h_rho		= new float_t[n_init];
		h_h			= new float_t[n_init];
		h_p			= new float_t[n_init];
		h_T			= new float_t[n_init];
		h_eps		= new float_t[n_init];

		h_S			= new mat3x3_t[n_init];
		h_S_t		= new mat3x3_t[n_init];
		h_S_der		= new mat3x3_t[n_init];
		h_v_der		= new mat3x3_t[n_init];

		h_fixed		= new float_t[n_init];
		h_blanked	= new float_t[n_init];
		h_tool_p	= new float_t[n_init];
		h_contact   = new float3_t[n_init];
	}

	int n = particles->N;

	cudaMemcpy(h_idx, particles->idx,    sizeof(int)*n,      cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pos, particles->pos,    sizeof(float4_t)*n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vel, particles->vel,    sizeof(float4_t)*n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vel_t, particles->vel_t,sizeof(float3_t)*n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rho, particles->rho,    sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_h,   particles->h,      sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_p,   particles->p,      sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_T,   particles->T,	 sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_eps, particles->eps_pl, sizeof(float_t)*n,  cudaMemcpyDeviceToHost);

	cudaMemcpy(h_S_der, particles->S_der, sizeof(mat3x3_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_S_t,   particles->S_t,   sizeof(mat3x3_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_S,     particles->S,     sizeof(mat3x3_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_v_der, particles->v_der, sizeof(mat3x3_t)*n,  cudaMemcpyDeviceToHost);

	cudaMemcpy(h_fixed, particles->fixed, sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_blanked, particles->blanked, sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_tool_p, particles->tool_particle, sizeof(float_t)*n,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_contact, particles->fc, sizeof(float3_t)*n,  cudaMemcpyDeviceToHost);

	int num_unblanked_part = 0;
	for (int i = 0; i < n; i++) {
		if (h_blanked[i] != 1.) {
			num_unblanked_part++;
		}
	}

	char buf[256];
	sprintf(buf, "results/vtk_out_%06d.vtk", global_time_step);
	FILE *fp = fopen(buf, "w+");

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");

	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");			// Particle positions
	fprintf(fp, "POINTS %d float\n", num_unblanked_part);
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%f %f %f\n", h_pos[i].x, h_pos[i].y, h_pos[i].z);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", num_unblanked_part, 2*num_unblanked_part);
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%d %d\n", 1, i);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", num_unblanked_part);
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%d\n", 1);
	}
	fprintf(fp, "\n");

	fprintf(fp, "POINT_DATA %d\n", num_unblanked_part);

	fprintf(fp, "SCALARS density float 1\n");		// Current particle density
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%f\n", h_rho[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS contact float 1\n");		// Current particle density
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%f\n", glm::length(vec3_t(h_contact[i].x, h_contact[i].y, h_contact[i].z)) > 0. ? 1. : 0.);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Temperature float 1\n");		// Current particle temperature
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%f\n", h_T[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Fixed float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%f\n", h_fixed[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Tool float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%f\n", h_tool_p[i]);
	}
	fprintf(fp, "\n");

//	fprintf(fp, "SCALARS Contact float 1\n");		// Current particle temperature
//	fprintf(fp, "LOOKUP_TABLE default\n");
//	for (unsigned int i = 0; i < n; i++) {
//		if (h_blanked[i]==1.) continue;
//		fprintf(fp, "%f\n", sqrt(h_contact[i].x*h_contact[i].x + h_contact[i].y*h_contact[i].y + h_contact[i].z*h_contact[i].z)*1e16);
//	}
//	fprintf(fp, "\n");

	fprintf(fp, "SCALARS EquivAccumPlasticStrain float 1\n");	// equivalent accumulated plastic strain
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		fprintf(fp, "%f\n", h_eps[i]);
	}
	fprintf(fp, "\n");


	fprintf(fp, "SCALARS Svm float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++) {
		if (h_blanked[i]==1.) continue;
		double sxx = h_S[i][0][0] - h_p[i];
		double sxy = h_S[i][0][1];
		double sxz = h_S[i][0][2];
		double syy = h_S[i][1][1] - h_p[i];
		double syz = h_S[i][1][2];
		double szz = h_S[i][2][2] - h_p[i];

		double svm2 = (sxx*sxx + syy*syy + szz*szz) - sxx * syy - sxx * szz - syy * szz + 3.0 * (sxy*sxy + syz*syz + sxz*sxz);
		double svm = (svm2 > 0) ? sqrt(svm2) : 0.;
		fprintf(fp, "%f\n", svm);
	}
	fprintf(fp, "\n");

	fclose(fp);
}


void vtk_writer_write(tool_3d_gpu *tool, char filename[256]) {		// write tool, based on iwf_mfree version, hand in filename
	auto tris = tool->get_cpu_tris();
	auto positions = tool->get_cpu_pos();

	unsigned int ntri = tris.size();						// numtool_3d_gpu *toolber of triangles
	unsigned int npos  = positions.size();					// number of positions

	char buf[256];
	sprintf(buf, "%s_%06d.vtk", filename, global_time_step);
	FILE *fp = fopen(buf, "w+");

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");

	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
	fprintf(fp, "POINTS %d float\n", npos);

	for (auto it = positions.begin(); it != positions.end(); ++it) {
		glm::dvec3 pos(it->x, it->y, it->z);
		fprintf(fp, "%f %f %f\n", pos.x, pos.y, pos.z);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", ntri, 3*ntri+ntri);

	for (auto it = tris.begin(); it != tris.end(); ++it) {
		fprintf(fp, "%d %d %d %d\n", 3, it->i1, it->i2, it->i3);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", ntri);
	for (auto it = tris.begin(); it != tris.end(); ++it) {
		fprintf(fp, "%d\n", 5);
	}

	fclose(fp);

}

void vtk_writer_write_unified(std::vector<tool_3d_gpu *> tools, char filename[256]) {		// write tool, based on iwf_mfree version, hand in filename

	unsigned int ntri = 0;
	unsigned int npos = 0;
	for (auto &tool : tools) {
		auto tris = tool->get_cpu_tris();
		auto positions = tool->get_cpu_pos();

		ntri += tris.size();						// numtool_3d_gpu *toolber of triangles
		npos += positions.size();					// number of positions
	}

	char buf[256];
	sprintf(buf, "%s_%06d.vtk", filename, global_time_step);
	FILE *fp = fopen(buf, "w+");

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");

	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
	fprintf(fp, "POINTS %d float\n", npos);

	std::vector<int> offsets;
	offsets.push_back(0);
	for (auto &tool : tools) {
		auto positions = tool->get_cpu_pos();
		for (auto it = positions.begin(); it != positions.end(); ++it) {
			glm::dvec3 pos(it->x, it->y, it->z);
			fprintf(fp, "%f %f %f\n", pos.x, pos.y, pos.z);
		}
		offsets.push_back(positions.size());
	}
	fprintf(fp, "\n");

	std::vector<int> prefix_sum(offsets.size());
	std::partial_sum (offsets.begin(), offsets.end(), prefix_sum.begin());

	fprintf(fp, "CELLS %d %d\n", ntri, 3*ntri+ntri);
	unsigned int iter = 0;
	for (auto &tool : tools) {
		auto tris = tool->get_cpu_tris();
		int offset = prefix_sum[iter];
		for (auto it = tris.begin(); it != tris.end(); ++it) {
			fprintf(fp, "%d %d %d %d\n", 3, it->i1+offset, it->i2+offset, it->i3+offset);
		}
		iter++;
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", ntri);
	for (auto &tool : tools) {
		auto tris = tool->get_cpu_tris();
		for (auto it = tris.begin(); it != tris.end(); ++it) {
			fprintf(fp, "%d\n", 5);
		}
	}

	fprintf(fp, "POINT_DATA %d\n", npos);

	fprintf(fp, "SCALARS id float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (int i = 0; i < tools.size(); i++) {
		for (auto &it : tools[i]->get_cpu_pos()) {
			fprintf(fp, "%d\n", i);
		}
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS active float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (int i = 0; i < tools.size(); i++) {
		for (auto &it : tools[i]->get_cpu_pos()) {
			fprintf(fp, "%d\n", (tools[i]->is_alive()) ? 1 : 0);
		}
	}
	fprintf(fp, "\n");

	fclose(fp);

}

void vtk_writer_write(particle_gpu *particles, tool_3d_gpu *tool) {
	char buf[256] = "results/vtk_tool_";

	vtk_writer_write(particles);
	vtk_writer_write(tool, buf);
}

void vtk_writer_write(tool_3d_gpu *tool, particle_gpu *particles) {
	char buf[256] = "results/vtk_tool_";

	vtk_writer_write(particles);
	vtk_writer_write(tool, buf);
}

void vtk_writer_write(std::vector<tool_3d_gpu *> tool, particle_gpu *particles) {

	vtk_writer_write(particles);

	unsigned int number_of_tool = 0;
	char filename[256] = "results/vtk_tool";

	if (!m_individual_tool_files) {
		vtk_writer_write_unified(tool, filename);
	} else {
		for (auto current_tool : tool) {
			sprintf(filename, "results/vtk_tool_%01d", number_of_tool);
			vtk_writer_write(current_tool, filename);
			number_of_tool++;
		}
	}

	if (global_blanking) {
		vtk_writer_write_blanking();
	}
}

void vtk_writer_write(particle_gpu *particles, std::vector<tool_3d_gpu *> tool) {	// write multiple tools  & particles, based on iwf_mfree version; HK, Mo, 25.06.2018
	vtk_writer_write(tool, particles);
}

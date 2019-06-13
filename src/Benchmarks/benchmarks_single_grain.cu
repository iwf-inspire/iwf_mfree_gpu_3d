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

#include "benchmarks_single_grain.h"

template<typename Out>
static void split(const std::string &s, char delim, Out &result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        result.push_back(item);
    }
}

particle_gpu *setup_single_grain_5tool(grid_base **grid) {

	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	joco_constants joco = make_joco_constants();

	trml_constants trml_wp = make_trml_constants();
	trml_constants trml_tool = make_trml_constants();

	bool sample_tools = true;
	bool interp_steady_temp = true;
	bool separate = false;
	bool thermal = true;

	float_t target_feed = 20*1e-6*100;
	int depth_fac = 3;
	float_t depth_wp = depth_fac*target_feed;

	float_t num_particles_in_feed = 12;     //12 = 1.5M is resolution used in paper
											// if initial positions are loaded from a result file, make sure that mass and smoothing length
											// are the same as the ones used to produce the result file. For ``init_hires.vtk'' this is 12

	int nz = depth_fac*num_particles_in_feed;

	float_t dz = depth_wp/(nz-1);

	std::vector<mesh_triangle> triangles;
	std::vector<vec3_t> positions;
	ldynak_read_triangles_from_tetmesh("pin25_reduziert_20062018.k", triangles, positions);

	vec3_t bbmin_tool, bbmax_tool;
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);
	geometry_translate(triangles, positions, vec3_t(0., 0., -bbmin_tool.z));	//lift tool to zero level
	geometry_translate(triangles, positions, vec3_t(0., 0., -target_feed ));		//push down to target feed again

	float_t ly_tool = bbmax_tool.y - bbmin_tool.y;
	geometry_translate(triangles, positions, vec3_t(0.,-bbmin_tool.y - ly_tool/2., 0.));		//center tool to y zero

	float_t nudge = 6.4e-4;
	float_t lx_tool = bbmax_tool.x - bbmin_tool.x;
	geometry_translate(triangles, positions, vec3_t(-bbmin_tool.x - lx_tool + nudge, 0., 0.));	//move tool in x direction such that bottom surface almost touches wp

	//update bb
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);

	std::vector<mesh_triangle> triangles_l_l = triangles;	//leading left
	std::vector<vec3_t> positions_l_l = positions;

	std::vector<mesh_triangle> triangles_l_r = triangles;	//leading right
	std::vector<vec3_t> positions_l_r = positions;

	std::vector<mesh_triangle> triangles_t_l = triangles;	//trailing left
	std::vector<vec3_t> positions_t_l = positions;

	std::vector<mesh_triangle> triangles_t_r = triangles;	//trailing right
	std::vector<vec3_t> positions_t_r = positions;

	float_t edge_diamond = 80*1e-6*100;
	float_t lx = 3*edge_diamond;
	float_t ly = 3*edge_diamond + 2*edge_diamond + 1*edge_diamond;
	float_t lz = depth_wp;

	float_t sp_length = (separate) ? 0.5*edge_diamond : 0;

	geometry_translate(triangles_l_l, positions_l_l, vec3_t(0., -1.5*edge_diamond - sp_length, 0.));
	geometry_translate(triangles_l_r, positions_l_r, vec3_t(0., +1.5*edge_diamond + sp_length, 0.));
	geometry_translate(triangles_t_l, positions_t_l, vec3_t(-edge_diamond, -0.5*edge_diamond - sp_length, 0.));
	geometry_translate(triangles_t_r, positions_t_r, vec3_t(-edge_diamond, +0.5*edge_diamond + sp_length, 0.));

//	geometry_translate(triangles,     positions,     vec3_t(lx*0.75, 0., 0.));
//	geometry_translate(triangles_l_l, positions_l_l, vec3_t(lx*0.75, 0., 0.));
//	geometry_translate(triangles_l_r, positions_l_r, vec3_t(lx*0.75, 0., 0.));
//	geometry_translate(triangles_t_l, positions_t_l, vec3_t(lx*0.75, 0., 0.));
//	geometry_translate(triangles_t_r, positions_t_r, vec3_t(lx*0.75, 0., 0.));

//	geometry_translate(triangles_t_l, positions_t_l, vec3_t(-lx, 0., 0.));
//	geometry_translate(triangles_t_r, positions_t_r, vec3_t(-lx, 0., 0.));

	int nx = lx/dz;	//equi particle spacing
	int ny = ly/dz;

	int n = nx*ny*nz;

	int part_iter = 0;
	float4_t* pos = new float4_t[n];
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			for (int k = 0; k < nz; k++) {
				float_t px = i*dz; float_t py = j*dz-ly/2.; float_t pz = - k*dz;
				float4_t cur_pos;

				cur_pos.x = px;
				cur_pos.y = py;
				cur_pos.z = pz;

				pos[part_iter] = cur_pos;

				part_iter++;
			}
		}
	}

//	// read result file for positions if desired
//	auto vec_pos = vtk_read_init_pos("init_hires.vtk");
//	n = vec_pos.size();
//	float4_t* pos = new float4_t[n];
//	unsigned int pos_it = 0;
//	for (unsigned int i = 0; i < n; i++) {
//		pos[pos_it] = vec_pos[i];
//		pos_it++;
//	}
//	n = pos_it;

	printf("calculating with %d regular particles\n", n);

	float_t hdx = 1.7;
	phys.E    = 1.1;
	phys.nu   = 0.35;
	phys.rho0 = 4.43;
	phys.G    = phys.E/(2.*(1.+phys.nu));
	phys.K    = 2.0*phys.G*(1+phys.nu)/(3*(1-2*phys.nu));
	phys.mass = dz*dz*dz*phys.rho0;

	bool use_khan = true;

	if (use_khan) {
		joco.A		= 0.0110400;	// Mansur Akbari 2016, Tabelle 1
		joco.B		= 0.0103600;	// Mansur Akbari 2016, Tabelle 1
		joco.C		= 0.0139000;	// Mansur Akbari 2016, Tabelle 1
		joco.m		= 0.7794;		// Mansur Akbari 2016, Tabelle 1
		joco.n		= 0.6349;		// Mansur Akbari 2016, Tabelle 1
		joco.Tref	= 300.;
		joco.Tmelt	= 1723.0000;	// Mansur Akbari 2016, im Text S.65
		joco.eps_dot_ref = 1e-6;	// Mansur Akbari 2016, im Text S.65
		joco.clamp_temp = 1.;
	} else {
		joco.A		= 0.0086200;
		joco.B		= 0.0033100;
		joco.C		= 0.0100000;
		joco.m		= 0.8;
		joco.n		= 0.34;
		joco.Tref	= 300.;
		joco.Tmelt	= 1836.0000;
		joco.eps_dot_ref = 1e-6;
		joco.clamp_temp = 1.;
	}

	float_t rho0_tool = 3.5;	// https://en.wikipedia.org/wiki/Diamond
	trml_wp.T_init = joco.Tref;
	if (thermal) {
		trml_wp.cp = 553*1e-8;			// Heat Capacity
		trml_wp.tq = 0.9;				// Taylor-Quinney Coefficient
		trml_wp.k  = 7.1*1e-13;			// Thermal Conduction
		trml_wp.alpha = trml_wp.k/(phys.rho0*trml_wp.cp);	// Thermal diffusivity
		trml_wp.eta = 0.9;

		trml_tool.cp = 520*1e-8;	// https://www.engineeringtoolbox.com/specific-heat-solids-d_154.html
		trml_tool.tq = 0.;			// no plastic deformation in diamond
		trml_tool.k = 2200*1e-13;	// https://en.wikipedia.org/wiki/Material_properties_of_diamond
		trml_tool.alpha = trml_tool.k/(rho0_tool*trml_tool.cp);
	}

	float_t c0 = sqrt(phys.K/phys.rho0);

	corr.alpha   = 1.;
	corr.beta    = 1.;
	corr.eta     = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3;
	{
		float_t h1   = 1./(hdx*dz);
		float_t q    = dz*h1;
		float_t fac  = (M_1_PI)*h1*h1*h1;;
		corr.wdeltap = fac*(1 - 1.5*q*q*(1-0.5*q));
	}

	float4_t *vel    = new float4_t[n];
	float_t  *h      = new float_t[n];
	float_t  *rho    = new float_t[n];
	float_t  *fixed  = new float_t[n];
	float_t  *T      = new float_t[n];
	float_t  *tool_p = new float_t[n];

	for (int i = 0; i < n; i++) {	// initialize particles
		rho[i]   = phys.rho0;
		h[i]     = hdx*dz;
		vel[i].x =   0.;
		vel[i].y =   0.;
		vel[i].z =   0.;
		fixed[i] =   0.;
		T[i]     = joco.Tref;
		tool_p[i] = 0.;
	}

	unsigned int Nr_BC_particles = 0;	// Counter boundary particles
	for (unsigned int i = 0; i < n; i++) {	// define boundary particles
		//if (pos[i].z < dz/2) {
		if (pos[i].z < -lz+dz/2.0) {
			fixed[i] = 1;
			Nr_BC_particles++;
		}
	}

	//---------------------------------------------------------------------

//	float_t v_tool = 8e-5;	//speed akbari
	float_t v_tool = 3e-3;	//speed KW
//	float_t v_tool = 570*100/1e6; //reckless speed

	float_t friction_coeff = 0.35;	//fric ruttiumann
//	float_t friction_coeff = 0.6;	//fric akbari
//	float_t friction_coeff = 0.8;	//fric akbari increased

	mesh_compute_vertex_normals(triangles, positions);
	mesh_compute_vertex_normals(triangles_l_l, positions_l_l);
	mesh_compute_vertex_normals(triangles_l_r, positions_l_r);
	mesh_compute_vertex_normals(triangles_t_l, positions_t_l);
	mesh_compute_vertex_normals(triangles_t_r, positions_t_r);

	float_t avg_h = mesh_average_edge_length(triangles);

	std::vector<tool_3d*> cpu_tools;

	//remove this grain for pattern 3
	auto cpu_tool_l_m = new tool_3d(triangles, positions, avg_h);
	cpu_tools.push_back(cpu_tool_l_m);
	global_tool.push_back(new tool_3d_gpu(cpu_tool_l_m, vec3_t(v_tool, 0., 0.), phys));

	auto cpu_tool_l_l = new tool_3d(triangles_l_l, positions_l_l, avg_h);
	cpu_tools.push_back(cpu_tool_l_l);
	global_tool.push_back(new tool_3d_gpu(cpu_tool_l_l, vec3_t(v_tool, 0., 0.), phys));
	auto cpu_tool_l_r = new tool_3d(triangles_l_r, positions_l_r, avg_h);
	cpu_tools.push_back(cpu_tool_l_r);
	global_tool.push_back(new tool_3d_gpu(cpu_tool_l_r, vec3_t(v_tool, 0., 0.), phys));

	auto cpu_tool_t_l = new tool_3d(triangles_t_l, positions_t_l, avg_h);
	cpu_tools.push_back(cpu_tool_t_l);
	global_tool.push_back(new tool_3d_gpu(cpu_tool_t_l, vec3_t(v_tool, 0., 0.), phys));
	auto cpu_tool_t_r = new tool_3d(triangles_t_r, positions_t_r, avg_h);
	cpu_tools.push_back(cpu_tool_t_r);
	global_tool.push_back(new tool_3d_gpu(cpu_tool_t_r, vec3_t(v_tool, 0., 0.), phys));

	for (auto &it : global_tool)  {
		it->set_algorithm_type(tool_3d_gpu::contact_algorithm::spatial_hashing);
		it->set_mu(friction_coeff);
	}

	int n_tool = 0;
	std::vector<float4_t> steady_state_tool;
	float3_t bbmin_steady = make_float3_t(FLT_MAX, FLT_MAX, FLT_MAX);
	float_t dx_steady = 0.;

	if (interp_steady_temp) {
		std::string line;
		std::ifstream infile("thermal_steady_lowres.txt", std::ifstream::in);

		while (std::getline(infile, line)) {
			std::vector<std::string> tokens;
			split(line, ' ', tokens);
			if (tokens.size() != 4) {
				continue;
			}

			float_t x = std::stod (tokens[0], NULL);
			float_t y = std::stod (tokens[1], NULL);
			float_t z = std::stod (tokens[2], NULL);
			float_t t = std::stod (tokens[3], NULL);

			bbmin_steady.x = std::min(x, bbmin_steady.x);
			bbmin_steady.y = std::min(y, bbmin_steady.y);
			bbmin_steady.z = std::min(z, bbmin_steady.z);

			steady_state_tool.push_back(make_float4_t(x,y,z,t));
		}

		float_t ddx = fabs(steady_state_tool[0].x - steady_state_tool[1].x);
		float_t ddy = fabs(steady_state_tool[0].y - steady_state_tool[1].y);
		float_t ddz = fabs(steady_state_tool[0].z - steady_state_tool[1].z);

		dx_steady = std::max(std::max(ddx, ddy), ddz);
	}

	if (sample_tools) {
		std::vector<float4_t> samples;
		for (auto it : cpu_tools) {
			std::vector<vec3_t> cur_samples = it->sample(dz);
			std::vector<float4_t> cur_samples_flt_4;
			std::vector<float4_t> shifted_steady_tool(steady_state_tool.size());
			float3_t bbmin_cur_tool = make_float3_t(FLT_MAX, FLT_MAX, FLT_MAX);

			for (auto jt: cur_samples) {
				cur_samples_flt_4.push_back(make_float4_t(jt.x, jt.y, jt.z, 0.));

				bbmin_cur_tool.x = std::min(jt.x, bbmin_cur_tool.x);
				bbmin_cur_tool.y = std::min(jt.y, bbmin_cur_tool.y);
				bbmin_cur_tool.z = std::min(jt.z, bbmin_cur_tool.z);
			}

			if (interp_steady_temp) {
				for (unsigned int i = 0; i < steady_state_tool.size(); i++) {
					shifted_steady_tool[i].x = steady_state_tool[i].x - bbmin_steady.x + bbmin_cur_tool.x;
					shifted_steady_tool[i].y = steady_state_tool[i].y - bbmin_steady.y + bbmin_cur_tool.y;
					shifted_steady_tool[i].z = steady_state_tool[i].z - bbmin_steady.z + bbmin_cur_tool.z;
					shifted_steady_tool[i].w = steady_state_tool[i].w;
				}
				interp_temps(shifted_steady_tool, dx_steady*dx_steady*dx_steady, std::max(float_t(1.5)*dx_steady, hdx*dz), cur_samples_flt_4);
			}

			samples.insert(samples.end(), cur_samples_flt_4.begin(), cur_samples_flt_4.end());
		}

		n_tool = samples.size();
		float4_t *tool_pos = new float4_t[n_tool];

		float3_t bbmin_cur_tool = make_float3_t(FLT_MAX, FLT_MAX, FLT_MAX);

		for (int i = 0; i < n_tool; i++) {
			tool_pos[i].x = samples[i].x;
			tool_pos[i].y = samples[i].y;
			tool_pos[i].z = samples[i].z;
			tool_pos[i].w = samples[i].w;
		}

		pos    = (float4_t*) realloc(pos,    sizeof(float4_t)*(n+n_tool));
		vel    = (float4_t*) realloc(vel,    sizeof(float4_t)*(n+n_tool));
		rho    = (float_t*)  realloc(rho,    sizeof(float_t)*(n+n_tool));
		T      = (float_t*)  realloc(T,      sizeof(float_t)*(n+n_tool));
		h      = (float_t*)  realloc(h,      sizeof(float_t)*(n+n_tool));
		fixed  = (float_t*)  realloc(fixed,  sizeof(float_t)*(n+n_tool));
		tool_p = (float_t*)  realloc(tool_p, sizeof(float_t)*(n+n_tool));

		int tool_pos_iter = 0;
		for (int i = n; i < n+n_tool; i++) {
			pos[i] = tool_pos[tool_pos_iter];
			tool_pos_iter++;

			rho[i]   = rho0_tool;
			h[i]     = hdx*dz;
			vel[i].x =   0.;
			vel[i].y =   0.;
			vel[i].z =   0.;
			fixed[i] =   0.;
			T[i]     = joco.Tref;
			if (interp_steady_temp) {
				T[i] = fmax(tool_pos[tool_pos_iter].w, joco.Tref);
			}
			tool_p[i] = 1.;
		}

		//fix back boundaries of grains
		for (int i = n; i < n+n_tool; i++) {
			//top
			if (pos[i].z > bbmax_tool.z - 1.5*dz) {
				fixed[i] = 1.;
				Nr_BC_particles++;
			}

			//back plane leading
			//left
			vec3_t p1_l_l(-0.0082535, -0.0179347, -2.41e-05);
			vec3_t p2_l_l(-0.0067885, -0.019261, 0.0019517);
			vec3_t p3_l_l(-0.0013051, -0.011949, -2.41e-05);
			//middle
			vec3_t p1_l_m(-0.0082535, -0.00193475, -2.41e-05);
			vec3_t p2_l_m(-0.0067885, -0.00326105, 0.0019517);
			vec3_t p3_l_m(-0.0013051, 0.00405095, -2.41e-05);
			//right
			vec3_t p1_l_r(-0.0082535, 0.0140652, -2.41e-05);
			vec3_t p2_l_r(-0.0067885, 0.012739, 0.0019517);
			vec3_t p3_l_r(-0.0013051, 0.020051, -2.41e-05);

			//back plane trailing
			//left
			vec3_t p1_t_l(-0.0162535, -0.00993475, -2.41e-05);
			vec3_t p2_t_l(-0.0147885, -0.011261, 0.0019517);
			vec3_t p3_t_l(-0.0093051, -0.00394905, -2.41e-05);
			//right
			vec3_t p1_t_r(-0.0162535, 0.00606525, -2.41e-05);
			vec3_t p2_t_r(-0.0147885, 0.00473895, 0.0019517);
			vec3_t p3_t_r(-0.0093051, 0.0120509, -2.41e-05);

			vec3_t nrm_l_l = glm::cross(p2_l_l-p1_l_l, p2_l_l-p3_l_l);
			vec3_t nrm_l_m = glm::cross(p2_l_m-p1_l_m, p2_l_m-p3_l_m);
			vec3_t nrm_l_r = glm::cross(p2_l_r-p1_l_r, p2_l_r-p3_l_r);
			vec3_t nrm_t_l = glm::cross(p2_t_l-p1_t_l, p2_t_l-p3_t_l);
			vec3_t nrm_t_r = glm::cross(p2_t_r-p1_t_r, p2_t_r-p3_t_r);

			vec3_t qp(pos[i].x, pos[i].y, pos[i].z);
			float_t d_l_l = fabs(glm::dot(nrm_l_l, qp-p1_l_l)/glm::length(nrm_l_l));
			float_t d_l_m = fabs(glm::dot(nrm_l_m, qp-p1_l_m)/glm::length(nrm_l_m));
			float_t d_l_r = fabs(glm::dot(nrm_l_r, qp-p1_l_r)/glm::length(nrm_l_r));
			float_t d_t_l = fabs(glm::dot(nrm_t_l, qp-p1_t_l)/glm::length(nrm_t_l));
			float_t d_t_r = fabs(glm::dot(nrm_t_r, qp-p1_t_r)/glm::length(nrm_t_r));

			if (d_l_l < 2*dz || d_l_m < 2*dz || d_l_r < 2*dz || d_t_l < 2*dz || d_t_r < 2*dz) {
				fixed[i] = 1.;
				Nr_BC_particles++;
			}

		}

		printf("calculating with %d tool particles\n", n_tool);
	}

	//---------------------------------------------------------------------

	const float_t CFL = 0.3;
	float_t delta_t_max = CFL*hdx*dz/(sqrt(v_tool) + c0);
	global_time_dt = 0.5*delta_t_max;
	global_time_final = (lx+2*edge_diamond)/v_tool;

    printf("t final %f\n", global_time_final);

	printf("max recommended dt %e, dt used %e\n", delta_t_max, global_time_dt);

	//---------------------------------------------------------------------

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, tool_p, n+n_tool);
	assert(check_cuda_error());

	*grid = new grid_gpu_green(n+n_tool, make_float3_t(std::min(-15.0*dz, bbmin_tool.x-15.0*dz - edge_diamond), -15.0*dz-ly/2, -1.5*nz*dz), make_float3_t(lx+15.0*dz, ly+15.0*dz-ly/2, 3.0*bbmax_tool.z), hdx*dz);
	global_blanking = new blanking(vec3_t(std::min(-15.0*dz, bbmin_tool.x - 15.0*dz - edge_diamond), -15.0*dz-ly/2, -1.5*nz*dz), vec3_t(lx+15.0*dz, ly+15.0*dz-ly/2, 3.0*bbmax_tool.z), 2.0*2.0);

	//---------------------------------------------------------------------

	printf("Number of BC-particles: %u \n", Nr_BC_particles);

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco);
	actions_setup_thermal_constants_wp(trml_wp);
	actions_setup_thermal_constants_tool(trml_tool);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_workpiece(trml_wp);
	interactions_setup_thermal_constants_tool(trml_tool, global_tool[0]);
	interactions_setup_geometry_constants(*grid);

	global_tool_forces = new tool_forces(global_tool.size());

	return particles;
}

particle_gpu *setup_single_grain_1tool_realscale(grid_base **grid) {
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml = make_trml_constants();
	joco_constants joco = make_joco_constants();

	float_t target_feed = 30*1e-6*100;	//30 mu in cm
	int depth_fac = 3;
	float_t depth_wp = depth_fac*target_feed;

	float_t num_particles_in_feed = 5;		//5 = 1.2M particles, which was used for the paper
	int nz = depth_fac*num_particles_in_feed;

	float_t dz = depth_wp/(nz-1);

	std::vector<mesh_triangle> triangles;
	std::vector<vec3_t> positions;
	ldynak_read_triangles_from_tetmesh("Diamant_von_Mansur_04072018_SDB1125-2025-D851.k", triangles, positions);

	vec3_t bbmin_tool, bbmax_tool;
	geometry_scale(triangles, positions, vec3_t(1./10000.0,1./10000.0,1./10000.0));	//micron to cm
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);
	geometry_translate(triangles, positions, vec3_t(0., 0., -bbmin_tool.z));	//lift tool to zero level
	geometry_translate(triangles, positions, vec3_t(0., 0., -target_feed ));		//push down to target feed again

	float_t ly_tool = bbmax_tool.y - bbmin_tool.y;
	geometry_translate(triangles, positions, vec3_t(0.,-bbmin_tool.y - ly_tool/2., 0.));		//center tool to y zero

	float_t lx_tool = bbmax_tool.x - bbmin_tool.x;
	geometry_translate(triangles, positions, vec3_t(-bbmin_tool.x - lx_tool + 260*100*1e-6, 0., 0.));	//move tool in x direction such that bottom surface almost touches wp

	//tool is at final position, update bbox
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);

	float_t lx = 2*270*1e-6*100;	//from measured base surface
	float_t ly = 2*320*1e-6*100;
	float_t lz = depth_wp;

	int nx = lx/dz;	//equi particle spacing
	int ny = ly/dz;

	int n = nx*ny*nz;
	printf("calculating with %d\n", n);

	int part_iter = 0;
	float4_t* pos = new float4_t[n];
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			for (int k = 0; k < nz; k++) {
				float_t px = i*dz; float_t py = j*dz-ly/2.; float_t pz = - k*dz;
				float4_t cur_pos;

				cur_pos.x = px;
				cur_pos.y = py;
				cur_pos.z = pz;

				pos[part_iter] = cur_pos;

				part_iter++;
			}
		}
	}

	//---------------------------------------------------------------------

	float_t hdx = 1.7;
	phys.E    = 1.1;
	phys.nu   = 0.35;
	phys.rho0 = 4.43;
	phys.G    = phys.E/(2.*(1.+phys.nu));
	phys.K    = 2.0*phys.G*(1+phys.nu)/(3*(1-2*phys.nu));
	phys.mass = dz*dz*dz*phys.rho0;

	bool use_khan = true;

	if (use_khan) {
		joco.A		= 0.0110400;	// Mansur Akbari 2016, Tabelle 1
		joco.B		= 0.0103600;	// Mansur Akbari 2016, Tabelle 1
		joco.C		= 0.0139000;	// Mansur Akbari 2016, Tabelle 1
		joco.m		= 0.7794;		// Mansur Akbari 2016, Tabelle 1
		joco.n		= 0.6349;		// Mansur Akbari 2016, Tabelle 1
		joco.Tref	= 300.;
		joco.Tmelt	= 1723.0000;	// Mansur Akbari 2016, im Text S.65
		joco.eps_dot_ref = 1e-6;	// Mansur Akbari 2016, im Text S.65
		joco.clamp_temp = 1.;
	} else {
		joco.A		= 0.0086200;
		joco.B		= 0.0033100;
		joco.C		= 0.0100000;
		joco.m		= 0.8;
		joco.n		= 0.34;
		joco.Tref	= 300.;
		joco.Tmelt	= 1836.0000;
		joco.eps_dot_ref = 1e-6;
		joco.clamp_temp = 1.;
	}

	trml.cp = 553*1e-8;			// Heat Capacity
	trml.tq = 0.9;				// Taylor-Quinney Coefficient
	trml.k  = 7.1*1e-13;			// Thermal Conduction
	trml.alpha = trml.k/(phys.rho0*trml.cp);	// Thermal diffusivity
	trml.eta = 0.;
	trml.T_init = joco.Tref;

	float_t c0 = sqrt(phys.K/phys.rho0);

	corr.alpha   = 1.;
	corr.beta    = 1.;
	corr.eta     = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3;
	{
		float_t h1   = 1./(hdx*dz);
		float_t q    = dz*h1;
		float_t fac  = (M_1_PI)*h1*h1*h1;;
		corr.wdeltap = fac*(1 - 1.5*q*q*(1-0.5*q));
	}

	float4_t *vel   = new float4_t[n];
	float_t  *h     = new float_t[n];
	float_t  *rho   = new float_t[n];
	float_t  *fixed = new float_t[n];
	float_t  *T     = new float_t[n];

	for (int i = 0; i < n; i++) {	// initialize particles
		rho[i]   = phys.rho0;
		h[i]     = hdx*dz;
		vel[i].x =   0.;
		vel[i].y =   0.;
		vel[i].z =   0.;
		fixed[i] =   0;
		T[i]     = joco.Tref;
	}

	unsigned int Nr_BC_particles = 0;	// Counter boundary particles
	for (unsigned int i = 0; i < n; i++) {	// define boundary particles
		//if (pos[i].z < dz/2) {
		if (pos[i].z < -lz+dz/2.0) {
			fixed[i] = 1;
			Nr_BC_particles++;
		}

		if (pos[i].x > lx - 5*dz) {
			fixed[i] = 1;
			Nr_BC_particles++;
		}
	}

	printf("Number of BC-particles: %u \n", Nr_BC_particles);

	//---------------------------------------------------------------------

//	float_t v_tool = 8e-5;	//speed akbari
	float_t v_tool = 3e-3;	//speed KW
	float_t friction_coeff = 0.35;	//fric ruttiumann
//	float_t friction_coeff = 0.6;	//fric akbari
//	float_t friction_coeff = 0.8;	//fric akbari increased

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, n);
	assert(check_cuda_error());

	mesh_compute_vertex_normals(triangles, positions);
	float_t avg_h = mesh_average_edge_length(triangles);

	auto cpu_tool = new tool_3d(triangles, positions, avg_h);
	global_tool.push_back(new tool_3d_gpu(cpu_tool, vec3_t(v_tool, 0., 0.), phys));

	global_tool[0]->set_algorithm_type(tool_3d_gpu::contact_algorithm::exhaustive);
	global_tool[0]->set_mu(friction_coeff);

	//---------------------------------------------------------------------

	const float_t CFL = 0.3;
	float_t delta_t_max = CFL*hdx*dz/(sqrt(v_tool) + c0);
	global_time_dt = 0.5*delta_t_max;
	global_time_final = lx/v_tool;

	printf("max recommended dt %e, dt used %e\n", delta_t_max, global_time_dt);

	//---------------------------------------------------------------------

	*grid = new grid_gpu_green(n, make_float3_t(-15.0*dz, -15.0*dz-ly/2, -1.5*nz*dz), make_float3_t(lx+15.0*dz, ly+15.0*dz-ly/2, 2.0*bbmax_tool.z), hdx*dz);
	global_blanking = new blanking(vec3_t(-15.0*dz, -15.0*dz-ly/2, -1.5*nz*dz), vec3_t(lx+15.0*dz, ly+15.0*dz-ly/2, 2.0*bbmax_tool.z), 2.0*2.0);

	//---------------------------------------------------------------------

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco);
	actions_setup_thermal_constants_wp(trml);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_workpiece(trml);
	interactions_setup_geometry_constants(*grid);

	global_tool_forces = new tool_forces(global_tool.size());

	return particles;
}

particle_gpu *setup_single_grain_1tool(grid_base **grid) {
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml = make_trml_constants();
	joco_constants joco = make_joco_constants();

	float_t target_feed = 20*1e-6*100;
	int depth_fac = 3;
	float_t depth_wp = depth_fac*target_feed;

	float_t num_particles_in_feed = 18;     //18 = ~2M reference simulation
	int nz = depth_fac*num_particles_in_feed;

	float_t dz = depth_wp/(nz-1);

	std::vector<mesh_triangle> triangles;
	std::vector<vec3_t> positions;
	ldynak_read_triangles_from_tetmesh("pin25_reduziert_20062018.k", triangles, positions);

	vec3_t bbmin_tool, bbmax_tool;
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);

	geometry_translate(triangles, positions, vec3_t(0., 0., -bbmin_tool.z));	//lift tool to zero level
	geometry_translate(triangles, positions, vec3_t(0., 0., -target_feed ));		//push down to target feed again

	float_t ly_tool = bbmax_tool.y - bbmin_tool.y;
	geometry_translate(triangles, positions, vec3_t(0.,-bbmin_tool.y - ly_tool/2., 0.));		//center tool to y zero

	float_t nudge = 6.4e-4;

	float_t lx_tool = bbmax_tool.x - bbmin_tool.x;
	geometry_translate(triangles, positions, vec3_t(-bbmin_tool.x - lx_tool + nudge, 0., 0.));	//move tool in x direction such that bottom surface almost touches wp

	//tool is at final position, update bbox
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);

	float_t lx = 2*80*1e-6*100;	//full diamond measures
	float_t ly = 2*80*1e-6*100;
	float_t lz = depth_wp;

	int nx = lx/dz;	//equi particle spacing
	int ny = ly/dz;

	int n = nx*ny*nz;
	printf("calculating with %d\n", n);

	int part_iter = 0;
	float4_t* pos = new float4_t[n];
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			for (int k = 0; k < nz; k++) {
				float_t px = i*dz; float_t py = j*dz-ly/2.; float_t pz = - k*dz;
				float4_t cur_pos;

				cur_pos.x = px;
				cur_pos.y = py;
				cur_pos.z = pz;

				pos[part_iter] = cur_pos;

				part_iter++;
			}
		}
	}

	//---------------------------------------------------------------------

	float_t hdx = 1.7;
	phys.E    = 1.1;
	phys.nu   = 0.35;
	phys.rho0 = 4.43;
	phys.G    = phys.E/(2.*(1.+phys.nu));
	phys.K    = 2.0*phys.G*(1+phys.nu)/(3*(1-2*phys.nu));
	phys.mass = dz*dz*dz*phys.rho0;

	bool use_khan = false;

	if (use_khan) {
		joco.A		= 0.0110400;	// Mansur Akbari 2016, Tabelle 1
		joco.B		= 0.0103600;	// Mansur Akbari 2016, Tabelle 1
		joco.C		= 0.0139000;	// Mansur Akbari 2016, Tabelle 1
		joco.m		= 0.7794;		// Mansur Akbari 2016, Tabelle 1
		joco.n		= 0.6349;		// Mansur Akbari 2016, Tabelle 1
		joco.Tref	= 300.;
		joco.Tmelt	= 1723.0000;	// Mansur Akbari 2016, im Text S.65
		joco.eps_dot_ref = 1e-6;	// Mansur Akbari 2016, im Text S.65
		joco.clamp_temp = 1.;
	} else {
		joco.A		= 0.0086200;
		joco.B		= 0.0033100;
		joco.C		= 0.0100000;
		joco.m		= 0.8;
		joco.n		= 0.34;
		joco.Tref	= 300.;
		joco.Tmelt	= 1836.0000;
		joco.eps_dot_ref = 1e-6;
		joco.clamp_temp = 1.;
	}

	trml.cp = 553*1e-8;			// Heat Capacity
	trml.tq = 0.9;				// Taylor-Quinney Coefficient
	trml.k  = 7.1*1e-13;			// Thermal Conduction
	trml.alpha = trml.k/(phys.rho0*trml.cp);	// Thermal diffusivity
	trml.T_init = joco.Tref;

	float_t c0 = sqrt(phys.K/phys.rho0);

	corr.alpha   = 1.;
	corr.beta    = 1.;
	corr.eta     = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3;	//<---!!!!
	{
		float_t h1   = 1./(hdx*dz);
		float_t q    = dz*h1;
		float_t fac  = (M_1_PI)*h1*h1*h1;;
		corr.wdeltap = fac*(1 - 1.5*q*q*(1-0.5*q));
	}

	float4_t *vel   = new float4_t[n];
	float_t  *h     = new float_t[n];
	float_t  *rho   = new float_t[n];
	float_t  *fixed = new float_t[n];
	float_t  *T     = new float_t[n];

	for (int i = 0; i < n; i++) {	// initialize particles
		rho[i]   = phys.rho0;
		h[i]     = hdx*dz;
		vel[i].x =   0.;
		vel[i].y =   0.;
		vel[i].z =   0.;
		fixed[i] =   0;
		T[i]     = joco.Tref;
	}

	unsigned int Nr_BC_particles = 0;	// Counter boundary particles
	for (unsigned int i = 0; i < n; i++) {	// define boundary particles
		//if (pos[i].z < dz/2) {
		if (pos[i].z < -lz+dz/2.0) {
			fixed[i] = 1;
			Nr_BC_particles++;
		}

		if (pos[i].x > lx - dz/2.0) {
			fixed[i] = 1;
			Nr_BC_particles++;
		}
	}

	printf("Number of BC-particles: %u \n", Nr_BC_particles);

	//---------------------------------------------------------------------

//	float_t v_tool = 8e-5;	//speed akbari
	float_t v_tool = 3e-3;	//speed KW
//	float_t v_tool = 570*100/1e6; //breakneck speed

	float_t friction_coeff = 0.35;	//fric ruttiumann
//	float_t friction_coeff = 0.6;	//fric akbari
//	float_t friction_coeff = 0.8;	//fric akbari increased

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, n);
	assert(check_cuda_error());

	mesh_compute_vertex_normals(triangles, positions);
	float_t avg_h = mesh_average_edge_length(triangles);

	auto cpu_tool = new tool_3d(triangles, positions, avg_h);
	global_tool.push_back(new tool_3d_gpu(cpu_tool, vec3_t(v_tool, 0., 0.), phys));

	global_tool[0]->set_algorithm_type(tool_3d_gpu::contact_algorithm::exhaustive);
	global_tool[0]->set_mu(friction_coeff);

	//---------------------------------------------------------------------

	const float_t CFL = 0.3;
	float_t delta_t_max = CFL*hdx*dz/(sqrt(v_tool) + c0);
	global_time_dt = 0.5*delta_t_max;
	global_time_final = lx/v_tool;

	printf("max recommended dt %e, dt used %e\n", delta_t_max, global_time_dt);

	//---------------------------------------------------------------------

	*grid = new grid_gpu_green(n, make_float3_t(-15.0*dz, -15.0*dz-ly/2, -1.5*nz*dz), make_float3_t(lx+15.0*dz, ly+15.0*dz-ly/2, 3.0*bbmax_tool.z), hdx*dz);
	global_blanking = new blanking(vec3_t(-15.0*dz, -15.0*dz-ly/2, -1.5*nz*dz), vec3_t(lx+15.0*dz, ly+15.0*dz-ly/2, 3.0*bbmax_tool.z), 2.0*2.0);

	//---------------------------------------------------------------------

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco);
	actions_setup_thermal_constants_wp(trml);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_workpiece(trml);
	interactions_setup_geometry_constants(*grid);

	global_tool_forces = new tool_forces(global_tool.size());

	return particles;
}

particle_gpu *setup_single_grain_1tool_trml_steady(grid_base **grid) {
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml_wp = make_trml_constants();
	trml_constants trml_tool = make_trml_constants();
	joco_constants joco = make_joco_constants();

	bool sample_tool = false;

	//	float_t v_tool = 8e-5;	//speed akbari
	float_t v_tool = 3e-3;	//speed KW
	float_t friction_coeff = 0.35;	//fric ruttiumann
	//	float_t friction_coeff = 0.6;	//fric akbari
	//	float_t friction_coeff = 0.8;	//fric akbari increased

	float_t target_feed = 20*1e-6*100;	//20 mu in cm
	int depth_fac = 3;
	float_t depth_wp = depth_fac*target_feed;

	float_t num_particles_in_feed = 6;
	int nz = depth_fac*num_particles_in_feed;

	float_t dz = depth_wp/(nz-1);

	std::vector<mesh_triangle> triangles;
	std::vector<vec3_t> positions;

	ldynak_read_triangles_from_tetmesh("pin25_reduziert_20062018.k", triangles, positions);

	vec3_t bbmin_tool, bbmax_tool;
//	geometry_scale(triangles, positions, vec3_t(1./10000.0,1./10000.0,1./10000.0));	//micron to cm
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);
	geometry_translate(triangles, positions, vec3_t(0., 0., -bbmin_tool.z));	//lift tool to zero level
	geometry_translate(triangles, positions, vec3_t(0., 0., -target_feed ));		//push down to target feed again

	float_t ly_tool = bbmax_tool.y - bbmin_tool.y;
	geometry_translate(triangles, positions, vec3_t(0.,-bbmin_tool.y - ly_tool/2., 0.));		//center tool to y zero

	float_t nudge = 6.4e-4;
	float_t lx_tool = bbmax_tool.x - bbmin_tool.x;
	geometry_translate(triangles, positions, vec3_t(-bbmin_tool.x - lx_tool + nudge, 0., 0.));	//move tool in x direction such that bottom surface almost touches wp

	ldynak_dump("translated_tool.k", triangles, positions);

	//tool is at final position, update bbox
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);

	mesh_compute_vertex_normals(triangles, positions);
	float_t avg_h = mesh_average_edge_length(triangles);

	auto cpu_tool = new tool_3d(triangles, positions, avg_h);
	std::vector<vec3_t> samples;
	if (sample_tool)
		samples = cpu_tool->sample(dz);

	float_t lx = 80*80*1e-6*100;	//full diamond measures
	float_t ly = 2*80*1e-6*100;
	float_t lz = depth_wp;

	int nx = lx/dz;	//equi particle spacing
	int ny = ly/dz;

	int n = nx*ny*nz;
	int n_tool = samples.size();
	printf("calculating with %d regular particles\n", n);
	printf("calculating with %d tool particles\n", n_tool);

	int part_iter = 0;
	float4_t* pos = new float4_t[n+n_tool];
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			for (int k = 0; k < nz; k++) {
				float_t px = i*dz; float_t py = j*dz-ly/2.; float_t pz = - k*dz;
				float4_t cur_pos;

				cur_pos.x = px;
				cur_pos.y = py;
				cur_pos.z = pz;

				pos[part_iter] = cur_pos;

				part_iter++;
			}
		}
	}

	for (auto &it: samples) {
		float4_t cur_pos;

		cur_pos.x = it.x;
		cur_pos.y = it.y;
		cur_pos.z = it.z;

		pos[part_iter] = cur_pos;

		part_iter++;
	}

	//---------------------------------------------------------------------

	float_t hdx = 1.7;
	phys.E    = 1.1;
	phys.nu   = 0.35;
	phys.rho0 = 4.43;
	phys.G    = phys.E/(2.*(1.+phys.nu));
	phys.K    = 2.0*phys.G*(1+phys.nu)/(3*(1-2*phys.nu));
	phys.mass = dz*dz*dz*phys.rho0;

	bool use_khan = false;

	if (use_khan) {
		joco.A		= 0.0110400;	// Mansur Akbari 2016, Tabelle 1
		joco.B		= 0.0103600;	// Mansur Akbari 2016, Tabelle 1
		joco.C		= 0.0139000;	// Mansur Akbari 2016, Tabelle 1
		joco.m		= 0.7794;		// Mansur Akbari 2016, Tabelle 1
		joco.n		= 0.6349;		// Mansur Akbari 2016, Tabelle 1
		joco.Tref	= 300.;
		joco.Tmelt	= 1723.0000;	// Mansur Akbari 2016, im Text S.65
		joco.eps_dot_ref = 1e-6;	// Mansur Akbari 2016, im Text S.65
		joco.clamp_temp = 1.;
	} else {
		joco.A		= 0.0086200;
		joco.B		= 0.0033100;
		joco.C		= 0.0100000;
		joco.m		= 0.8;
		joco.n		= 0.34;
		joco.Tref	= 300.;
		joco.Tmelt	= 1836.0000;
		joco.eps_dot_ref = 1e-6;
		joco.clamp_temp = 1.;
	}

	//Thermal properties Ti6Al4v from
	//https://www.azom.com/properties.aspx?ArticleID=1547
	trml_wp.cp = 553*1e-8;			// Heat Capacity
	trml_wp.tq = 0.9;				// Taylor-Quinney Coefficient
	trml_wp.k  = 7.1*1e-13;			// Thermal Conduction
	trml_wp.alpha = trml_wp.k/(phys.rho0*trml_wp.cp);	// Thermal diffusivity
	trml_wp.T_init = joco.Tref;
	trml_wp.eta = 0.9;

	trml_tool.cp = 520*1e-8;	// https://www.engineeringtoolbox.com/specific-heat-solids-d_154.html
	trml_tool.tq = 0.;			// no plastic deformation in diamond
	trml_tool.k = 2200*1e-13;	// https://en.wikipedia.org/wiki/Material_properties_of_diamond
	float_t rho0_tool = 3.5;	// https://en.wikipedia.org/wiki/Diamond
	trml_tool.alpha = trml_tool.k/(rho0_tool*trml_tool.cp);

	float_t c0 = sqrt(phys.K/phys.rho0);

	corr.alpha   = 1.;
	corr.beta    = 1.;
	corr.eta     = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3;
	{
		float_t h1   = 1./(hdx*dz);
		float_t q    = dz*h1;
		float_t fac  = (M_1_PI)*h1*h1*h1;;
		corr.wdeltap = fac*(1 - 1.5*q*q*(1-0.5*q));
	}

	float4_t *vel   = new float4_t[n+n_tool];
	float_t  *h     = new float_t[n+n_tool];
	float_t  *rho   = new float_t[n+n_tool];
	float_t  *fixed = new float_t[n+n_tool];
	float_t  *T     = new float_t[n+n_tool];
	float_t  *tool_p = new float_t[n+n_tool];

	for (int i = 0; i < n+n_tool; i++) {	// initialize particles
		rho[i]   = phys.rho0;
		h[i]     = hdx*dz;
		vel[i].x =   0.;
		vel[i].y =   0.;
		vel[i].z =   0.;
		fixed[i] =   0;
		T[i]     = joco.Tref;

		if (i >= n) {
			tool_p[i] = 1.;
		} else {
			tool_p[i] = 0.;
		}
	}

	unsigned int Nr_BC_particles = 0;	// Counter boundary particles
	{
		vec3_t p1(-0.0082535, -0.00193475, 0.0004759);
		vec3_t p2(-0.0067885, -0.00326105, 0.0024517);
		vec3_t p3(-0.0013051, 0.00405095, 0.0004759);

		vec3_t nrm = glm::cross(p2-p1, p2-p3);

		for (unsigned int i = 0; i < n+n_tool; i++) {	// define boundary particles

			if (pos[i].z < -lz+dz/2.0) {
				fixed[i] = 1;
				Nr_BC_particles++;
			}

			if (tool_p[i] == 1.) {	//thermal sinks at tool bounds

				if (pos[i].z > bbmax_tool.z - 1.5*dz) {
					fixed[i] = 1.;
					Nr_BC_particles++;
				}

				vec3_t qp(pos[i].x, pos[i].y, pos[i].z);
				float_t d = glm::dot(nrm, qp-p1)/glm::length(nrm);

				if (d < 1.5*dz) {
					fixed[i] = 1.;
					Nr_BC_particles++;
				}
			}
		}
	}

	printf("Number of BC-particles: %u \n", Nr_BC_particles);

	//---------------------------------------------------------------------

	global_tool.push_back(new tool_3d_gpu(cpu_tool, vec3_t(v_tool, 0., 0.), phys));

	global_tool[0]->set_algorithm_type(tool_3d_gpu::contact_algorithm::exhaustive);
	global_tool[0]->set_mu(friction_coeff);

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, tool_p, n+n_tool);
	assert(check_cuda_error());

	//---------------------------------------------------------------------

	const float_t CFL = 0.3;
	float_t delta_t_max = CFL*hdx*dz/(sqrt(v_tool) + c0);
	global_time_dt = 0.5*delta_t_max;
	global_time_final = lx/v_tool;

	printf("max recommended dt %e, dt used %e\n", delta_t_max, global_time_dt);

	//---------------------------------------------------------------------

	*grid = new grid_gpu_green(n+n_tool, make_float3_t(bbmin_tool.x -lx_tool , -15.0*dz-ly/2, -1.5*nz*dz), make_float3_t(lx/40.+15.0*dz, ly+15.0*dz-ly/2, 3.0*bbmax_tool.z), hdx*dz);
	(*grid)->set_bbox_vel(make_float3_t(v_tool, 0., 0.));
	global_blanking = new blanking(vec3_t(bbmin_tool.x -lx_tool, -15.0*dz-ly/2, -1.5*nz*dz), vec3_t(lx/40.+15.0*dz, ly+15.0*dz-ly/2, 3.0*bbmax_tool.z),
			vec3_t(v_tool, 0., 0.), 2.0*2.0);

	//---------------------------------------------------------------------

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco);
	actions_setup_thermal_constants_wp(trml_wp);
	actions_setup_thermal_constants_tool(trml_tool);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_workpiece(trml_wp);
	interactions_setup_thermal_constants_tool(trml_tool, global_tool[0]);
	interactions_setup_geometry_constants(*grid);

	global_tool_forces = new tool_forces(global_tool.size());

	return particles;
}



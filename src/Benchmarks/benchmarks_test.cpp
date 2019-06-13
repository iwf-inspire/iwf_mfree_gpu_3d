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

#include "benchmarks_test.h"

particle_gpu *setup_ring_contact(int nbox, grid_base **grid) {
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml = make_trml_constants();
	joco_constants joco = make_joco_constants();

	//problem dimensions (monaghan & gray)
	float_t ri = 0.035;
	float_t ro = 0.04;
	float_t spacing = ro + 0.001;

	float_t dx = 2*ro/(nbox-1);
	float_t hdx = 1.7;

	//if true a rubber like material is used, otherwise steel
	const bool rubber = true;

	if (rubber) {
		phys.E    = 1e7;
		phys.nu   = 0.4;
		phys.rho0 = 1.;
		phys.G    = phys.E/(2.*(1.+phys.nu));
		phys.K    = 2.0*phys.G*(1+phys.nu)/(3*(1-2*phys.nu));
		phys.mass = dx*dx*dx*phys.rho0;
	} else {
		phys.E    = 200e9;
		phys.nu   = 0.29;
		phys.rho0 = 7830.0;
		phys.G    = phys.E/(2.*(1.+phys.nu));
		phys.K    = 2.0*phys.G*(1+phys.nu)/(3*(1-2*phys.nu));
		phys.mass = dx*dx*dx*phys.rho0;

		joco.A		= 792.0e6;
		joco.B		= 510.0e6;
		joco.C		= 0.014;
		joco.m		= 1.03;
		joco.n		= 0.26;
		joco.Tref	= 273.0;
		joco.Tmelt	= 1793.0;
		joco.eps_dot_ref = 1;

		trml.cp = 477;
		trml.tq = .9;
		trml.k  = 50;
		trml.alpha = trml.k/(phys.rho0*trml.cp);
		trml.T_init = joco.Tref;
	}

	corr.alpha   = 1.;
	corr.beta    = 1.;
	corr.eta     = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3;

	{
		float_t h1   = 1./(hdx*dx);
		float_t q    = dx*h1;
		float_t fac  = (M_1_PI)*h1*h1*h1;
		corr.wdeltap = fac*(1 - 1.5*q*q*(1-0.5*q));
	}

	printf("usign stresseps %e\n", corr.stresseps);

	std::vector<float4_t> pos;
	for (int i = 0; i < nbox; i++) {
		for (int j = 0; j < nbox; j++) {
			for (int k = 0; k < nbox; k++) {
				float_t px = -ro+i*dx; float_t py = -ro+j*dx; float_t pz = -ro+k*dx;
				float_t dist = sqrt(px*px + py*py + pz*pz);
				if (dist < ro && dist >= ri) {
					float4_t posl, posr;

					posl.x = px-spacing;
					posl.y = py;
					posl.z = pz;

					pos.push_back(posl);
				}
			}
		}
	}

	int n = pos.size();

	float_t vel_ring = 250.;
	float_t c0 = sqrt(phys.K/phys.rho0);

	float_t delta_t_max = 0.8*hdx*dx/(vel_ring + c0);
	global_time_dt = 0.5*delta_t_max;
	global_time_final = 3*ro/vel_ring;

	*grid = new grid_gpu_green(n, make_float3_t(-.2,-.06,-.06), make_float3_t(+.1,+.06, +.06), hdx*dx);
	global_blanking = new blanking(vec3_t(-.2,-.06,-.06), vec3_t(+.1,+.06, +.06));

	printf("calculating with %d\n", n);

	float4_t *vel   = new float4_t[n];
	float_t  *h     = new float_t[n];
	float_t  *rho   = new float_t[n];
	float_t  *T     = new float_t[n];
	float_t  *fixed = new float_t[n];

	for (int i = 0; i < n; i++) {
		rho[i] = phys.rho0;
		h[i] = hdx*dx;
		vel[i].x = 250.;
		vel[i].y =   0.;
		vel[i].z =   0.;
		T[i] = joco.Tref;
		fixed[i] = 0.;
	}

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	if (!rubber) {
		actions_setup_johnson_cook_constants(joco);
		actions_setup_thermal_constants_wp(trml);
	}

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	if (!rubber) {
		interactions_setup_thermal_constants_workpiece(trml);
	}
	interactions_setup_geometry_constants(*grid);

	float4_t *pos_f = new float4_t[n];
	for (int i = 0; i < n; i++) {
		pos_f[i] = pos[i];
	}
	particle_gpu *particles = new particle_gpu(pos_f, vel, rho, T, h, fixed, n);

	assert(check_cuda_error());

	//-----------------------------------------------------------------

	std::vector<mesh_triangle> triangles;
	std::vector<vec3_t>    positions;
	json_read_triangles("cube_rot.json", triangles, positions);
	geometry_scale_to_unity(triangles, positions, M_PI/4.);

	for (auto it = triangles.begin(); it != triangles.end(); ++it) {
		it->p1.z -= 0.5;
		it->p2.z -= 0.5;
		it->p3.z -= 0.5;

		it->p1.y -= 0.5;
		it->p2.y -= 0.5;
		it->p3.y -= 0.5;
	}

	for (auto it = positions.begin(); it != positions.end(); ++it) {
		it->z -= 0.5;
		it->y -= 0.5;
	}

	for (auto it = triangles.begin(); it != triangles.end(); ++it) {
		it->p1.x*=0.1;
		it->p2.x*=0.1;
		it->p3.x*=0.1;
	}

	for (auto it = positions.begin(); it != positions.end(); ++it) {
		it->x*=0.1;
	}

	{
		FILE *fp = fopen("normals.txt", "w+");
		for (auto &it : triangles) {

			vec3_t center = float_t(1./3.)*(it.p1 + it.p2 + it.p3);
			vec3_t normal = it.normal;

			fprintf(fp, "%f %f %f %f %f %f\n", center.x, center.y, center.z, normal.x, normal.y, normal.z);
		}

		fclose(fp);
	}

	{
		FILE *fp = fopen("pos.txt", "w+");
		for (auto &it : positions) {
			fprintf(fp, "%f %f %f\n", it.x, it.y, it.z);
		}

		fclose(fp);
	}

	mesh_compute_vertex_normals(triangles, positions);
	double avg_h = mesh_average_edge_length(triangles);

	tool_3d *cpu_tool = new tool_3d(triangles, positions, avg_h);
	tool_3d_gpu *gpu_tool = new tool_3d_gpu(cpu_tool, vec3_t(0.), phys);
	gpu_tool->set_algorithm_type(tool_3d_gpu::contact_algorithm::exhaustive);
	global_tool.push_back(gpu_tool);

	return particles;
}

particle_gpu *setup_rings(int nbox, grid_base **grid) {
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();

	//problem dimensions (monaghan & gray)
	float_t ri = 0.035;
	float_t ro = 0.04;
	float_t spacing = ro + 0.001;

	float_t dx = 2*ro/(nbox-1);
	float_t hdx = 1.5;

	phys.E    = 1e7;
	phys.nu   = 0.4;
	phys.rho0 = 1.;
	phys.G    = phys.E/(2.*(1.+phys.nu));
	phys.K    = 2.0*phys.G*(1+phys.nu)/(3*(1-2*phys.nu));
	phys.mass = dx*dx*dx*phys.rho0;

	corr.alpha   = 1.;
	corr.beta    = 1.;
	corr.eta     = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.5;

	{
		float_t h1   = 1./(hdx*dx);
		float_t q    = dx*h1;
		float_t fac  = (M_1_PI)*h1*h1*h1;;
		corr.wdeltap = fac*(1 - 1.5*q*q*(1-0.5*q));
	}

	std::vector<float4_t> pos;
	for (int i = 0; i < nbox; i++) {
		for (int j = 0; j < nbox; j++) {
			for (int k = 0; k < nbox; k++) {
				float_t px = -ro+i*dx; float_t py = -ro+j*dx; float_t pz = -ro+k*dx;
				float_t dist = sqrt(px*px + py*py + pz*pz);
				if (dist < ro && dist >= ri) {
					float4_t posl, posr;

					posl.x = px-spacing;
					posl.y = py;
					posl.z = pz;

					posr.x = px+spacing;
					posr.y = py;
					posr.z = pz;

					pos.push_back(posl);
					pos.push_back(posr);
				}
			}
		}
	}

	int n = pos.size();

	global_time_dt = 1e-7;
	global_time_final = 8e3*global_time_dt;

	*grid = new grid_gpu_green(n, make_float3_t(-.2,-.06,-.06), make_float3_t(+.2,+.06, +.06), hdx*dx);

	printf("calculating with %d\n", n);

	float4_t *vel = new float4_t[n];
	float_t  *h   = new float_t[n];
	float_t  *rho = new float_t[n];

	for (int i = 0; i < n; i++) {
		rho[i] = phys.rho0;
		h[i] = hdx*dx;
		vel[i].x = (pos[i].x < 0) ? 250 : -250;
		vel[i].y = 0.;
		vel[i].z = 0.;
	}

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_geometry_constants(*grid);

	float4_t *pos_f = new float4_t[n];
	for (int i = 0; i < n; i++) {
		pos_f[i] = pos[i];
	}
	particle_gpu *particles = new particle_gpu(pos_f, vel, rho, h, n);

	assert(check_cuda_error());
	return particles;
}

particle_gpu *setup_cylinder_impact(int nbox, grid_base **grid, bool use_art_stress) {
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();

	float_t ri = 0.035;
	float_t ro = 0.040;
	float_t height = (ro-ri);
	float_t spacing = ro + 0.001;

	float_t dx = 2*ro/(nbox-1);
	float_t hdx = 1.7;

	phys.E    = 1e7;
	phys.nu   = 0.4;
	phys.rho0 = 1.;
	phys.G    = phys.E/(2.*(1.+phys.nu));
	phys.K    = 2.0*phys.G*(1+phys.nu)/(3*(1-2*phys.nu));
	phys.mass = dx*dx*dx*phys.rho0;

	corr.alpha   = 1.;
	corr.beta    = 1.;
	corr.eta     = 0.1;
	corr.xspheps = 0.5;

	if (use_art_stress) {
		corr.stresseps = 0.3;
		{
			float_t h1   = 1./(hdx*dx);
			float_t q    = dx*h1;
			float_t fac  = (M_1_PI)*h1*h1*h1;;
			corr.wdeltap = fac*(1 - 1.5*q*q*(1-0.5*q));
		}
	}

	int nheight = height/dx;

	std::vector<float4_t> pos;
	for (int i = 0; i < nbox; i++) {
		for (int j = 0; j < nbox; j++) {
			for (int k = 0; k < nheight; k++) {
				float_t px = -ro+i*dx; float_t py = -ro+j*dx; float_t pz = k*dx;
				float_t dist = sqrt(px*px + py*py);
				if (dist < ro && dist >= ri) {
					float4_t posl, posr;

					posl.x = px-spacing;
					posl.y = py;
					posl.z = pz;

					posr.x = px+spacing;
					posr.y = py;
					posr.z = pz;

					pos.push_back(posl);
					pos.push_back(posr);
				}
			}
		}
	}

	int n = pos.size();

	global_time_dt = 1e-7*0.3;
	float_t vel_cylinders = 280;
	global_time_final = ro/vel_cylinders*3;

	*grid = new grid_gpu_green(n, make_float3_t(-.2,-.06,-.06), make_float3_t(+.2,+.06, +.06), hdx*dx);

	printf("calculating with %d\n", n);

	float4_t *vel = new float4_t[n];
	float_t  *h   = new float_t[n];
	float_t  *rho = new float_t[n];

	for (int i = 0; i < n; i++) {
		rho[i] = phys.rho0;
		h[i] = hdx*dx;
		vel[i].x = (pos[i].x < 0) ? vel_cylinders : -vel_cylinders;
		vel[i].y = 0.;
		vel[i].z = 0.;
	}

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_geometry_constants(*grid);

	float4_t *pos_f = new float4_t[n];
	for (int i = 0; i < n; i++) {
		pos_f[i] = pos[i];
	}
	particle_gpu *particles = new particle_gpu(pos_f, vel, rho, h, n);

	assert(check_cuda_error());
	return particles;
}

particle_gpu *setup_disk(int nbox, grid_base **grid) {
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();

	float_t ri = 0.035;
	float_t ro = 0.040;
	float_t height = (ro-ri);

	float_t dx = 2*ro/(nbox-1);
	float_t hdx = 1.7;

	phys.E    = 1e7;
	phys.nu   = 0.4;
	phys.rho0 = 1.;
	phys.G    = phys.E/(2.*(1.+phys.nu));
	phys.K    = 2.0*phys.G*(1+phys.nu)/(3*(1-2*phys.nu));
	phys.mass = dx*dx*dx*phys.rho0;

	corr.alpha   = 1.;
	corr.beta    = 1.;
	corr.eta     = 0.1;
	corr.xspheps = 0.5;

	int nheight = height/dx;

	std::vector<float4_t> pos;
	for (int i = 0; i < nbox; i++) {
		for (int j = 0; j < nbox; j++) {
			for (int k = 0; k < nheight; k++) {
				float_t px = -ro+i*dx; float_t py = -ro+j*dx; float_t pz = k*dx;
				float_t dist = sqrt(px*px + py*py);
				if (dist < ro && dist >= ri) {
					float4_t pos_i;

					pos_i.x = px;
					pos_i.y = py;
					pos_i.z = pz;

					pos.push_back(pos_i);
				}
			}
		}
	}

	int n = pos.size();

	global_time_dt = 1e-7;
	float_t vel_rot = 50;
	global_time_final = ro/vel_rot*M_PI*2*1.1;	//one rotation + some slack to account for numerical dissipation

	*grid = new grid_gpu_green(n, make_float3_t(-.2,-.06,-.06), make_float3_t(+.2,+.06, +.06), hdx*dx);

	printf("calculating with %d\n", n);

	float4_t *vel = new float4_t[n];
	float_t  *h   = new float_t[n];
	float_t  *rho = new float_t[n];

	for (int i = 0; i < n; i++) {
		rho[i] = phys.rho0;
		h[i] = hdx*dx;
		vel[i].x = (-pos[i].y)/ro*vel_rot;
		vel[i].y = (+pos[i].x)/ro*vel_rot;
		vel[i].z = 0.;
	}

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_geometry_constants(*grid);

	float4_t *pos_f = new float4_t[n];
	for (int i = 0; i < n; i++) {
		pos_f[i] = pos[i];
	}
	particle_gpu *particles = new particle_gpu(pos_f, vel, rho, h, n);

	assert(check_cuda_error());
	return particles;
}

particle_gpu *setup_plastic_ball_plastic_wall_impact(int nbox, grid_base **grid) {
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml = make_trml_constants();
	joco_constants joco = make_joco_constants();

	float_t fac = 15;
	int nx_wall = nbox;
	int ny_wall = fac*nbox;
	int nz_wall = fac*nbox;

	float_t l = 0.1;
	float_t wall_thickness = 1/fac*l;
	float_t r = 0.2*l;

	float_t dx = wall_thickness/(nx_wall-1);
	float_t dy = l/(ny_wall-1);
	float_t dz = l/(nz_wall-1);

	int nx_ball = ceil(2*r/dx);

	float_t hdx = 1.7;
	float_t ddx = fmax(dx,fmax(dy,dz));

	phys.E    = 200e9;
	phys.nu   = 0.29;
	phys.rho0 = 7830.0;
	phys.G    = phys.E/(2.*(1.+phys.nu));
	phys.K    = 2.0*phys.G*(1+phys.nu)/(3*(1-2*phys.nu));
	phys.mass = dx*dx*dx*phys.rho0;

	joco.A		= 792.0e6;
	joco.B		= 510.0e6;
	joco.C		= 0.014;
	joco.m		= 1.03;
	joco.n		= 0.26;
	joco.Tref	= 273.0;
	joco.Tmelt	= 1793.0;
	joco.eps_dot_ref = 1;

	trml.cp = 477;
	trml.tq = .9;
	trml.k  = 50;
	trml.alpha = trml.k/(phys.rho0*trml.cp);
	trml.T_init = joco.Tref;

	corr.alpha   = 1.;
	corr.beta    = 1.;
	corr.eta     = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3;

	vec3_t offset_wall(r, -l/2., -l/2.);

	std::vector<float4_t> vec_pos;

	//wall
	for (int i = 0; i < nx_wall; i++) {
		for (int j = 0; j < ny_wall; j++) {
			for (int k = 0; k < nz_wall; k++) {
				float_t px = i*dx; float_t py = j*dy; float_t pz = k*dz;
				float4_t cur_pos;

				cur_pos.x = px + offset_wall.x;
				cur_pos.y = py + offset_wall.y;
				cur_pos.z = pz + offset_wall.z;

				vec_pos.push_back(cur_pos);
			}
		}
	}

	vec3_t offset_projectile(-3*dx, 0., 0.);
	for (int i = 0; i < nx_ball; i++) {
		for (int j = 0; j < nx_ball; j++) {
			for (int k = 0; k < nx_ball; k++) {
				float_t px = i*dx-r; float_t py = j*dx-r; float_t pz = k*dx-r;
				float_t dist = sqrt(px*px + py*py + pz*pz);
				if (dist <= r) {
					float4_t cur_pos;

					cur_pos.x = px + offset_projectile.x;
					cur_pos.y = py + offset_projectile.y;
					cur_pos.z = pz + offset_projectile.z;

					vec_pos.push_back(cur_pos);
				}
			}
		}
	}

	int n = vec_pos.size();
	float4_t *pos = new float4_t[n];
	std::copy(vec_pos.begin(), vec_pos.end(), pos);

	float4_t *vel   = new float4_t[n];
	float_t  *h     = new float_t[n];
	float_t  *rho   = new float_t[n];
	float_t  *fixed = new float_t[n];
	float_t  *T     = new float_t[n];

	const float_t v_impact = 1000.0;

	for (int i = 0; i < n; i++) {
		rho[i]   = phys.rho0;
		h[i]     = hdx*ddx;
		vel[i].x =   (pos[i].x < 0.) ? v_impact : 0.;
		vel[i].y =   0.;
		vel[i].z =   0.;
		fixed[i] =   0;
		T[i]     = joco.Tref;
	}

	for (unsigned int i = 0; i < n; i++) {
		if (pos[i].z < -l/2+dx/2) {
			fixed[i] = 1;
		}

		if (pos[i].z > +l/2-dx/2) {
			fixed[i] = 1;
		}

		if (pos[i].y < -l/2+dx/2) {
			fixed[i] = 1;
		}

		if (pos[i].y > +l/2-dx/2) {
			fixed[i] = 1;
		}
	}

	printf("calculating with %d\n", n);

	vec3_t bbmin, bbmax;
	geometry_get_bb(pos, n, bbmin, bbmax, 2*r);
	*grid = new grid_gpu_green(n, make_float3_t(bbmin.x, bbmin.y, bbmin.z), make_float3_t(bbmax.x, bbmax.y, bbmax.z), hdx*ddx);
	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, n);

	//-----------------------------------------------------------------

	float_t c0 = sqrt(phys.K/phys.rho0);
	float_t delta_t_max = 0.3*hdx*dz/(v_impact + c0);
	global_time_dt = 0.8*delta_t_max;
	global_time_final = 2*r/v_impact;

	printf("max recommended dt %e, dt used %e\n", delta_t_max, global_time_dt);

	//-----------------------------------------------------------------

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco);
	actions_setup_thermal_constants_wp(trml);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_workpiece(trml);
	interactions_setup_geometry_constants(*grid);

	assert(check_cuda_error());

	return particles;

}

particle_gpu *setup_solid_ball_plastic_wall_impact(int nbox, grid_base **grid) {
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml = make_trml_constants();
	joco_constants joco = make_joco_constants();

	int nx = nbox;
	int ny = 10*nbox;
	int nz = 10*nbox;

	int n = nx*ny*nz;

	double l = 1.;

	double dx = 0.1*l/(nx-1);
	double dy = l/(ny-1);
	double dz = l/(nz-1);

	double hdx = 1.7;
	double ddx = fmax(dx,fmax(dy,dz));

	phys.E    = 200e9;
	phys.nu   = 0.29;
	phys.rho0 = 7830.0;
	phys.G    = phys.E/(2.*(1.+phys.nu));
	phys.K    = 2.0*phys.G*(1+phys.nu)/(3*(1-2*phys.nu));
	phys.mass = dx*dx*dx*phys.rho0;

	joco.A		= 792.0e6;
	joco.B		= 510.0e6;
	joco.C		= 0.014;
	joco.m		= 1.03;
	joco.n		= 0.26;
	joco.Tref	= 273.0;
	joco.Tmelt	= 1793.0;
	joco.eps_dot_ref = 1;

	trml.cp = 477;
	trml.tq = .9;
	trml.k  = 50;
	trml.alpha = trml.k/(phys.rho0*trml.cp);
	trml.T_init = joco.Tref;

	corr.alpha   = 1.;
	corr.beta    = 1.;
	corr.eta     = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.;

	glm::dvec3 offset(0, -l/2., -l/2.);

	int part_iter = 0;
	float4_t* pos = new float4_t[n];
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			for (int k = 0; k < nz; k++) {
				float_t px = i*dx; float_t py = j*dy; float_t pz = k*dz;
				float4_t cur_pos;

				cur_pos.x = px + offset.x;
				cur_pos.y = py + offset.y;
				cur_pos.z = pz + offset.z;

				pos[part_iter] = cur_pos;

				part_iter++;
			}
		}
	}

	printf("calculating with %d\n", n);

	float4_t *vel   = new float4_t[n];
	float_t  *h     = new float_t[n];
	float_t  *rho   = new float_t[n];
	float_t  *fixed = new float_t[n];
	float_t  *T     = new float_t[n];

	for (int i = 0; i < n; i++) {
		rho[i]   = phys.rho0;
		h[i]     = hdx*ddx;
		vel[i].x =   0.;
		vel[i].y =   0.;
		vel[i].z =   0.;
		fixed[i] =   0;
		T[i]     = joco.Tref;
	}

	for (unsigned int i = 0; i < n; i++) {
		if (pos[i].z < -l/2+dx/2) {
			fixed[i] = 1;
		}

		if (pos[i].z > +l/2-dx/2) {
			fixed[i] = 1;
		}

		if (pos[i].y < -l/2+dx/2) {
			fixed[i] = 1;
		}

		if (pos[i].y > +l/2-dx/2) {
			fixed[i] = 1;
		}
	}

	const double v_impact = 500.0;

	std::vector<mesh_triangle> triangles;
	std::vector<vec3_t>    positions;
	json_read_triangles("sphere_coarse.json", triangles, positions);
	geometry_scale_to_unity(triangles, positions, M_PI/4.);

	double rad = 0.125;
	geometry_scale(triangles, positions, vec3_t(2*rad, 2*rad, 2*rad));
	geometry_translate(triangles, positions, vec3_t(-2*rad-0.01*rad, -rad, -rad));
	geometry_print_bb(positions);

	mesh_compute_vertex_normals(triangles, positions);
	double avg_h = mesh_average_edge_length(triangles);

	tool_3d *cpu_tool = new tool_3d(triangles, positions, avg_h);
	global_tool.push_back(new tool_3d_gpu(cpu_tool, vec3_t(v_impact, 0., 0.), phys));

	//-----------------------------------------------------------------

	float_t c0 = sqrt(phys.K/phys.rho0);
	float_t delta_t_max = 0.3*hdx*dz/(v_impact + c0);
	global_time_dt = 0.8*delta_t_max;
	global_time_final = 6*rad/v_impact;

	global_blanking = new blanking(vec3_t(-2*rad+0.02, -l/2 - 0.01, -l/2 - 0.01), vec3_t(l+0.01, l/2 + 0.01, l/2 + 0.01));

	*grid = new grid_gpu_green(n, make_float3_t(-2*rad+0.02, -l/2 - 0.01, -l/2 - 0.01), make_float3_t(l+0.01, l/2 + 0.01, l/2 + 0.01), hdx*ddx);
	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, n);

	//-----------------------------------------------------------------

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco);
	actions_setup_thermal_constants_wp(trml);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_workpiece(trml);
	interactions_setup_geometry_constants(*grid);

	assert(check_cuda_error());

	return particles;
}

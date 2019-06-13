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

#include "actions_gpu.h"

#include "eigen_solver.cuh"
#include "plasticity.cuh"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

static bool m_plastic = false;
static bool m_thermal = false;			//consider thermal conduction in workpiece
static bool m_fric_heat_gen = false;	//consider that friction produces heat

__constant__ static phys_constants physics;
__constant__ static corr_constants correctors;
__constant__ static joco_constants johnson_cook;
__constant__ static trml_constants thermals_wp;
__constant__ static trml_constants thermals_tool;

__device__ __forceinline__ bool isnaninf(float_t val) {
	return isnan(val) || isinf(val);
}

__global__ void do_material_eos(const float_t *__restrict__ blanked, const float_t *__restrict__ in_tool,
		const float_t *__restrict__ rho, float_t *__restrict__ p, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (blanked[pidx] == 1.) return;
	if (in_tool[pidx] == 1.) {
		return;
	}

	float_t rho0 = physics.rho0;
	float_t c0   = sqrtf(physics.K/rho0);
	float_t rhoi = rho[pidx];
	p[pidx] = c0*c0*(rhoi - rho0);
}

__global__ void do_move_tool_particles(particle_gpu particles, vec3_t vel, float_t dt) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N_init) return;

	if (particles.tool_particle[pidx] != 1.) {
		return;
	}

	float4_t pos = particles.pos[pidx];
	vec3_t qp(pos.x, pos.y, pos.z);
	qp += vel*dt;

	particles.pos[pidx] = make_float4_t(qp.x, qp.y, qp.z, 0.);
}

__global__ void do_corrector_artificial_stress(const float_t *__restrict__ blanked, const float_t *__restrict__ in_tool,
		const float_t *__restrict__ rho, const float_t *__restrict__ p, const mat3x3_t *__restrict__ S,
		mat3x3_t *__restrict__ R, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (blanked[pidx] == 1.) return;
	if (in_tool[pidx] == 1.) return;

	float_t eps  = correctors.stresseps;

	if (eps == 0.) return;

	float_t rhoi = rho[pidx];
	float_t pi   = p[pidx];
	mat3x3_t Si  = S[pidx];

	double rhoi21 = 1./(rhoi*rhoi);

	float_t sxx  = Si[0][0] - pi;
	float_t syy  = Si[1][1] - pi;
	float_t szz  = Si[2][2] - pi;

	float_t sxy  = Si[0][1];
	float_t syz  = Si[1][2];
	float_t sxz  = Si[0][2];

	float3_t eigenvals;
	float3_t e1;
	float3_t e2;
	float3_t e3;

	solve_eigen(sxx, sxy, sxz, syy, syz, szz, eigenvals, e1, e2, e3);

	mat3x3_t Rot(e1.x, e2.x, e3.x,
			e1.y, e2.y, e3.y,
			e1.z, e2.z, e3.z);

	mat3x3_t Srot(eigenvals.x, 0., 0.,
			0., eigenvals.y, 0.,
			0., 0., eigenvals.z);

	if (Srot[0][0] > 0) {
		Srot[0][0] = -eps*Srot[0][0]*rhoi21;
	} else {
		Srot[0][0] = 0.;
	}

	if (Srot[1][1] > 0) {
		Srot[1][1] = -eps*Srot[1][1]*rhoi21;
	} else {
		Srot[1][1] = 0.;
	}

	if (Srot[2][2] > 0) {
		Srot[2][2] = -eps*Srot[2][2]*rhoi21;
	} else {
		Srot[2][2] = 0.;
	}

	mat3x3_t art_stress = Rot*Srot*glm::transpose(Rot);

	R[pidx] = art_stress;
}

__global__ void do_material_stress_rate_jaumann(const float_t *__restrict__ blanked, const float_t *__restrict__ in_tool,
		const mat3x3_t *__restrict__ v_der, const mat3x3_t *__restrict__ Stress,
		mat3x3_t *S_t, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (blanked[pidx] == 1.) return;
	if (in_tool[pidx] == 1.) return;

	float_t G = physics.G;

	mat3x3_t vi_der = v_der[pidx];
	mat3x3_t Si     = Stress[pidx];

	float_t vx_x = vi_der[0][0];
	float_t vx_y = vi_der[0][1];
	float_t vx_z = vi_der[0][2];

	float_t vy_x = vi_der[1][0];
	float_t vy_y = vi_der[1][1];
	float_t vy_z = vi_der[1][2];

	float_t vz_x = vi_der[2][0];
	float_t vz_y = vi_der[2][1];
	float_t vz_z = vi_der[2][2];

	float_t Sxx = Si[0][0];
	float_t Sxy = Si[0][1];
	float_t Sxz = Si[0][2];

	float_t Syx = Si[1][0];
	float_t Syy = Si[1][1];
	float_t Syz = Si[1][2];

	float_t Szx = Si[2][0];
	float_t Szy = Si[2][1];
	float_t Szz = Si[2][2];

	const mat3x3_t epsdot = mat3x3_t(	vx_x,				0.5*(vx_y + vy_x),	0.5*(vx_z + vz_x),
			                            0.5*(vx_y + vy_x),	vy_y,				0.5*(vy_z + vz_y),
			                            0.5*(vx_z + vz_x),	0.5*(vy_z + vz_y),	vz_z);

	const mat3x3_t omega  = mat3x3_t(	0.,					0.5*(vy_x - vx_y),	0.5*(vz_x - vx_z),
			                            0.5*(vx_y - vy_x),	0.,					0.5*(vz_y - vy_z),
			                            0.5*(vx_z - vz_x),	0.5*(vy_z - vz_y),	0.);

	const mat3x3_t S      = mat3x3_t(	Sxx, Sxy, Sxz,
										Sxy, Syy, Syz,
										Sxz, Syz, Szz);

	const mat3x3_t I = mat3x3_t(1.);

	float_t trace_epsdot = epsdot[0][0] + epsdot[1][1] + epsdot[2][2];

	mat3x3_t Si_t = float_t(2.)*G*(epsdot - float_t(1./3.)*trace_epsdot*I) + omega*S + S*glm::transpose(omega);	//Belytschko (3.7.9)

	S_t[pidx] = Si_t;
}

__global__ void do_material_fric_heat_gen(const float_t *__restrict__ blanked, const float_t *__restrict__ in_tool,
		const float4_t * __restrict__ vel, const float3_t * __restrict__ f_fric, const float3_t * __restrict__ n, float_t *__restrict__ T, float3_t vel_tool, float_t dt, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (blanked[pidx] == 1.) return;
	if (in_tool[pidx] == 1.) return;

	const float_t eta = thermals_wp.eta;

	//compute F_fric_mag;
	float3_t f_T =  f_fric[pidx];
	float_t  f_fric_mag = sqrtf(f_T.x*f_T.x + f_T.y*f_T.y + f_T.z*f_T.z);

	if (f_fric_mag == 0.) {
		return;
	}

	//compute v_rel
	float3_t normal     = n[pidx];
	float4_t v_particle = vel[pidx];
	float3_t v_diff     = make_float3_t(v_particle.x-vel_tool.x, v_particle.y-vel_tool.y, v_particle.z-vel_tool.z);

	float_t  v_diff_dot_normal = v_diff.x*normal.x + v_diff.y*normal.y + v_diff.z*normal.z;
	float3_t v_relative = make_float3_t(v_diff.x -  v_diff_dot_normal, v_diff.y - v_diff_dot_normal, v_diff.z - v_diff_dot_normal);

	float_t  v_rel_mag  = sqrtf(v_relative.x*v_relative.x + v_relative.y*v_relative.y + v_relative.z*v_relative.z);

	T[pidx] += eta*dt*f_fric_mag*v_rel_mag/(thermals_wp.cp*physics.mass);
}

__global__ void do_contmech_continuity(const float_t *__restrict__ blanked, const float_t *__restrict__ in_tool,
		const float_t *__restrict__ rho, const mat3x3_t *__restrict__ v_der,
		float_t *__restrict__ rho_t, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (blanked[pidx] == 1.) return;
	if (in_tool[pidx] == 1.) return;

	double   rhoi   = rho[pidx];
	mat3x3_t vi_der = v_der[pidx];

	float_t vx_x = vi_der[0][0];
	float_t vy_y = vi_der[1][1];
	float_t vz_z = vi_der[2][2];

	rho_t[pidx] = -rhoi*(vx_x + vy_y + vz_z);
}

__global__ void do_contmech_momentum(const float_t *__restrict__ blanked, const float_t *__restrict__ in_tool,
		const mat3x3_t *__restrict__ S_der, const float3_t *__restrict__ fc, const float3_t *__restrict__ ft,
		float3_t *__restrict__ vel_t, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (blanked[pidx] == 1.) return;
	if (in_tool[pidx] == 1.) return;

	float_t mass = physics.mass;

	mat3x3_t Si_der = S_der[pidx];
	float3_t fci    = fc[pidx];
	float3_t fti    = ft[pidx];
	float3_t veli_t = vel_t[pidx];

	float_t Sxx_x = Si_der[0][0];
	float_t Sxy_y = Si_der[0][1];
	float_t Sxz_z = Si_der[0][2];

	float_t Syx_x = Si_der[1][0];
	float_t Syy_y = Si_der[1][1];
	float_t Syz_z = Si_der[1][2];

	float_t Szx_x = Si_der[2][0];
	float_t Szy_y = Si_der[2][1];
	float_t Szz_z = Si_der[2][2];

	float_t fcx   = fci.x;
	float_t fcy   = fci.y;
	float_t fcz   = fci.z;

	float_t ftx   = fti.x;
	float_t fty   = fti.y;
	float_t ftz   = fti.z;

	//redundant mult and div by rho elimnated in der stress
	veli_t.x += Sxx_x + Sxy_y + Sxz_z + fcx / mass + ftx / mass;
	veli_t.y += Syx_x + Syy_y + Syz_z + fcy / mass + fty / mass;
	veli_t.z += Szx_x + Szy_y + Szz_z + fcz / mass + ftz / mass;

	vel_t[pidx] = veli_t;
}

__global__ void do_contmech_advection(const float_t *__restrict__ blanked, const float_t *__restrict__ in_tool,
		const float4_t *__restrict__ vel,
		float3_t *__restrict__ pos_t, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (blanked[pidx] == 1.) return;
	if (in_tool[pidx] == 1.) return;

	float4_t veli   = vel[pidx];
	float3_t posi_t = pos_t[pidx];

	float3_t posi_t_new;
	posi_t_new.x = posi_t.x + veli.x;
	posi_t_new.y = posi_t.y + veli.y;
	posi_t_new.z = posi_t.z + veli.z;

	pos_t[pidx] = posi_t_new;
}

__global__ void do_invalidate_rate(particle_gpu particles) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;

	bool invalid = false;

	invalid = invalid || isnaninf(particles.pos_t[pidx].x);
	invalid = invalid || isnaninf(particles.pos_t[pidx].y);
	invalid = invalid || isnaninf(particles.pos_t[pidx].z);

	invalid = invalid || isnaninf(particles.vel_t[pidx].x);
	invalid = invalid || isnaninf(particles.vel_t[pidx].y);
	invalid = invalid || isnaninf(particles.vel_t[pidx].z);

	invalid = invalid || isnaninf(particles.S_t[pidx][0][0]);
	invalid = invalid || isnaninf(particles.S_t[pidx][1][1]);
	invalid = invalid || isnaninf(particles.S_t[pidx][2][2]);

	invalid = invalid || isnaninf(particles.S_t[pidx][0][1]);
	invalid = invalid || isnaninf(particles.S_t[pidx][1][2]);
	invalid = invalid || isnaninf(particles.S_t[pidx][2][0]);

	invalid = invalid || isnaninf(particles.rho_t[pidx]);
	invalid = invalid || isnaninf(particles.T_t[pidx]);

	if (invalid) {
		particles.blanked[pidx] = 1.;
		printf("invalidated particle %d due to nan!\n", pidx);
	}
}

__global__ void do_check_valid_full(particle_gpu particles) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;

	if (particles.blanked[pidx] == 1.) {
		return;
	}

	bool invalid = false;

	invalid = invalid || isnaninf(particles.pos[pidx].x);
	invalid = invalid || isnaninf(particles.pos[pidx].y);
	invalid = invalid || isnaninf(particles.pos[pidx].z);

	invalid = invalid || isnaninf(particles.vel[pidx].x);
	invalid = invalid || isnaninf(particles.vel[pidx].y);
	invalid = invalid || isnaninf(particles.vel[pidx].z);

	invalid = invalid || isnaninf(particles.S[pidx][0][0]);
	invalid = invalid || isnaninf(particles.S[pidx][1][1]);
	invalid = invalid || isnaninf(particles.S[pidx][2][2]);

	invalid = invalid || isnaninf(particles.S[pidx][0][2]);
	invalid = invalid || isnaninf(particles.S[pidx][0][1]);
	invalid = invalid || isnaninf(particles.S[pidx][1][2]);

	invalid = invalid || isnaninf(particles.rho[pidx]);
	invalid = invalid || isnaninf(particles.T[pidx]);

	invalid = invalid || isnaninf(particles.eps_pl[pidx]);
	invalid = invalid || isnaninf(particles.eps_pl_dot[pidx]);
	invalid = invalid || isnaninf(particles.p[pidx]);

	if (invalid) {
		printf("found particle with nan values that is not blanked!\n");
	}
}

__global__ void do_plasticity_johnson_cook(particle_gpu particles, float_t dt) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;
	if (particles.blanked[pidx] == 1.) return;
	if (particles.tool_particle[pidx] == 1.) return;

	float_t mu = physics.G;

	mat3x3_t S = particles.S[pidx];
	float_t Strialxx = S[0][0];
	float_t Strialyy = S[1][1];
	float_t Strialzz = S[2][2];
	float_t Strialxy = S[0][1];
	float_t Strialyz = S[1][2];
	float_t Strialzx = S[2][0];

	float_t norm_Strial = sqrt_t(Strialxx*Strialxx + Strialyy*Strialyy + Strialzz*Strialzz + 2*(Strialxy*Strialxy + Strialyz*Strialyz + Strialzx*Strialzx));

	float_t eps_pl     = particles.eps_pl[pidx];
	float_t eps_pl_dot = particles.eps_pl_dot[pidx];
	float_t T          = particles.T[pidx];

	if (johnson_cook.clamp_temp) {
		T = (T > johnson_cook.Tmelt) ? johnson_cook.Tmelt - 1 : T;
	} else {
		if (T > johnson_cook.Tmelt) {
			printf("Particle melted!\n");
		}
	}

	float_t svm = sqrt_t(3.0/2.0)*norm_Strial;

	float_t sigma_Y = sigma_yield(johnson_cook, eps_pl, eps_pl_dot, T);

	// elastic case
	if (svm < sigma_Y) {
		particles.eps_pl_dot[pidx] = 0.;
		return;
	}

	bool failed = false;
	float_t delta_lambda = solve_secant(johnson_cook, fmax(eps_pl_dot*dt*sqrt(2./3.), 1e-8), 1e-6,
			norm_Strial, eps_pl, T, dt, physics.G, failed);

	if (failed) {
		printf("%d: %f %f %f: eps_pl %f, eps_pl_dot %f, T %f\n", pidx, particles.pos[pidx].x, particles.pos[pidx].y, particles.pos[pidx].z,
				particles.eps_pl[pidx], particles.eps_pl_dot[pidx], particles.T[pidx]);
	}

	float_t eps_pl_new = eps_pl + sqrtf(2.0/3.0) * fmaxf(delta_lambda,0.);
	float_t eps_pl_dot_new = sqrtf(2.0/3.0) *  fmaxf(delta_lambda,0.) / dt;

	particles.eps_pl[pidx] = eps_pl_new;
	particles.eps_pl_dot[pidx] = eps_pl_dot_new;

	mat3x3_t S_new;
	S_new[0][0] = Strialxx - Strialxx/norm_Strial*delta_lambda*2.*mu;
	S_new[1][1] = Strialyy - Strialyy/norm_Strial*delta_lambda*2.*mu;
	S_new[2][2] = Strialzz - Strialzz/norm_Strial*delta_lambda*2.*mu;

	S_new[0][1] = Strialxy - Strialxy/norm_Strial*delta_lambda*2.*mu;
	S_new[1][0] = Strialxy - Strialxy/norm_Strial*delta_lambda*2.*mu;

	S_new[1][2] = Strialyz - Strialyz/norm_Strial*delta_lambda*2.*mu;
	S_new[2][1] = Strialyz - Strialyz/norm_Strial*delta_lambda*2.*mu;

	S_new[2][0] = Strialzx - Strialzx/norm_Strial*delta_lambda*2.*mu;
	S_new[0][2] = Strialzx - Strialzx/norm_Strial*delta_lambda*2.*mu;

	particles.S[pidx] = S_new;

	//plastic work to heat
	if (thermals_wp.tq != 0.) {
		float_t delta_eps_pl = eps_pl_new - eps_pl;
		float_t sigma_Y = sigma_yield(johnson_cook, eps_pl_new, eps_pl_dot_new, T);
		float_t rho = particles.rho[pidx];
		particles.T[pidx] += thermals_wp.tq/(thermals_wp.cp*rho)*delta_eps_pl*sigma_Y;
	}
}

__global__ void do_boundary_conditions_thermal(particle_gpu particles) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;

	if (particles.fixed[pidx] == 1.) {
		particles.T[pidx] = thermals_wp.T_init;

		if (thermals_wp.T_init == 0.) {
			printf("WARNING: set temp to zero due to boundary condition!\n");
		}
	}
}

__global__ void do_boundary_conditions(particle_gpu particles) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;
	if (particles.blanked[pidx] == 1.) return;
	if (particles.tool_particle[pidx] == 1.) return;

	if (particles.fixed[pidx]) {
		particles.vel[pidx].x = particles.vel_bc[pidx].x;
		particles.vel[pidx].y = particles.vel_bc[pidx].y;
		particles.vel[pidx].z = particles.vel_bc[pidx].z;

		particles.fc[pidx].x = 0.;
		particles.fc[pidx].y = 0.;
		particles.fc[pidx].z = 0.;

		particles.pos_t[pidx].x = 0.;
		particles.pos_t[pidx].y = 0.;
		particles.pos_t[pidx].z = 0.;

		particles.vel_t[pidx].x = 0.;
		particles.vel_t[pidx].y = 0.;
		particles.vel_t[pidx].z = 0.;
	}
}

__global__ void do_blanking(particle_gpu particles, float_t vel_max_squared, vec3_t bbmin, vec3_t bbmax) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N_init) return;

	float4_t pos = particles.pos[pidx];
	float4_t vel = particles.vel[pidx];

	bool too_fast = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z >= vel_max_squared;

	bool in_x = pos.x >= bbmin.x && pos.x <= bbmax.x;
	bool in_y = pos.y >= bbmin.y && pos.y <= bbmax.y;
	bool in_z = pos.z >= bbmin.z && pos.z <= bbmax.z;

	if (too_fast || !in_x || !in_y || !in_z) {
		particles.blanked[pidx] = 1.;
	} else {

		bool invalid = false;

		invalid = invalid || isnaninf(particles.pos_t[pidx].x);
		invalid = invalid || isnaninf(particles.pos_t[pidx].y);
		invalid = invalid || isnaninf(particles.pos_t[pidx].z);

		invalid = invalid || isnaninf(particles.vel_t[pidx].x);
		invalid = invalid || isnaninf(particles.vel_t[pidx].y);
		invalid = invalid || isnaninf(particles.vel_t[pidx].z);

		invalid = invalid || isnaninf(particles.S_t[pidx][0][0]);
		invalid = invalid || isnaninf(particles.S_t[pidx][1][1]);
		invalid = invalid || isnaninf(particles.S_t[pidx][2][2]);

		invalid = invalid || isnaninf(particles.S_t[pidx][0][1]);
		invalid = invalid || isnaninf(particles.S_t[pidx][1][2]);
		invalid = invalid || isnaninf(particles.S_t[pidx][2][0]);

		invalid = invalid || isnaninf(particles.rho_t[pidx]);
		invalid = invalid || isnaninf(particles.T_t[pidx]);

		if (!invalid) {		//never, ever, unblank a particle with nan or inf values
			particles.blanked[pidx] = 0.;
		}
	}
}

//---------------------------------------------------------------------

// float2 + struct
struct add_float4 {
    __device__ float4_t operator()(const float4_t& a, const float4_t& b) const {
        float4_t r;
        r.x = a.x + b.x;
        r.y = a.y + b.y;
        r.z = a.z + b.z;
        return r;
    }
 };

void material_eos(particle_gpu *particles) {
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size-1) / block_size);
	dim3 dB(block_size);
	do_material_eos<<<dG,dB>>>(particles->blanked, particles->tool_particle, particles->rho, particles->p, particles->N);
	check_cuda_error("After material_eos\n");
}

void corrector_artificial_stress(particle_gpu *particles) {
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size-1) / block_size);
	dim3 dB(block_size);
	do_corrector_artificial_stress<<<dG,dB>>>(particles->blanked, particles->tool_particle, particles->rho, particles->p, particles->S, particles->R, particles->N);
	check_cuda_error("After Corrector Artifical Stress\n");
}

void material_stress_rate_jaumann(particle_gpu *particles) {
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size-1) / block_size);
	dim3 dB(block_size);
	do_material_stress_rate_jaumann<<<dG,dB>>>(particles->blanked, particles->tool_particle, particles->v_der, particles->S, particles->S_t, particles->N);
	check_cuda_error("After material_stress_rate_jaumann\n");
}

void contmech_continuity(particle_gpu *particles) {
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size-1) / block_size);
	dim3 dB(block_size);
	do_contmech_continuity<<<dG,dB>>>(particles->blanked, particles->tool_particle, particles->rho, particles->v_der, particles->rho_t, particles->N);
	check_cuda_error("After contmech_continuity\n");
}

void contmech_momentum(particle_gpu *particles) {
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size-1) / block_size);
	dim3 dB(block_size);
	do_contmech_momentum<<<dG,dB>>>(particles->blanked, particles->tool_particle, particles->S_der, particles->fc, particles->ft, particles->vel_t, particles->N);
	check_cuda_error("After contmech_momentum\n");
}

void contmech_advection(particle_gpu *particles) {
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size-1) / block_size);
	dim3 dB(block_size);
	do_contmech_advection<<<dG,dB>>>(particles->blanked, particles->tool_particle, particles->vel, particles->pos_t, particles->N);
	check_cuda_error("After contmech_advection\n");
}

void plasticity_johnson_cook(particle_gpu *particles) {
	if (!m_plastic) return;
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size-1) / block_size);
	dim3 dB(block_size);
	do_plasticity_johnson_cook<<<dG,dB>>>(*particles, global_time_dt);
	check_cuda_error("After johnson_cook\n");
}

void perform_boundary_conditions_thermal(particle_gpu *particles) {
	if (!m_thermal) return;
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size-1) / block_size);
	dim3 dB(block_size);
	do_boundary_conditions_thermal<<<dG,dB>>>(*particles);
	check_cuda_error("After boundary_conditions_thermal\n");
}

void perform_boundary_conditions(particle_gpu *particles) {
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size-1) / block_size);
	dim3 dB(block_size);
	do_boundary_conditions<<<dG,dB>>>(*particles);
	check_cuda_error("After boundary_conditions\n");
}

void actions_move_tool_particles(particle_gpu *particles, tool_3d_gpu *tool) {
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N_init + block_size-1) / block_size);
	dim3 dB(block_size);

	vec3_t vel = tool->get_vel();

	do_move_tool_particles<<<dG,dB>>>(*particles, vel, global_time_dt);
	cudaThreadSynchronize();
}

void perform_blanking(particle_gpu *particles, blanking *global_blanking) {
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N_init + block_size-1) / block_size);
	dim3 dB(block_size);

	vec3_t bbmin, bbmax;
	global_blanking->get_bb(bbmin, bbmax);
	do_blanking<<<dG,dB>>>(*particles, global_blanking->get_max_vel_squared(), bbmin, bbmax);
	check_cuda_error("After blanking\n");
}

void perform_blanking_dbg(particle_gpu *particles, blanking *global_blanking) {

	static float_t *h_blanking = 0;

	if (h_blanking == 0) {
		h_blanking = new float_t[particles->N_init];
	}

	for (int i = 0; i < particles->N_init; i++)  {
		float_t rand_num = rand()/((float_t) RAND_MAX);
		h_blanking[i] = (rand_num > 0.5) ? 1 : 0;
	}

	cudaMemcpy(particles->blanked, h_blanking, sizeof(float_t)*particles->N_init, cudaMemcpyHostToDevice);
	check_cuda_error("after blanking dbg!\n");
}

void actions_setup_physical_constants(phys_constants physics_h) {
	if (physics_h.mass == 0 || isnan(physics_h.mass)) {
		printf("WARNING: invalid mass set!\n");
	}

	cudaMemcpyToSymbol(physics, &physics_h, sizeof(phys_constants), 0, cudaMemcpyHostToDevice);
}

void actions_setup_corrector_constants(corr_constants correctors_h) {

	if (correctors_h.stresseps > 0.) {
		printf("using artificial stresses with eps %f\n", correctors_h.stresseps);
	}

	cudaMemcpyToSymbol(correctors, &correctors_h, sizeof(corr_constants), 0, cudaMemcpyHostToDevice);
}

void actions_setup_johnson_cook_constants(joco_constants johnson_cook_h) {
	cudaMemcpyToSymbol(johnson_cook, &johnson_cook_h, sizeof(joco_constants), 0, cudaMemcpyHostToDevice);
	m_plastic = true;
}

void actions_setup_thermal_constants_wp(trml_constants thermal_h) {
	cudaMemcpyToSymbol(thermals_wp, &thermal_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);
	if (thermal_h.tq != 0.) {
		printf("considering generation of heat due to plastic work\n");
	}

	if (thermal_h.eta != 0.) {
		printf("considering that friction generates heat\n");
		m_fric_heat_gen = true;
	}

	if (thermal_h.alpha != 0.) {
		m_thermal = true;
#if !(defined(Thermal_Conduction_Brookshaw) ||defined(Thermal_Conduction_Brookshaw))
		printf("warning! heat conduction constants set but no heat conduction algorithm active!");
#endif
	}
}

void actions_setup_thermal_constants_tool(trml_constants thermal_h) {
	cudaMemcpyToSymbol(thermals_tool, &thermal_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);

	if (thermal_h.alpha != 0.) {
		m_thermal = true;

#if !(defined(Thermal_Conduction_Brookshaw) ||defined(Thermal_Conduction_Brookshaw))
		printf("warning! heat conduction constants set but no heat conduction algorithm active!");
#endif
	}
}

void material_fric_heat_gen(particle_gpu *particles, vec3_t vel) {
	if (!m_fric_heat_gen) {
		return;
	}

	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size-1) / block_size);
	dim3 dB(block_size);
	do_material_fric_heat_gen<<<dG,dB>>>(particles->blanked, particles->tool_particle, particles->vel, particles->ft, particles->n,
			particles->T, make_float3_t(vel.x, vel.y, vel.z), global_time_dt, particles->N);
	check_cuda_error("After material_fric_heat_gen\n");
}

void debug_check_valid(particle_gpu *particles) {
	thrust::device_ptr<float4_t> t_pos(particles->pos);
	float4_t ini;
	ini.x = 0.;
	ini.y = 0.;
	ini.z = 0.;
	ini = thrust::reduce(t_pos, t_pos + particles->N, ini, add_float4());

	if (isnan(ini.x) || isnan(ini.y) || isnan(ini.z)) {
		printf("nan found!\n");
		exit(-1);
	}
}

void debug_check_valid_full(particle_gpu *particles) {
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size-1) / block_size);
	dim3 dB(block_size);

	do_check_valid_full<<<dG,dB>>>(*particles);
}

void debug_invalidate(particle_gpu *particles) {
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size-1) / block_size);
	dim3 dB(block_size);

	do_invalidate_rate<<<dG,dB>>>(*particles);
	cudaThreadSynchronize();
}

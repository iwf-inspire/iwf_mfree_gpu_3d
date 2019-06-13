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

//this file hold both the class for the representation of the tool on the gpu as well as the recording of the tool _forces_ on the gpu
//tool foces
//	- quite straight forward. hold num_tool floats which can be written to atomically
//	- can write to a file if required
//tool_3d
//  - a tool_3d_cpu needs to be constructed first to construct a tool on the gpu
//	- holds triangles defining the tool on gpu. a copy is held on cpu for output purposes only
//	- main purpose of the tool class is to enable inside/outside/distance to queries for particles
//	- there are two algorithms for this purpose, either an exhaustive one (i.e. simply checking all triangles) or one using spatial hashing
//  - for small meshes it is beneficial to simply check all triangles
//	- the spatial hashing algorithm uses the ``boxed'' triangles constructed by grid_cpu_tri.*, which are copied to the gpu
//	- both algorithms come in two flavors, safe and un-safe versions. The ``safe'' algorithms use a ray casting algorithm to determine
//    if a point is in- or outside. The unsafe versions simple check the cross product with the nearest normal
//  - the un-safe versions should be safe in almost all cases, except for tools with large bounding boxes which are thinly populated
//    (i.e. particles far away from the actual surface need to be tested)

#ifndef TOOL_3D_GPU_H_
#define TOOL_3D_GPU_H_

#include "constants_structs.h"
#include "surface_tri_mesh.h"
#include "types.h"
#include "particle_gpu.h"

extern float_t global_time_dt;
extern float_t global_time_final;
extern float_t global_time_current;
extern int global_time_step;

class tool_3d;

//---------------------------------------------------------------

//NOTE: keep this single prec, double atomics are only available on telsa
class tool_forces {
private:
	unsigned int m_num_tool = 0;
	float3 *m_forces = 0;	//forces on device
	FILE *m_fp = 0;
	bool m_verbose = true;

public:
	float3 *get_forces_device() const; //returns raw pointer on device
	const std::vector<float3> get_forces_host() const;

	void reset();
	void report(unsigned int step) const;

	tool_forces(unsigned int num_tool);
	~tool_forces();
};

extern tool_forces *global_tool_forces;

//---------------------------------------------------------------

class tool_3d_gpu {
public:

	enum contact_algorithm {
		exhaustive, spatial_hashing
	};

	// move tool with it's velocity
	void update_tool();

	// set velocity of tool
	void set_vel(vec3_t vel);

	// set initial position before analysis start
	void set_initial_tool_position(vec3_t pos_init);

	// set friction coefficient
	void set_mu(float_t mu);

	// set contact stiffness
	void set_contact_alpha(float_t alpha);

	// get velocity of tool
	vec3_t get_vel();

	//get current bounding box
	void get_bbox(vec3_t &min, vec3_t &max) const;

	//get cpu triangles for plotting
	const std::vector<mesh_triangle> &get_cpu_tris() const;

	//get cpu triangles for plotting
	const std::vector<vec3_t> &get_cpu_pos() const;

	//is the tool thermal?
	void set_thermal(bool is_thermal);
	void set_physics(phys_constants phys);
	bool thermal() const;

	//return friction coefficient
	float_t mu() const;

	//is current time between birth and death of the tool
    bool is_alive() const;

    //compute contact force using contact force approach by nianfei with friction force approach from lsdyna user manual
	void compute_contact_force(particle_gpu *particles, bool record_force = false);

	//sample the tool with particles
	void sample(const tool_3d* tool, std::vector<float4_t> &samples, float_t dx);

	// use slow or fast contact search?
	void set_algorithm_type(tool_3d_gpu::contact_algorithm algo_type);

	// tool can be deactivated depending on current time (i.e. the tool is only active between birth and death)
	void set_birth_death(float_t birth, float_t death);

	tool_3d_gpu(const tool_3d* tool, vec3_t vel,
			phys_constants phys, float_t alpha = 0.1);

	tool_3d_gpu(const tool_3d* tool, float_t alpha = 0.1);

	tool_3d_gpu();

private:
	tool_3d_gpu::contact_algorithm m_algo_type = tool_3d_gpu::contact_algorithm::spatial_hashing;

	// cells and triangles on gpu
	int *m_cells = 0;
	mesh_triangle_gpu *m_boxed_triangles = 0;
	mesh_triangle_gpu *m_triangles = 0;

	//everything below this line is stored on cpu
	//----------------------------------------------------------

	// counts
	int m_num_cell = 0;
	int m_num_boxed_tris = 0;
	int m_num_tris = 0;

	//tool velocity
	vec3_t m_vel;

	//birth & death
	float_t m_birth = 0.;
	float_t m_death = FLT_MAX;

	//geometry constants for bounding box (early rejection in contact)
	geom_constants m_geometry_constants;

	bool m_is_thermal = false;

	//friction
	float_t m_mu = 0.;

	//needed for output only
	std::vector<mesh_triangle> m_cpu_mesh;
	std::vector<vec3_t> m_cpu_pos;
	std::vector<vec3_t> m_cpu_pos_init;

	//unique identifier
	unsigned int idx;
};

#endif /* TOOL_3D_GPU_H_ */

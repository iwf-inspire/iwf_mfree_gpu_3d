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

// (abstract) base class for the spatial hashing structure for particles
//  - the grid has three responsibilities
//		- compute hashes and re-order the particles. this is simple and done with a kernel call and thrust call to actually sort the hashes
//      - derive the cell lists from those hashes. This is not trivial since the serial algorithm is a linear scan (i.e. walk along the hashes
//        array and look for the memory locations where the hashes differ).
//      - actually reorder the complete particle attributes
//	- the derived classes implement the latter two aspects differently
//	- there are currently two implementations of this class with ideas be green and rothlin
//  - greens class is faster, the one by rothlin conserves memory
//  - two modes, so to speak, are implemented:
//		- the geometry (bounding box etc) is updated in each step to a tight bounding box around the current particle positions
//      - the geometry is locked. it is never updated and particles escaping the bounding box are assigned an invalid hash
//        (and consequently removed from the simulation)
//  - the second option is quite a bit faster, since it saves some costly thrust reductions
//  - a locked bounding box may also moved, i.e. along with the global blanking structure to achieve an SPH_BOX like behvariour
//    inspired by LSDYNA

#ifndef GRID_H_
#define GRID_H_

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#include <stdio.h>

#include "particle_gpu.h"
#include "tool_3d.h"
#include "tool_3d_gpu.h"
#include "types.h"
#include "blanking.h"

extern blanking *global_blanking;
extern float_t global_time_dt;

class grid_base {

protected:
	//geometry
	float_t m_dx = 0.;
	float_t m_lx = 0., m_ly = 0., m_lz = 0.;
	long int m_nx = 0, m_ny = 0, m_nz = 0;
	long int m_num_cell = 0;

	float_t m_bbmin_x = 0., m_bbmax_x = 0.;
	float_t m_bbmin_y = 0., m_bbmax_y = 0.;
	float_t m_bbmin_z = 0., m_bbmax_z = 0.;

	float_t m_bbmin_x_init = 0., m_bbmax_x_init = 0.;
	float_t m_bbmin_y_init = 0., m_bbmax_y_init = 0.;
	float_t m_bbmin_z_init = 0., m_bbmax_z_init = 0.;

	float3_t m_vel_bbox;
	bool m_min_fixed = false;
	bool m_max_fixed = false;

	long int m_max_cell = 0;
	int m_num_part = 0;

	bool m_geometry_locked = false;
	bool m_hard_blank = true;

	void set_geometry(float3_t bbmin, float3_t bbmax, float_t h);

public:
	long int nx() const;
	long int ny() const;
	long int nz() const;

	float_t bbmin_x() const;
	float_t bbmin_y() const;
	float_t bbmin_z() const;
	float_t bbmax_x() const;
	float_t bbmax_y() const;
	float_t bbmax_z() const;
	float_t dx() const;
	bool is_locked() const;
	long int num_cell() const;

	void assign_hashes(particle_gpu *partilces) const;
	void update_geometry(particle_gpu *particles, float_t kernel_width = 2.0);

	virtual void sort(particle_gpu *particles) const = 0;
	virtual void get_cells(particle_gpu *particles, int *cell_start, int *cell_end) = 0;

	void set_bbox_vel(float3_t bbox_vel);
	void set_max_fixed(bool fixed);
	void set_min_fixed(bool fixed);
	void set_hard_blank(bool hard_blank);

	void adapt_particle_number(particle_gpu *particles) const;

	grid_base(int max_cell, int num_part);

	//locks geometry! calls to update geometry will NOT work, i.e. not adapt to the bbox spanned by particles
	//bbox velocity is respected
	grid_base(int num_part, float3_t bbmin, float3_t bbmax, float_t h);
	virtual ~grid_base();
};

#endif /* GRID_H_ */

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

#ifndef BLANKING_H_
#define BLANKING_H_

#include "tool_3d_gpu.h"
#include "types.h"

//class for global, consistent handling of blanking
//	blanking may be done by bounding box or max velocity (locally diverging sim)
//	the bounding box may move (translate or stretch)
//	set min or max of bb to fixed to stretch bb instead of translating it

class blanking {
private:
	vec3_t m_bbmin_init;
	vec3_t m_bbmax_init;
	vec3_t m_bbmin;
	vec3_t m_bbmax;
	vec3_t m_vel_bbox;

	float_t m_max_vel_squared = FLT_MAX;

	bool m_min_fixed = false;
	bool m_max_fixed = false;
public:
	//constructors for various blanking criteria
	//	basically a moving or extending bounding box can be set, along with a maximal velocity the particles may have
	blanking(vec3_t bbmin, vec3_t bbmax);
	blanking(vec3_t bbmin, vec3_t bbmax, vec3_t bbvel);

	blanking(float_t max_vel);

	blanking(vec3_t bbmin, vec3_t bbmax, float_t max_vel);
	blanking(vec3_t bbmin, vec3_t bbmax, vec3_t bbvel, float_t max_vel);

	//one of the sides can be fixed to achieve an expanding rather than moving bounding box
	void set_max_fixed(bool fixed);
	void set_min_fixed(bool fixed);

	// move bounding box with specified velocity / velocities
	void update();

	//does the blanking bounding box move?
	bool is_static() const;

	void get_bb(vec3_t &bbmin, vec3_t &bbmax) const;
	float_t get_max_vel_squared() const;
};

#endif /* BLANKING_H_ */

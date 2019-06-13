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

#include "blanking.h"

blanking::blanking(vec3_t bbmin, vec3_t bbmax) :
		m_bbmin(bbmin), m_bbmax(bbmax), m_bbmin_init(bbmin), m_bbmax_init(bbmax) {}

blanking::blanking(vec3_t bbmin, vec3_t bbmax, vec3_t bbvel) :
		m_bbmin(bbmin), m_bbmax(bbmax), m_bbmin_init(bbmin), m_bbmax_init(bbmax), m_vel_bbox(bbvel) {}

blanking::blanking(float_t max_vel) :
		m_max_vel_squared(max_vel) {}

blanking::blanking(vec3_t bbmin, vec3_t bbmax, float_t max_vel) :
		m_bbmin(bbmin), m_bbmax(bbmax), m_bbmin_init(bbmin), m_bbmax_init(bbmax), m_max_vel_squared(max_vel*max_vel) {}

blanking::blanking(vec3_t bbmin, vec3_t bbmax, vec3_t bbvel, float_t max_vel) :
		m_bbmin(bbmin), m_bbmax(bbmax), m_bbmin_init(bbmin), m_bbmax_init(bbmax), m_vel_bbox(bbvel), m_max_vel_squared(max_vel*max_vel) {}

void blanking::set_max_fixed(bool fixed) {
	m_min_fixed = fixed;
}

void blanking::set_min_fixed(bool fixed) {
	m_max_fixed = fixed;
}

void blanking::update() {
	if (!m_min_fixed) {
		m_bbmin = m_bbmin_init + global_time_current*m_vel_bbox;
	}

	if (!m_max_fixed) {
		m_bbmax = m_bbmax_init + global_time_current*m_vel_bbox;
	}
}

bool blanking::is_static() const {
	if (m_bbmin == m_bbmax) {
		return true;
	}

	return glm::length(m_vel_bbox) == 0.;
}

void blanking::get_bb(vec3_t &bbmin, vec3_t &bbmax) const {
	bbmin = m_bbmin;
	bbmax = m_bbmax;
}

float_t blanking::get_max_vel_squared() const {
	return m_max_vel_squared;
}

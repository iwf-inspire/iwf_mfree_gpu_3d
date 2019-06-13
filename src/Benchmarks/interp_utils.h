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

#ifndef INTERP_UTILS_H_
#define INTERP_UTILS_H_

#include "../types.h"

#include <vector>

//uniformly weighted SPH interpolation in N2
//	interps w value from vals_in onto vals_out
void interp_temps(const std::vector<float4_t> &vals_in, float_t weights, float_t h, std::vector<float4_t> &vals_out);

#endif /* INTERP_UTILS_H_ */

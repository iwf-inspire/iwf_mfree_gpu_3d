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

#include "interp_utils.h"

static float4_t cubic_spline(float4_t posi, float4_t posj, float_t hi) {

	float4_t w;
	w.x = 0.;
	w.y = 0.;
	w.z = 0.;
	w.w = 0.;

	float_t xij = posi.x-posj.x;
	float_t yij = posi.y-posj.y;
	float_t zij = posi.z-posj.z;

	float_t rr2=xij*xij+yij*yij+zij*zij;
	float_t h1 = 1/hi;
	float_t fourh2 = 4*hi*hi;
	if(rr2>=fourh2 || rr2 < 1e-8) {
		return w;
	}

	float_t der;
	float_t val;

	float_t rad=sqrtf(rr2);
	float_t q=rad*h1;
	float_t fac = (M_1_PI)*h1*h1*h1;	// 3D*h1*h1;

	const bool radgt=q>1;
	if (radgt) {
		float_t _2mq  = 2-q;
		float_t _2mq2 = _2mq*_2mq;
		val = 0.25*_2mq2*_2mq;
		der = -0.75*_2mq2 * h1/rad;
	} else {
		val = 1 - 1.5*q*q*(1-0.5*q);
		der = -3.0*q*(1-0.75*q) * h1/rad;
	}

	w.x = val*fac;
	w.y = der*xij*fac;
	w.z = der*yij*fac;
	w.w = der*zij*fac;

	return w;
}

void interp_temps(const std::vector<float4_t> &vals_in, float_t weights, float_t h, std::vector<float4_t> &vals_out) {

	static int calls = 0;

	{
		char buf[256];
		sprintf(buf, "vals_in_%04d.txt", calls);
		FILE *fp =  fopen(buf, "w+");
		for (auto it : vals_in) {
			fprintf(fp, "%f %f %f %f\n", it.x, it.y, it.z, it.w);
		}
		fclose(fp);
	}

	for (auto &it : vals_out) {
		float_t T = 0.;
		for (auto &jt : vals_in) {
			float4_t w = cubic_spline(it, jt, h);
			T += weights*w.x*jt.w;
		}
		it.w = T;
	}

	{
		char buf[256];
		sprintf(buf, "vals_out_%04d.txt", calls);
		FILE *fp =  fopen(buf, "w+");
		for (auto it : vals_out) {
			fprintf(fp, "%f %f %f %f\n", it.x, it.y, it.z, it.w);
		}
		fclose(fp);
	}

	calls++;
}

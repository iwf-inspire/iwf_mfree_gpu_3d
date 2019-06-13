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

// this file implements the radial return w.r.t. the johnson cook flow stress model
//	- a modified Secant algorithm is used to minimize the objective function
//  - this function is called by the plasticity action in actions.h

#ifndef PLASTICITY_CUH_
#define PLASTICITY_CUH_

#include "types.h"

//johnson cook flow stress model
__device__ double sigma_yield(joco_constants jc, double eps_pl, double eps_pl_dot, double t) {
	double theta = (t - jc.Tref)/(jc.Tmelt - jc.Tref);

	double Term_A = jc.A + jc.B * pow(eps_pl, jc.n);
	double Term_B = 1.0;

	double eps_dot = eps_pl_dot / jc.eps_dot_ref;

	if (eps_dot > 1.0) {
		Term_B = 1.0 + jc.C * log_t(eps_dot);
	} else {
		Term_B = pow((1.0 + eps_dot), jc.C);
	}

	double Term_C = 1.0 - pow(theta, jc.m);
	return Term_A * Term_B * Term_C;
}

//optimization function for radial return with J_2 plasticity
__device__ double opt_fun(joco_constants jc, double delta_lambda,
		double norm_Strial, double eps_pl_init, double t, double delta_t, double mu) {

	double eps_pl_equiv = eps_pl_init + sqrt(2.0/3.0) * fmax(delta_lambda,0.);
	double eps_pl_equiv_dot = sqrt(2.0/3.0) *  fmax(delta_lambda,0.) / delta_t;

	double sigmaY = sigma_yield(jc, eps_pl_equiv, eps_pl_equiv_dot, t);

	return norm_Strial - 2*mu*delta_lambda - sqrt(2.0/3.0)*sigmaY;
}

//secant algorithm (for a potentially arbitrary optmization function) with some tweaks for the situation at hand
__device__ double solve_secant(joco_constants jc, double init, double tol,
		double norm_Strial, double eps_pl_init, double t, double delta_t, double mu, bool &failed) {

	double delta_lambda = init;
	double delta_lambda_old = delta_lambda;

	int iter = 0;
	int max_iter = 100;

	do {
		double g = opt_fun(jc, delta_lambda, norm_Strial, eps_pl_init, t, delta_t, mu);
		double delta_lambda_1 = delta_lambda*1.01;
		double g_1 = opt_fun(jc, delta_lambda_1, norm_Strial, eps_pl_init, t, delta_t, mu);

		double slope = (g - g_1) / (delta_lambda - delta_lambda_1);

		delta_lambda_old = delta_lambda;
		delta_lambda = delta_lambda - g/slope;

		if (delta_lambda < 0.) {
			delta_lambda = -0.1*delta_lambda;
		}

		if (fabs(delta_lambda - delta_lambda_old) < tol) {
			return delta_lambda;
		}

		if (iter > max_iter) {
			failed = true;
			return delta_lambda;
		}

		iter++;
	} while(true);
}

#endif /* PLASTICITY_CUH_ */

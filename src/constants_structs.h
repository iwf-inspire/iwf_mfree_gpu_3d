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

#ifndef CONSTANTS_STRUCTS_H_
#define CONSTANTS_STRUCTS_H_

#include "types.h"

#include <cstring>

struct phys_constants {
	float_t E;		// youngs modulus
	float_t nu;		// poisson ratio
	float_t rho0;	// reference density
	float_t K;		// bulk modulus
	float_t G;		// shear modulus
	float_t mass;	// particle mass
};

phys_constants make_phys_constants();

struct trml_constants {
	float_t cp;     // thermal capacity
	float_t tq;	    // taylor quinney constant
	float_t k;	    // thermal conductivity
	float_t alpha;	// thermal diffusitivty
	float_t T_init;	// initial temperature
	float_t eta;	// fraction of frictional power turned to heat
};

trml_constants make_trml_constants();

struct corr_constants {
	float_t wdeltap  ;	// value of kernel function at init particle spacing (for artificial stress)
	float_t stresseps;	// intensity of artificial stress
	float_t xspheps  ;	// XSPH factor (balance between interpolated and advection velocity)
	float_t alpha    ;	// artificial viscosity constant
	float_t beta     ;  // artificial viscosity constant
	float_t eta      ;  // artificial viscosity constant
};

corr_constants make_corr_constants();

//johnson cook material constants
struct joco_constants {
	float_t A;
	float_t B;
	float_t C;
	float_t n;
	float_t m;
	float_t Tmelt;
	float_t Tref;
	float_t eps_dot_ref;
	float_t clamp_temp;			//limit temperature of particles to melting temp in johnson cook?
};

joco_constants make_joco_constants();

//geometrical constants for spatial hashing
struct geom_constants {

	// number of cells in each direction
	int nx;
	int ny;
	int nz;

	// extents of spatial hashing grid
	float_t bbmin_x;
	float_t bbmin_y;
	float_t bbmin_z;

	float_t bbmax_x;
	float_t bbmax_y;
	float_t bbmax_z;

	// edge length of a cell
	float_t dx;
};

geom_constants make_geom_constants();

#endif /* CONSTANTS_STRUCTS_H_ */

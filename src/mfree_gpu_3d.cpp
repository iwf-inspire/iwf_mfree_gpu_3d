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

#include <cuda_profiler_api.h>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <nvml.h>
#include <fenv.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "blanking.h"
#include "Benchmarks/benchmarks_single_grain.h"
#include "Benchmarks/benchmarks_test.h"
#include "vtk_writer.h"
#include "grid_gpu_rothlin.h"
#include "leap_frog.h"
#include "types.h"
#include "tool_3d_gpu.h"

//globals
std::vector<tool_3d_gpu *> global_tool;
tool_forces *global_tool_forces = 0;
blanking *global_blanking = 0;

float_t global_time_dt = 0;
float_t global_time_final = 0;
float_t global_time_current = 0;
int global_time_step = 0;

int poll_temp() {
	FILE *in;
	char buff[512];

	if(!(in = popen("nvidia-smi | grep '[0-9][0-9]C' | awk '{print $3}' | sed 's/C//'", "r"))){
		return -1;
	}

	if (fgets(buff, sizeof(buff), in)!=NULL){
		int temp;
		sscanf(buff, "%d", &temp);
		pclose(in);

		if (temp >= 82) {
			printf("card running dangerousely hot, exiting!\n");
			exit(-1);
		}

		return temp;
	}

	pclose(in);
	return -1;
}

int main(int argc, char *argv[]) {

	//init cuda resources
	cudaFree(0);

	//make results directoy if not present
	struct stat st = {0};
	if (stat("results", &st) == -1) {
		mkdir("results", 0777);
	}

	//clear files from result directory
	int ret;
	ret = system("rm results/*.txt");
	ret = system("rm results/*.vtk");

#if defined Thermal_Conduction_Brookshaw && defined Thermal_Conduction_PSE	// Check #define Thermal_Conduction_Brookshaw & #define Thermal_Conduction_PSE from types.h
	printf("Either define thermal conduction in types.h via Thermal_Conduction_Brookshaw OR Thermal_Conduction_PSE, but not both!\n");
	exit(-1);
#endif

	//spatial hashing structure for fast neighbor search
	grid_base *grid = 0;

	//preliminary benchmarks
//----------------------------------------------------------------------
//	particle_gpu *particles = setup_rings(80, &grid);
//	particle_gpu *particles = setup_disk(80, &grid);
//	particle_gpu *particles = setup_ring_contact(80, &grid);

//	particle_gpu *particles = setup_solid_ball_plastic_wall_impact(10, &grid);
//	particle_gpu *particles = setup_plastic_ball_plastic_wall_impact(10, &grid);

//	particle_gpu *particles = setup_cylinder_impact(80, &grid, true);

	//single and multigrain studies
//----------------------------------------------------------------------
//	particle_gpu *particles = setup_single_grain_1tool_realscale(&grid);
	particle_gpu *particles = setup_single_grain_1tool(&grid);
//	particle_gpu *particles = setup_single_grain_5tool(&grid);
//	particle_gpu *particles = setup_single_grain_1tool_trml_steady(&grid);

	check_cuda_error("init\n");

	if (global_tool.size() == 0) {
		global_tool.push_back(new tool_3d_gpu());
	}

	leap_frog    *stepper = new leap_frog(particles->N, grid->num_cell());

	cudaEvent_t start, stop, intermediate;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&intermediate);
	cudaEventRecord(start);

	int last_step = global_time_final/global_time_dt;
	int freq = last_step/100;

	freq = std::max(freq,1);

#ifdef USE_DOUBLE
	printf("this executable uses DOUBLE precision!\n");
#else
	printf("this executable uses SINGLE precision!\n");
#endif

#ifdef CSPM
	printf("using CSPM\n");
#endif

#ifdef Thermal_Conduction_PSE
	printf("thermal conduction enabled using PSE\n");
#endif

#ifdef Thermal_Conduction_Brookshaw
	printf("thermal conduction enabled using Brookshaw approximation\n");
#endif

	unsigned int print_iter = 0;
	for (int step = 0; step < last_step; step++) {

		bool record_this_step = step % freq == 0;

		if (step > 0) {
			stepper->step(particles, grid, (record_this_step && global_tool_forces != 0));
		}

		if (step == 100) {
			cudaEventRecord(intermediate);
			cudaEventSynchronize(intermediate);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, intermediate);

			float_t seconds_so_far = milliseconds/1e3;
			float_t percent_done = 100*step/((float_t) last_step);
			float_t time_left = seconds_so_far/percent_done*100;
			printf("estimated time: %f seconds, which is %f hours\n", time_left, time_left/60./60.);
		}

		if (record_this_step) {
			vtk_writer_write(particles, global_tool);		// write particles & tool(s) to VTK
			check_cuda_error();

			cudaEventRecord(intermediate);
			cudaEventSynchronize(intermediate);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, intermediate);

			//runtime and temperature report
			float_t seconds_so_far = milliseconds/1e3;
			float_t percent_done = 100*step/((float_t) last_step);
			float_t time_left = seconds_so_far/percent_done*100;
			int temp = 0;

			printf("%06d of %06d: %02.1f percent done, %.2f of est runtime %.2f, at temp %d\n",
					step, last_step, percent_done, seconds_so_far, time_left, temp);

			//write forces on tool (if any)
			if (global_tool_forces != 0) {
				global_tool_forces->report(print_iter);
			}

			//increase numbering of output files
			print_iter++;
		}

		//maintenance of simulation time and increment
		global_time_step++;
		global_time_current += global_time_dt;
	}

	check_cuda_error();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	//final runtime report
	printf("gpu time: (seconds) %f\n", milliseconds/1e3);

	return 0;
}

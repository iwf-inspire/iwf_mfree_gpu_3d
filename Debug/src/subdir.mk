################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/blanking.cpp \
../src/constants_structs.cpp \
../src/geometry_utils.cpp \
../src/grid_cpu_tri.cpp \
../src/json_reader.cpp \
../src/ldynak_reader.cpp \
../src/mfree_gpu_3d.cpp \
../src/surface_tri_mesh.cpp \
../src/tool_3d.cpp \
../src/types.cpp \
../src/vtk_reader.cpp \
../src/vtk_writer.cpp 

CU_SRCS += \
../src/actions_gpu.cu \
../src/eigen_solver.cu \
../src/grid_gpu_base.cu \
../src/grid_gpu_green.cu \
../src/grid_gpu_rothlin.cu \
../src/interactions_gpu.cu \
../src/leap_frog.cu \
../src/particle_gpu.cu \
../src/tool_3d_gpu.cu 

CU_DEPS += \
./src/actions_gpu.d \
./src/eigen_solver.d \
./src/grid_gpu_base.d \
./src/grid_gpu_green.d \
./src/grid_gpu_rothlin.d \
./src/interactions_gpu.d \
./src/leap_frog.d \
./src/particle_gpu.d \
./src/tool_3d_gpu.d 

OBJS += \
./src/actions_gpu.o \
./src/blanking.o \
./src/constants_structs.o \
./src/eigen_solver.o \
./src/geometry_utils.o \
./src/grid_cpu_tri.o \
./src/grid_gpu_base.o \
./src/grid_gpu_green.o \
./src/grid_gpu_rothlin.o \
./src/interactions_gpu.o \
./src/json_reader.o \
./src/ldynak_reader.o \
./src/leap_frog.o \
./src/mfree_gpu_3d.o \
./src/particle_gpu.o \
./src/surface_tri_mesh.o \
./src/tool_3d.o \
./src/tool_3d_gpu.o \
./src/types.o \
./src/vtk_reader.o \
./src/vtk_writer.o 

CPP_DEPS += \
./src/blanking.d \
./src/constants_structs.d \
./src/geometry_utils.d \
./src/grid_cpu_tri.d \
./src/json_reader.d \
./src/ldynak_reader.d \
./src/mfree_gpu_3d.d \
./src/surface_tri_mesh.d \
./src/tool_3d.d \
./src/types.d \
./src/vtk_reader.d \
./src/vtk_writer.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_60,code=sm_60  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_30,code=compute_30 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_60,code=sm_60  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_60,code=sm_60  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



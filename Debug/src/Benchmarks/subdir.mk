################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Benchmarks/benchmarks_test.cpp \
../src/Benchmarks/interp_utils.cpp 

CU_SRCS += \
../src/Benchmarks/benchmarks_single_grain.cu 

CU_DEPS += \
./src/Benchmarks/benchmarks_single_grain.d 

OBJS += \
./src/Benchmarks/benchmarks_single_grain.o \
./src/Benchmarks/benchmarks_test.o \
./src/Benchmarks/interp_utils.o 

CPP_DEPS += \
./src/Benchmarks/benchmarks_test.d \
./src/Benchmarks/interp_utils.d 


# Each subdirectory must supply rules for building sources it contributes
src/Benchmarks/%.o: ../src/Benchmarks/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_60,code=sm_60  -odir "src/Benchmarks" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_30,code=compute_30 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_60,code=sm_60  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/Benchmarks/%.o: ../src/Benchmarks/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_60,code=sm_60  -odir "src/Benchmarks" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



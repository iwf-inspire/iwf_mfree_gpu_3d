# iwf_mfree_gpu_3d

This is the source code to the publication **Meshless Single Grain Cutting Simulations on the GPU**, to be published in a special issue of the International Journal of Mechatronics and Manufacturing Systems [**IJMMS**](https://www.inderscience.com/jhome.php?jcode=ijmms). Its main purpose is to simulate single or multi grain cutting processes at high resolution, using GPU computing via CUDA as acceleration strategy. The calculations are completely meshfree using Smoothed Particle Hydrodynamics (SPH). Besides the parallelization on the GPU, the package offers some unique features not available in commercial solver packages like LSDYNA or ABAQUS:

* Stabilization of the solution using the complete array of techniques presented in Gray, J. P., J. J. Monaghan, and R. P. Swift. "SPH elastic dynamics." Computer methods in applied mechanics and engineering 190.49-50 (2001): 6641-6662. 
* Thermal solver using either Particle Strength Exchange or the Brookshaw Approximation, including thermal contact between the tool and the workpiece. 
* Linear complete Kernels using the Randles Libserky (also known as CSPM) correction.

Parallelization on the GPU allows for dramatic runtime efficiency, bringing simulations in the range of millions of particles to the realm of overnight computations using a single nVidia Tesla P100. In fact, to the best of our knowledge, **iwf_mfree_gpu_3d** was used to produce the most high resolution meshless metal cutting simulations to date.

Result frames can be viewed using [ParaView](https://www.paraview.org/) using the legacy VTK format. Some typical result frames are shown below:

Single grain cutting, realistic chip curling is achieved using linear complete kernels:

![single](https://raw.githubusercontent.com/mroethli/iwf_mfree_gpu_3d/master/img/single.jpg)

Multi grain cutting. Interactions between grains due to heated and pre-stressed material can be studied.

![multi](https://raw.githubusercontent.com/mroethli/iwf_mfree_gpu_3d/master/img/multi.jpg)

**iwf_mfree_gpu_3d** was tested on various versions of Ubuntu Linux. The only dependency is [GLM](https://glm.g-truc.net/0.9.9/index.html). Make files for both a Release version and a Debug build are provided. Tested under NVCC with the GCC suite as host compiler, but any C++11 compliant host compiler should suffice. **iwf_mfree_gpu_3d** was devleloped at [ETHZ](www.ethz.ch) by the following authors

* Matthias Röthlin, mroethli@ethz.ch
* Hagen Klippel, hklippel@ethz.ch
* Mohamadreza Afrasiabi, afrasiabi@ethz.ch

**iwf_mfree_gpu_3d** is free software and licensed under GPLv3

**iwf_mfree_gpu_3d** contains parts of other free software packages:

* David Eberlys Eigen solver, which is part of [GeometricTools](www.geometrictools.com/), was ported to CUDA. Boost License
* The Axis Aligned Bounding Box Triangle cover test by [Tomas Akenine-Möller](http://cs.lth.se/tomas-akenine-moller/) is used in the hashing procedure for the contact algorithm. Public Domain. 
* Ray Triangle Intersection Test, Möller–Trumbore algorithm, taken from [wikipedia](https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm).
* The particles CUDA [example](https://github.com/zchee/cuda-sample/tree/master/5_Simulations/particles) by nVidia was adapted for one of the two available particle hashing codes. See [License](https://github.com/NVIDIA/cuda-samples/blob/master/LICENSE)

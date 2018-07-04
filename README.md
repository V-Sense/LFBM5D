# Light field super-resolution with the SR-LFBM5D filter

![LFBM5D SR example](https://v-sense.scss.tcd.ie/wp-content/uploads/2018/02/cover-1.png)

This is a MATLAB/C/C++ implementation of the SR-LFBM5D filter for light field super-resolution described in [1], extending the LFBM5D denoising filter [2]. For more details see [our webpage](https://v-sense.scss.tcd.ie/?p=1551).
If you use or adapt this code in your work (either as a stand-alone tool or as a component of any algorithm), you need to cite the appropriate paper [1].
* Author    : Martin Alain (Email: <alainm@scss.tcd.ie>)
* Copyright : (C) 2018 [V-SENSE](https://v-sense.scss.tcd.ie)
* Licence   : GPL v3+, see GPLv3.txt

This code is built upon the C/C++ implementation of the LFBM5D denoising filter released with [2] available on the master branch. The main loop of the algorithm is implemented in MATLAB, calling the executable compiled from the C/C++ sources during the first step.

## Light field conventions

The input light field should be in the form of a collection of PNG sub-aperture images located in a common folder. The naming convention for the sub-aperture images must be of the form <SAI_name>\_<s_idx>\_<t_idx>.png, where <SAI_name> is the common name for all images (e.g. 'SAI' as in our testing example, 'Img', 'Cam'...), and <s_idx> and <t_idx> correspond to the angular indexes of the light field images.

Futhermore, the following conventions/notations are adopted in the code:
- Angular size will be denoted awidth and aheight (and associated derivations) for angular width and angular height (noted $n_a$ in [1]);
- In general, prefix 'a' before a variable stands for angular;
- Indexes for angular dimensions will be denoted s and t (and associated derivations);
- Image size will be denoted as usual width and height (and associated derivations);
- Indexes for image dimensions will be denoted i and j (legacy from the BM3D implementation, noted x and y in [1]);
- LF  = Light Field;
- SAI = Sub-Aperture Image.

## Source code compilation

The source code can be compiled on Unix/Linux/Mac OS using the make program.

This code requires the libpng library and the fftw library (e.g. on Ubuntu run `apt-get install libpng-dev` and `apt-get install libfftw3-dev`, on Mac OS run `brew install libpng` and `brew install fftw`).

Furthermore we highly recommend using the [Open Multi-Processing multithread parallelization](http://openmp.org/) to speed up the processing time. Simply add OMP=1 after the make command to use OpenMP.

Instructions to compile the code:
- Download/clone the repository and go to the corresponding directory.
- Run: `make [OMP=1]`. This generates an executable named LFBM5DSR which will be called in the MATLAB BM5DSR function.

Note that compared to the denoising filter, only the hard-thresholding step is performed. If you wish to perform the additional Wiener filtering step, you need to uncomment the corresponding part of the code in the main.cpp file, and re-compile the code. You also need to uncomment line 169 in the BM5DSR.m file.

## Usage

- First run the generateDatasetSR.m script to generate the low resolution and corresponding upsampled (using a bicubic filter) light fields.
- Then run the runBM5DSR.m script to get the super-resolution results.

In this version of the code, the test light field provided in testing/sourceLF is used. The results are written in testing/SR_xX, where X is the scaling factor.

To use your own dataset or different scaling factors, you need to modify the getParams.m file. Note that the getParams() function is used to ensure that the generateDatasetSR and runBM5DSR scripts run with the same parameters.

## References

[1] Martin Alain, Aljosa Smolic, "Light Field Super-Resolution via LFBM5D Sparse Coding", IEEE International Conference on Image Processing (ICIP 2018), 2018, https://v-sense.scss.tcd.ie/?p=1551

[2] Martin Alain, Aljosa Smolic, "Light Field Denoising by Sparse 5D Transform Domain Collaborative Filtering", IEEE International Workshop on Multimedia Signal Processing (MMSP 2017), 2017, https://v-sense.scss.tcd.ie/?p=893

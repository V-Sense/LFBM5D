# Light field denoising with the LFBM5D filter

![LFBM5D denoising example](https://v-sense.scss.tcd.ie/wp-content/uploads/2017/08/comparison_before_after_dn.png)

This is a C/C++ implementation of the LFBM5D filter for light field denoising described in [1]. For more details see [our webpage](https://v-sense.scss.tcd.ie/?p=893).
If you use or adapt this code in your work (either as a stand-alone tool or as a component of any algorithm), you need to cite the appropriate paper [1].
* Author    : Martin Alain (Email: <alainm@scss.tcd.ie>)
* Copyright : (C) 2017 [V-SENSE](https://v-sense.scss.tcd.ie)
* Licence   : GPL v3+, see GPLv3.txt

This code is built upon the [IPOL implementation of the BM3D denoising filter](https://doi.org/10.5201/ipol.2012.l-bm3d) released with [2].

In addition to the LFBM5D filter, this code provides an implementation of light field denoising using the BM3D filter applied to every sub-aperture images (used to provide results in [1]).

## Light field conventions

The input light field should be in the form of a collection of PNG sub-aperture images located in a common folder. The scanning order of the images over the 2D angular dimensions needs to be a Z-scan, either row-wise or column-wise (which is specified by a a parameter in the command line). Make sure the scanning order is respected when you type the 'ls' command in the light field folder.

Futhermore, the following conventions/notations are adopted in the code:
- Angular size will be denoted awidth and aheight (and associated derivations) for angular width and angular height (noted $n_a$ in [1]);
- In general, prefix 'a' before a variable stands for angular;
- Indexes for angular dimensions will be denoted s and t (and associated derivations);
- Image size will be denoted as usual width and heigth (and associated derivations);
- Indexes for image dimensions will be denoted i and j (legacy from the BM3D implementation, noted x and y in [1]);
- LF  = Light Field;
- SAI = Sub-Aperture Image.

## Source code compilation

The source code can be compiled on Unix/Linux/Mac OS using the make program. For Windows 10 users, we highly recommend [installing Bash shell to run Linux commands](https://www.windowscentral.com/how-install-bash-shell-command-line-windows-10).

This code requires the libpng library and the fftw library (e.g. on Ubuntu run `apt-get install libpng-dev` and `apt-get install libfftw3-dev`, on Mac OS run `brew install libpng` and `brew install fftw`).

Furthermore we highly recommend using the [Open Multi-Processing multithread parallelization](http://openmp.org/) to speed up the processing time. Simply add OMP=1 after the make command to use OpenMP.

Instructions to compile the code:
- Download/clone the repository and go to the corresponding directory.
- To compile the LFBM5D filter run: `make [OMP=1]`. This generates an executable named LFBM5Ddenoising.
- To compile the BM3D applied on SAIs run: `make -f Makefile_LFBM3D [OMP=1]`. This generates an executable named LFBM3Ddenoising.

## Testing

For testing purposes, we provide in folder ./testing/sourceLF a (very) small example light field consisting of 3x3 color sub-aperture images of size 256x256, obtained from the Lego Knights light field of the [Stanford dataset](http://lightfield.stanford.edu/lfs.html).

First run: `mkdir ./testing/noisyLF ./testing/basicLF ./testing/denoisedLF ./testing/diffLF`

Then run one of the following command to test the executable of your choice:
- `./LFBM5Ddenoising ./testing/sourceLF 3 3 1 1 row 25 2.7 ./testing/noisyLF ./testing/basicLF ./testing/denoisedLF ./testing/diffLF 8 18 6 16 4 id sadct haar 0 16 18 6 8 4 dct sadct haar 0 opp 0 ./testing/outputMeasuresLFBM5D.txt`
- `./LFBM3Ddenoising ./testing/sourceLF 3 3 1 1 row 25 2.7 ./testing/noisyLF ./testing/basicLF ./testing/denoisedLF ./testing/diffLF 16 16 8 3 bior 0 32 16 8 3 dct 0 opp 8 ./testing/outputMeasuresLFBM3D.txt`

## Command line examples

The following command line examples were used to generate results in [1]. See the next section for a detailed decription of command line parameters.

Synthetic noise:
- [Stanford dataset](http://lightfield.stanford.edu/lfs.html):
	- `./LFBM5Ddenoising path/to/sourceLF 17 17 1 1 row 10 2.7 path/to/noisyLF path/to/basicLF path/to/denoisedLF path/to/diffLF 8 18 6 16 4 id sadct haar 0 16 18 6 8 4 dct sadct haar 0 opp 0 outputMeasuresLFBM5D.txt`
	- `./LFBM3Ddenoising path/to/sourceLF 17 17 1 1 row 10 2.7 path/to/noisyLF path/to/basicLF path/to/denoisedLF path/to/diffLF 16 16 8 3 bior 0 32 16 8 3 dct 0 opp 0 outputMeasuresLFBM3D.txt`

- [EPFL dataset](https://mmspg.epfl.ch/EPFL-light-field-image-dataset):
	- `./LFBM5Ddenoising path/to/sourceLF 15 15 1 1 row 10 2.7 path/to/noisyLF path/to/basicLF path/to/denoisedLF path/to/diffLF 1 18 3 16 3 bior sadct haar 0 8 18 3 8 3 dct sadct haar 0 opp 0 outputMeasuresLFBM5D.txt`
	- `./LFBM3Ddenoising path/to/sourceLF 15 15 1 1 row 10 2.7 path/to/noisyLF path/to/basicLF path/to/denoisedLF path/to/diffLF 16 16 8 3 bior 0 32 16 8 3 dct 0 opp 0 outputMeasuresLFBM3D.txt`
	
Lenslet noise removal:
- EPFL dataset:
	- `./LFBM5Ddenoising none 15 15 1 1 row 10 2.7 path/to/noisyLF path/to/basicLF path/to/denoisedLF none 8 18 6 16 4 bior sadct haar 0 16 18 6 8 4 dct sadct haar 0 opp 0 outputMeasuresLFBM5D.txt`
	- `./LFBM3Ddenoising none 15 15 1 1 row 10 2.7 path/to/noisyLF path/to/basicLF path/to/denoisedLF none 16 16 8 3 bior 0 32 16 8 3 dct 0 opp 0 outputMeasuresLFBM3D.txt`
	

## Generic commands and parameters description
The generic ways to run the executables are listed below:

- `./LFBM5Ddenoising LFSourceDir awidth aheight aswSizeHard aswSizeWien angMajor sigma lambda LFNoisyDir LFBasicDir LFDenoisedDir LFDiffDir NHard nSimHard nDispHard khard pHard tau2DHard tau4DHard tau5DHard useSDHard NWien nSimWien nDispWien kWien pWien tau2DWien tau4DWien tau5DWien useSDWien colorSpace nbThreads resultsFile`
				  
- `./LFBM3Ddenoising LFSourceDir awidth aheight aswSizeHard aswSizeWien angMajor sigma lambda LFNoisyDir LFBasicDir LFDenoisedDir LFDiffDir NHard nHard kHard pHard tau2DHard NWien nWien kWien pWien tau2DWien colorSpace nbThreads resultsFile`

with parameters:
- LFSourceDir (string): input source light field directory containing noise-free sub-aperture images (used as ground truth in PSNR computation). Use 'none' for this parameter to specify that no ground truth is available and direclty use LFNoisyDir as input.
- awidth, aheight (int): angular size of the light field.
- aswSizeHard, aswSizeWien (int): half size of the angular search window for the hard thresholding and Wiener step respectively (the full angular search window size is noted $n_a$ in [1]).
	- recommended value: 1.
- angMajor (string): indicate if the sub-aperture images of the light field are ordered row-wise or column-wise.
	- expected value: 'row' or 'col'.
- sigma (float): standard deviation of additive white gaussian noise. If no ground truth is provided (LFSourceDir=none), this parameter should correspond to an estimated value of the noise level in the input light field from LFNoisyDir.
- lambda (float): threshold coefficient for the hard thresholding step (see equation (6) in [1]).
	- recommended value: 2.7.
- LFNoisyDir (string): light field directory which will contain the noisy sub-aperture images.
- LFBasicDir (string): light field directory which will contain the sub-aperture images after the hard thresholding step of the algorithm.
- LFDenoisedDir (string): light field directory which will contain the sub-aperture images after the Wiener step of the algorithm.
- LFDiffDir (string): directory which will contain the difference between the input sub-aperture images from LFSourceDir and the denoised sub-aperture images from LFDenoisedDir.
- NHard, NWien (int, power of 2): maximum number of similar patches for self-similarities search for the hard thresholding and Wiener step respectively.
- nHard, nWien (for BM3D) / nSimHard, nSimWien (for LFBM5D) (int): half-size of the search window for self-similarities search for the hard thresholding and Wiener step respectively.
- nDispHard, nDispWien (int): half-size of the search window for disparity compensation for the hard thresholding and Wiener step respectively.
- khard, kWien (int): patch size for the hard thresholding and Wiener step respectively.
- pHard, pWien (int): processing step in row and column for the hard thresholding and Wiener step respectively.
- tau2DHard, tau2DWien (string): 2D spatial transform for the hard thresholding and Wiener step respectively (noted $\tau_{2D}^s$ in [1]).
	- expected value: 'id' for no transform, 'dct' for 2D discrete cosine transform, 'bior' for 2D biorthogonal 1.5 wavelet.
- tau4DHard, tau4DWien (string): 2D angular transform for the hard thresholding and Wiener step respectively (noted $\tau_{2D}^a$ in [1]).
	- expected value: 'id' for no transform, 'dct' for 2D discrete cosine transform, 'sadct' for 2D shape-adaptive discrete cosine transform.
- tau5DHard, tau5DWien (string): 1D transform along 5th dimension for the hard thresholding and Wiener step respectively (noted $\tau_{1D}$ in [1]).
	- expected value: 'hw' for 1D Hadamard-Walsh transform, 'haar' for 1D Haar wavelet, 'dct' for 1D discrete cosine transform.
- useSDHard, useSDWien (bool): indicate if the standard variation for the weighted aggregation is used for the hard thresholding and Wiener step respectively.
	- recommended value: 0.
- colorSpace (string): choice of the color space applied on each sub-aperture image.
	- expected value: 'rgb', 'yuv', 'ycbcr', 'opp'.
	- recommended value: 'opp'.
- nbThreads (int, power of 2): specifies the number of threads to use with OpenMP. If 0, the maximum number of available threads is used.
	- recommended value: 0.
- resultsFile (string): text file which will contains all the objective measures PSNR for each sub-aperture image as well as average value and standard deviation over the light field.


## References

[1] Martin Alain, Aljosa Smolic, "Light Field Denoising by Sparse 5D Transform Domain Collaborative Filtering", IEEE International Workshop on Multimedia Signal Processing (MMSP 2017), 2017, https://v-sense.scss.tcd.ie/?p=893

[2] Marc Lebrun, "An Analysis and Implementation of the BM3D Image Denoising Method", Image Processing On Line, 2 (2012), pp 175-213, https://doi.org/10.5201/ipol.2012.l-bm3d

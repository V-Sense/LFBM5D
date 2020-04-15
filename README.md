# Light field denoising with the LFBM5D filter

![LFBM5D denoising example](https://v-sense.scss.tcd.ie/wp-content/uploads/2017/08/comparison_before_after_dn.png)

This is a C/C++ implementation of the LFBM5D filter for light field denoising described in [1]. For more details see [our webpage](https://v-sense.scss.tcd.ie/?p=893).
If you use or adapt this code in your work (either as a stand-alone tool or as a component of any algorithm), you need to cite the appropriate paper [1].
* Author    : Martin Alain (Email: <alainm@scss.tcd.ie>)
* Copyright : (C) 2017 [V-SENSE](https://v-sense.scss.tcd.ie)
* Licence   : GPL v3+, see GPLv3.txt

This code is built upon the [IPOL implementation of the BM3D denoising filter](https://doi.org/10.5201/ipol.2012.l-bm3d) released with [2].

In addition to the LFBM5D filter, this code provides an implementation of light field denoising using the BM3D filter applied to every sub-aperture images (used to provide results in [1]).

For the super-resolution extension SR-LFBM5D [3], switch to the SR branch.

## Light field conventions

The input light field should be in the form of a collection of PNG sub-aperture images located in a common folder. The naming convention for the sub-aperture images must be of the form <SAI_name>\_<s_idx>\_<t_idx>.png, where <SAI_name> is the common name for all images (e.g. 'SAI' as in our testing example, 'Img', 'Cam'...), and <s_idx> and <t_idx> correspond to the angular indexes of the light field images. If the <s_idx> corresponds to the row indexes of your light field, the light field is considered to be ordered row-wise, and the parameter angMajor should be set to 'row'. Otherwise, the light field is considered to be ordered column-wise, and the parameter angMajor should be set to 'col'.
See the next sections for some examples.

Futhermore, the following conventions/notations are adopted in the code:
- Angular size will be denoted awidth and aheight (and associated derivations) for angular width and angular height (noted $n_a$ in [1]);
- In general, prefix 'a' before a variable stands for angular;
- Indexes for angular dimensions will be denoted s and t (and associated derivations);
- Image size will be denoted as usual width and height (and associated derivations);
- Indexes for image dimensions will be denoted i and j (legacy from the BM3D implementation, noted x and y in [1]);
- LF  = Light Field;
- SAI = Sub-Aperture Image.

## Source code compilation

The source code can be compiled on Unix/Linux/Mac OS using the cmake and make program. For Windows 10 users, we provide an executable which should be compatible with x64 platforms. If the .exe is not working or you wish to modify the code, you will have to compile it yourself using Visual Studio, see install_Windows10.md for more details.

This code requires the libpng library and the fftw library. On Ubuntu run `apt-get install libpng-dev` and `apt-get install libfftw3-dev`, on Mac OS run `brew install libpng` and `brew install fftw`. On Windows the process is a bit more complex, see the instructions in install_Windows10.md.

Instructions to compile the code:
- Download/clone the repository and go to the corresponding directory.
- Create a build directory and cd into it, e.g. run: `mkdir build && cd build`.
- Run cmake: `cmake ..`. ( You might want to add the options similar to `-G "Visual Studio 15 2017 Win64"` on Windows 64).
- On Unix/Linux/Mac OS, run: `make`. On Windows 10 run the BM5DProject.sln and compile with Visual Studio. This generates two executables named LFBM5Ddenoising and LFBM3Ddenoising (.exe on Windows).

## Testing

For testing purposes, we provide in folder ./testing/sourceLF a (very) small example light field consisting of 3x3 color sub-aperture images of size 256x256, obtained from the Lego Knights light field of the [Stanford dataset](http://lightfield.stanford.edu/lfs.html).

First run: `mkdir ./testing/noisyLF ./testing/basicLF ./testing/denoisedLF ./testing/diffLF`

Then run one of the following command to test the executable of your choice:
- `./LFBM5Ddenoising ./testing/sourceLF SAI _ 3 3 1 1 1 1 row 25 2.7 ./testing/noisyLF ./testing/basicLF ./testing/denoisedLF ./testing/diffLF 8 18 6 16 4 id sadct haar 0 16 18 6 8 4 dct sadct haar 0 opp 0 ./testing/outputMeasuresLFBM5D.txt`
- `./LFBM3Ddenoising ./testing/sourceLF SAI _ 3 3 1 1 1 1 row 25 2.7 ./testing/noisyLF ./testing/basicLF ./testing/denoisedLF ./testing/diffLF 16 16 8 3 bior 0 32 16 8 3 dct 0 opp 8 ./testing/outputMeasuresLFBM3D.txt`

For Windows users, we recommend using the Developer Command Prompt for Visual Studio to run these commands, and replace ./LFBMxDdenoising by LFBMxDdenoising.exe. Make sure that libfftw3f-3.dll is in the same folder as your executable.

## Command line examples

The following command line examples were used to generate results in [1]. See the next section for a detailed description of command line parameters.

Synthetic noise:
- [Stanford dataset](http://lightfield.stanford.edu/lfs.html):
	- `./LFBM5Ddenoising path/to/sourceLF SAI _ 17 17 0 0 1 1 row 10 2.7 path/to/noisyLF path/to/basicLF path/to/denoisedLF path/to/diffLF 8 18 6 16 4 id sadct haar 0 16 18 6 8 4 dct sadct haar 0 opp 0 outputMeasuresLFBM5D.txt`
	- `./LFBM3Ddenoising path/to/sourceLF SAI _ 17 17 0 0 1 1 row 10 2.7 path/to/noisyLF path/to/basicLF path/to/denoisedLF path/to/diffLF 16 16 8 3 bior 0 32 16 8 3 dct 0 opp 0 outputMeasuresLFBM3D.txt`

- [EPFL dataset](https://mmspg.epfl.ch/EPFL-light-field-image-dataset):
	- `./LFBM5Ddenoising path/to/sourceLF SAI _ 15 15 0 0 1 1 row 10 2.7 path/to/noisyLF path/to/basicLF path/to/denoisedLF path/to/diffLF 1 18 3 16 3 bior sadct haar 0 8 18 3 8 3 dct sadct haar 0 opp 0 outputMeasuresLFBM5D.txt`
	- `./LFBM3Ddenoising path/to/sourceLF SAI _ 15 15 0 0 1 1 row 10 2.7 path/to/noisyLF path/to/basicLF path/to/denoisedLF path/to/diffLF 16 16 8 3 bior 0 32 16 8 3 dct 0 opp 0 outputMeasuresLFBM3D.txt`
	
Lenslet noise removal:
- EPFL dataset:
	- `./LFBM5Ddenoising none SAI _ 15 15 0 0 1 1 row 10 2.7 path/to/noisyLF path/to/basicLF path/to/denoisedLF none 8 18 6 16 4 bior sadct haar 0 16 18 6 8 4 dct sadct haar 0 opp 0 outputMeasuresLFBM5D.txt`
	- `./LFBM3Ddenoising none SAI _ 15 15 0 0 1 1 row 10 2.7 path/to/noisyLF path/to/basicLF path/to/denoisedLF none 16 16 8 3 bior 0 32 16 8 3 dct 0 opp 0 outputMeasuresLFBM3D.txt`
	
The following command line example was used to generate results in [4]. They provide faster processing for similar denoising performances to the command line above.
- `./LFBM5Ddenoising none SAI _ 15 15 0 0 1 1 row 10 2.7 path/to/noisyLF path/to/basicLF path/to/denoisedLF none 1 16 3 16 5 bior sadct haar 0 8 16 3 8 5 dct sadct haar 0 opp 0 outputMeasuresLFBM5D.txt`
	

## Generic commands and parameters description
The generic ways to run the executables are listed below:

- `./LFBM5Ddenoising LFSourceDir SAIName SAINameSeparator awidth aheight sIdxStart tIdxStart aswSizeHard aswSizeWien angMajor sigma lambda LFNoisyDir LFBasicDir LFDenoisedDir LFDiffDir NHard nSimHard nDispHard khard pHard tau2DHard tau4DHard tau5DHard useSDHard NWien nSimWien nDispWien kWien pWien tau2DWien tau4DWien tau5DWien useSDWien colorSpace nbThreads resultsFile`
				  
- `./LFBM3Ddenoising LFSourceDir SAIName SAINameSeparator awidth aheight sIdxStart tIdxStart aswSizeHard aswSizeWien angMajor sigma lambda LFNoisyDir LFBasicDir LFDenoisedDir LFDiffDir NHard nHard kHard pHard tau2DHard NWien nWien kWien pWien tau2DWien colorSpace nbThreads resultsFile`

with parameters:
- LFSourceDir (string): input source light field directory containing noise-free sub-aperture images (used as ground truth in PSNR computation). Use 'none' for this parameter to specify that no ground truth is available and direclty use LFNoisyDir as input.
- SAIName, SAINameSeparator (string): common name for all sub-aperture images contained in the LFSourceDir directory and character to separate angular indexes.
- awidth, aheight (int): angular size of the light field.
- sIdxStart, tIdxStart (int): first angular indexes of the light field, corresponding to the top left image. 
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

## Acknowledgement

Thanks to Pierre Allain from INRIA Rennes for providing the cmake files.


## References

[1] Martin Alain, Aljosa Smolic, "Light Field Denoising by Sparse 5D Transform Domain Collaborative Filtering", IEEE International Workshop on Multimedia Signal Processing (MMSP 2017), 2017, https://v-sense.scss.tcd.ie/research/light-fields/light-field-denoising/

[2] Marc Lebrun, "An Analysis and Implementation of the BM3D Image Denoising Method", Image Processing On Line, 2 (2012), pp 175-213, https://doi.org/10.5201/ipol.2012.l-bm3d

[3] Martin Alain, Aljosa Smolic, "Light Field Super-Resolution via LFBM5D Sparse Coding", IEEE International Conference on Image Processing (ICIP 2018), 2018, https://v-sense.scss.tcd.ie/research/light-fields/light-field-super-resolution-via-lfbm5d-sparse-coding/

[4] Pierre Matysiak, Mairead Grogan, Mikael Le Pendu, Martin Alain, Aljosa Smolic, "A pipeline for lenslet light field quality enhancement", IEEE International Conference on Image Processing (ICIP 2018), 2018, https://v-sense.scss.tcd.ie/research/light-fields/high-quality-light-field-extraction/

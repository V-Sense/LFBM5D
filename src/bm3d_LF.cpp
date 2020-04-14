/*
 * Copyright (c) 2017, Martin Alain <alainm@scss.tcd.ie>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */


/**
 * @file bm3d_LF.cpp
 * @brief BM3D denoising function for light field (each SAI/EPI is processed independantly)
 *
 * @author Martin Alain <alainm@scss.tcd.ie>
 **/

 #include <stdlib.h>
 #include <iostream>
 #include <string>

 #include "bm3d_LF.h"
 #include "utilities.h"

 #ifdef _OPENMP
    #include <omp.h>
 #endif

 using namespace std;

/** ----------------- **/
/** - Main function - **/
/** ----------------- **/
/**
 * @brief run BM3D process for each SAI or EPI of the light field. Depending on if OpenMP is used or not,
 *        and on the number of available threads, it divides the noisy
 *        SAI in sub_images, to process them in parallel.
 *
 * @param sigma: value of assumed noise of the noisy image;
 * @param LF_noisy: noisy light field;
 * @param LF_SAI_mask: mask to indicate if a noisy image should be processed (1) or not (0);
 * @param LF_basic: will be the basic estimation after the 1st step;
 * @param LF_denoised: will be the denoised final image;
 * @param width, height, chnls: size of the image;
 * @param nHard (resp. nWien): Half size of the search window for Hard Thresholding (resp. Wiener) step;
 * @param kHard (resp. kWien): Patches size for Hard Thresholding (resp. Wiener) step;
 * @param NHard (resp. NWien): number of nearest neighbor for a patch for Hard Thresholding (resp. Wiener) step;
 * @param pHard (resp. pWien): Processing step on row and columns for Hard Thresholding (resp. Wiener) step;
 * @param useSD_h (resp. useSD_w): if true, use weight based
 *        on the standard variation of the 3D group for the
 *        first (resp. second) step, otherwise use the number
 *        of non-zero coefficients after Hard Thresholding
 *        (resp. the norm of Wiener coefficients);
 * @param tau_2D_hard (resp. tau_2D_wien): 2D transform to apply
 *        on every 3D group for the first (resp. second) part.
 *        Allowed values are DCT and BIOR;
 * @param lambdaHard3D: Threshold for Hard Thresholding;
 * @param color_space: Transformation from RGB to YUV. Allowed
 *        values are RGB (do nothing), YUV, YCBCR and OPP;
 * @param nb_threads: number of threads for OpenMP parallel processing;
 * @param sub_img_name: name for LF images, either 'SAI' or 'EPI'.
 *
 * @return EXIT_FAILURE if color_space has not expected
 *         type, otherwise return EXIT_SUCCESS.
 **/
int run_bm3d_LF(
    const float sigma
,   vector<vector<float> > &LF_noisy
,   vector<unsigned> &LF_SAI_mask
,   vector<vector<float> > &LF_basic
,   vector<vector<float> > &LF_denoised
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned nHard
,   const unsigned nWien
,   const unsigned kHard
,   const unsigned kWien
,   const unsigned NHard
,   const unsigned NWien
,   const unsigned pHard
,   const unsigned pWien
,   const bool useSD_h
,   const bool useSD_w
,   const unsigned tau_2D_hard
,   const unsigned tau_2D_wien
,   const float    lambdaHard3D
,   const unsigned color_space
,   unsigned nb_threads
,   char *sub_img_name
){
    //! Check memory allocation
    if (LF_basic.size() != LF_noisy.size())
        LF_basic.resize(LF_noisy.size());
    if (LF_denoised.size() != LF_noisy.size())
        LF_denoised.resize(LF_noisy.size());

    cout << endl << "Number of threads which will be used: " << nb_threads;
    #ifdef _OPENMP
        cout << " (real available cores: " << omp_get_num_procs() << ")" << endl;
    #else
        cout << endl;
    #endif
        cout << endl;

    //! Run BM3D for each SAI
    for(unsigned st = 0; st < LF_noisy.size(); st++)
    {
        if(LF_SAI_mask[st])
        {
            cout << "\r - > Running BM3D filter on " + string(sub_img_name) + " " << st << flush;
            run_bm3d(sigma, LF_noisy[st], LF_basic[st], LF_denoised[st], width, height, chnls, 
                    nHard, nWien, kHard, kWien, NHard, NWien, pHard, pWien,
                    useSD_h, useSD_w, tau_2D_hard, tau_2D_wien, lambdaHard3D, color_space, nb_threads);
        }
    }
    cout << endl;
    return EXIT_SUCCESS;
}




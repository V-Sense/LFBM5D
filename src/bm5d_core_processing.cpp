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
 * @file bm5d_core_processing.cpp
 * @brief LFBM5D core denoising functions (step 1 and 2) - inner loop for each SAI
 *
 * @author Martin Alain <alainm@scss.tcd.ie>
 **/

 #include <stdlib.h>
 #include <iostream>
 #include <sstream>
 #include <fstream>
 #include <algorithm>
 #include <vector>
 #include <cmath>
 #include <numeric>

 #include "bm5d_core_processing.h"
 #include "utilities_LF.h"
 #include "lib_transforms.h"

 #define SQRT2     1.414213562373095
 #define SQRT2_INV 0.7071067811865475
 #define YUV       0
 #define YCBCR     1
 #define OPP       2
 #define RGB       3
 #define ID        4
 #define DCT       5
 #define SADCT     6
 #define BIOR      7
 #define HADAMARD  8
 #define HAAR      9
 #define NONE      10
 #define ROWMAJOR  11
 #define COLMAJOR  12

 #ifdef _OPENMP
    #include <omp.h>
 #endif

 using namespace std;

/**
 * @brief Run the basic process of LFBM5D (Hard Thresholding step). The results
 *        numerator and denominator are contained in LF_basic_num and LF_basic_den respectively.
 *        Each image in the light field has boundary, which
 *        are here only for block-matching and do not need to be
 *        denoised.
 *
 * @param sigma: value of assumed noise of the image to denoise;
 * @param lambdaHard5D: Threshold for Hard Thresholding
 * @param LF_noisy: noisy light field;
 * @param LF_basic_num: will contain the numerator (accumulation buffer of estimates) of the denoised light field after the 1st step;
 * @param LF_basic_den: will contain the denominator (aggregation weights) of the denoised light field after the 1st step;
 * @param LF_SAI_mask: indicate if sub-aperture image is empty (0) or not (1)
 * @param procSAI: keep track of processed SAIs;
 * @param cst, pst: s, t indexes of the center SAI (c) and the SAI being processed (p);
 * @param awidth, aheight: angular size of the LF;
 * @param width, height, chnls : size of noisy SAI, with borders;
 * @param nSimHard: size of the boundary around the center noisy SAI;
 * @param nDispHard: size of the boundary around the non-center noisy SAI;
 * @param kHard: patch size;
 * @param NHard: maximum number of similar patches for self-similarities search (center SAI);
 * @param pHard: processing step;
 * @param useSD: if true, use weight based on the standard variation
 *        of the 3D group for the first step, otherwise use the number
 *        of non-zero coefficients after Hard-thresholding;
 * @param color_space: Transformation from RGB to YUV. Allowed
 *        values are RGB (do nothing), YUV, YCBCR and OPP;
 * @param tau_2D, tau_4D, tau_5D: successive transform to apply on every 5D group
 *        Allowed values are ID, DCT and BIOR / ID, DCT, SADCT / and HADAMARD, HAAR and DCT respectively;
 * @param plan_*d_*: for convenience. Used by fftw for dct transform;
 * @param BM_elapsed_secs: Block matching processing time.
 *
 * @return none.
 **/
void bm5d_1st_step(
    const float sigma
,   float lambdaHard5D
,   vector<vector<float> > &LF_noisy
,   vector<vector<float> > &LF_basic_num
,   vector<vector<float> > &LF_basic_den
,   vector<unsigned> &LF_SAI_mask
,   const vector<unsigned> &procSAI
,   unsigned cst
,   unsigned pst
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned nSimHard
,   const unsigned nDispHard
,   const unsigned kHard
,   const unsigned NHard
,   const unsigned pHard
,   const bool     useSD
,   const unsigned color_space
,   const unsigned tau_2D
,   const unsigned tau_4D
,   const unsigned tau_5D
,   fftwf_plan * plan_2d_for_1
,   fftwf_plan * plan_2d_for_2
,   fftwf_plan * plan_2d_for_3
,   fftwf_plan * plan_2d_inv
,   fftwf_plan * plan_4d
,   fftwf_plan * plan_4d_inv
,   fftwf_plan * plan_4d_sa
,   fftwf_plan * plan_4d_sa_inv
,   fftwf_plan * plan_5d
,   fftwf_plan * plan_5d_inv
,   float &BM_elapsed_secs
){
    //! Check if OpenMP is used or if number of cores of the computer is > 1
    unsigned nb_threads = 1;

#ifdef _OPENMP
    nb_threads = omp_get_num_procs();

    //! In case where the number of processors isn't a power of 2
    if (!power_of_2(nb_threads))
        nb_threads = closest_power_of_2(nb_threads);
#endif

    //! Estimatation of sigma on each channel
    vector<float> sigma_table(chnls);
    if (estimate_sigma(sigma, sigma_table, chnls, color_space) != EXIT_SUCCESS)
        return;

    //! Parameters initialization
    const unsigned asize = awidth * aheight;
    const unsigned nHard = nSimHard + nDispHard;
    const float    tauMatch = (chnls == 1 ? 3.f : 1.f) * (sigma_table[0] < 35.0f ? 3000 : 5000); //! threshold used to determinate similarity between patches

    //! Find row and column index of patches to be processed
    vector<unsigned> row_ind, column_ind, column_ind_all_row; 
    vector<vector<unsigned> > column_ind_per_row;
    const unsigned max_nb_cols = ind_size(width - kHard + 1, nHard, pHard);
    if(pst == cst)
    {
        ind_initialize(row_ind,            height - kHard + 1, nHard, pHard);
        ind_initialize(column_ind_all_row, width  - kHard + 1, nHard, pHard);
    }
    else
        ind_initialize(row_ind, column_ind_per_row, height - kHard + 1, width  - kHard + 1, width, nHard, pHard, kHard, LF_basic_den[pst]);

    if(row_ind.size() == 0) //! check if there are any patches to process
    {
        //cout << "\t\tNo patch to denoise here." << endl;
        BM_elapsed_secs = 0.0f;
        return;
    }

    //! Compute current iteration basic estimate
    vector<vector<float> > LF_basic;
    if(compute_LF_estimate(LF_SAI_mask, LF_basic_num, LF_basic_den, LF_noisy, LF_basic, asize) != EXIT_SUCCESS)
        return;


    //! Initializations
    const unsigned kHard_2 = kHard * kHard;
    vector<vector<float> > group_4D_table(asize, vector<float>(chnls * kHard_2 * NHard * column_ind.size()));
    vector<float> wx_r_table;
    wx_r_table.reserve(chnls * column_ind.size());
    vector<unsigned> skip_table(column_ind.size(), 0);
    vector<float> tmp_5d(NHard);

    //! Preprocessing (KaiserWindow, Threshold, DCT normalization, ...)
    vector<float> kaiser_window(kHard_2);
    vector<float> coef_norm(kHard_2);
    vector<float> coef_norm_inv(kHard_2);
    preProcess(kaiser_window, coef_norm, coef_norm_inv, kHard);

    vector<float> coef_norm_4d(asize);
    vector<float> coef_norm_inv_4d(asize);
    if(tau_4D == DCT || tau_4D == SADCT)
        preProcess_4d(coef_norm_4d, coef_norm_inv_4d, awidth, aheight);

    const unsigned max_dct_size = awidth > aheight ? awidth : aheight;
    vector<vector<float> > coef_norm_4d_sa(max_dct_size-1);
    vector<vector<float> > coef_norm_inv_4d_sa(max_dct_size-1);
    if(tau_4D == SADCT)
        preProcess_4d_sadct(coef_norm_4d_sa, coef_norm_inv_4d_sa, max_dct_size);
    const bool sa_use_mean_sub = false;

    vector<float> coef_norm_5d(NHard), coef_norm_inv_5d(NHard);
    
    //! Preprocessing of Bior table
    vector<float> lpd, hpd, lpr, hpr;
    bior15_coef(lpd, hpd, lpr, hpr);

    //! Adapat threshold depending on number of DCTs
    if(tau_2D == ID && tau_4D == DCT)
        lambdaHard5D /= (float)(SQRT2);

    //! Precompute Bloc-Matching between the SAI being processed and all other SAIs
    timestamp_t start_BM = get_timestamp();
    vector<vector<vector<unsigned> > > patch_table_LF(asize);
    vector<vector<unsigned> > shape_table_LF(asize); //! To store sadct shapes
    if(pst == cst)
    {
        for(unsigned st = 0; st < asize; st++)
        {
            if(st == pst)
                precompute_BM(patch_table_LF[pst], LF_basic[pst], width, height, kHard, NHard, nHard, nSimHard, pHard, tauMatch);
            else if(LF_SAI_mask[st])
                if(precompute_BM_stereo(patch_table_LF[st], shape_table_LF[st], LF_basic[pst], LF_basic[st], width, height, kHard, nHard, nDispHard, 1, tauMatch) != EXIT_SUCCESS)
                    return;
        }
    }
    else
    {
        for(unsigned st = 0; st < asize; st++)
        {
            if(st == pst)
                precompute_BM(patch_table_LF[pst], LF_basic[pst], width, height, kHard, NHard, nHard, nSimHard, tauMatch, row_ind, column_ind_per_row);
            else if(LF_SAI_mask[st])
                if(precompute_BM_stereo(patch_table_LF[st], shape_table_LF[st], LF_basic[pst], LF_basic[st], width, height, kHard, nHard, nDispHard, tauMatch, row_ind, column_ind_per_row) != EXIT_SUCCESS)
                    return;
        }
    }
    timestamp_t end_BM = get_timestamp();
    BM_elapsed_secs = float(end_BM-start_BM) / 1000000.0f;
    
    vector<vector<float> > table_2D(asize, vector<float>((2 * nHard + 1) * width * chnls * kHard_2, 0.0f));

    if(pst == cst)
    {
        //! Loop on i_r
        for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
        {
            const unsigned i_r = row_ind[ind_i];

            //! Update of table_2D
            if (tau_2D == ID)
            {
                for(unsigned st = 0; st < asize; st++)
                    if(LF_SAI_mask[st])
                        id_2d_process(table_2D[st], LF_noisy[st], nHard, width, height, chnls, kHard, i_r);
            }
            else if (tau_2D == DCT)
            {
                for(unsigned st = 0; st < asize; st++)
                    if(LF_SAI_mask[st])
                        dct_2d_process(table_2D[st], LF_noisy[st], plan_2d_for_1, plan_2d_for_2, nHard,
                                       width, height, chnls, kHard, i_r, pHard, coef_norm,
                                       row_ind[0], row_ind.back());
            }
            else if (tau_2D == BIOR)
            {
                for(unsigned st = 0; st < asize; st++)
                    if(LF_SAI_mask[st])
                        bior_2d_process(table_2D[st], LF_noisy[st], nHard, width, height, chnls,
                                        kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);
            }           
            column_ind = column_ind_all_row;

            //! Update row variables
            wx_r_table.clear();
            for(unsigned st = 0; st < asize; st++)
                group_4D_table[st].clear();
            
            //! Loop on j_r
            for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
            {
                //! Initialization
                const unsigned j_r = column_ind[ind_j];
                const unsigned k_r = i_r * width + j_r;

                //! Number of similar patches
                const unsigned nSx_r = patch_table_LF[pst][k_r].size();
                //! Build the 4D groups
                vector<vector<float> > group_4D(nSx_r, vector<float>(chnls * asize * kHard_2, 0.0f));
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned ind_pst = patch_table_LF[pst][k_r][n];
                    for (unsigned c = 0; c < chnls; c++)
                        for(unsigned st = 0; st < asize; st++)
                            if(LF_SAI_mask[st])
                            {
                                const unsigned ind_st = ((st == pst) ? ind_pst : patch_table_LF[st][ind_pst][0]) + (nHard - i_r) * width; //! index is in Table_2D, not SAI !
                                for (unsigned k = 0; k < kHard_2; k++)
                                    group_4D[n][st + k * asize + c * kHard_2 * asize] =
                                        table_2D[st][k + ind_st * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                }

                //! Get mask for SADCT
                vector<unsigned> shape_mask(asize), shape_idx(asize), shape_mask_col(asize, 0), shape_idx_col(asize), shape_mask_dct(asize, 0);
                bool use_sadct = false;
                if(tau_4D == SADCT)
                {
                    unsigned shape_size = 0;
                    //! Fill mask
                    for (unsigned st = 0; st < asize; st++)
                    {
                        shape_mask[st] = st == pst ? 1 : LF_SAI_mask[st] ? shape_table_LF[st][k_r] : 0;
                        shape_size += shape_mask[st];
                    }
                    
                    for(unsigned s = 0; s < aheight; s++)
                    {
                        unsigned r_idx = 0;
                        for(unsigned t = 0; t < awidth; t++)
                            if(shape_mask[s * awidth + t])
                                shape_idx[s * awidth + r_idx++] = t;
                    }

                    use_sadct = shape_size == asize ? false : true; //! Only use sadct when shape is not equal to the patch (square)
                }

                //! Remove patches mean before SADCT
                vector<vector<float> > group_4D_mean(nSx_r, vector<float>(chnls * kHard_2, 0.0f));
                if(use_sadct && sa_use_mean_sub)
                {
                    for (unsigned n = 0; n < nSx_r; n++)
                        for (unsigned c = 0; c < chnls; c++)
                            for (unsigned k = 0; k < kHard_2; k++)
                            {
                                //! Compute mean
                                float patch_mean = 0.0f;
                                float patch_size = 0.0f;
                                for (unsigned st = 0; st < asize; st++)
                                    if(shape_mask[st])
                                    {
                                        patch_mean += group_4D[n][k * asize + c * kHard_2 * asize + st];
                                        patch_size++;
                                    }
                                patch_mean /= patch_size;

                                //! Remove mean
                                for (unsigned st = 0; st < asize; st++)
                                    group_4D[n][k * asize + c * kHard_2 * asize + st] -= patch_mean;

                                //! Store mean
                                group_4D_mean[n][k + c * kHard_2] = patch_mean;
                            }
                }
                                 
                //! Compute 4D DCT, if tau_4D == ID, no need to process
                if (tau_4D == DCT || (tau_4D == SADCT && !use_sadct))
                    for (unsigned n = 0; n < nSx_r; n++)
                        dct_4d_process(group_4D[n], plan_4d, awidth, aheight, chnls, kHard, coef_norm_4d);
                else if(tau_4D == SADCT || use_sadct)
                    for (unsigned n = 0; n < nSx_r; n++)
                        sadct_4d_process(group_4D[n], shape_mask, shape_idx, shape_mask_col, shape_idx_col, shape_mask_dct,
                                         plan_4d_sa, awidth, aheight, chnls, kHard, coef_norm_4d_sa);

                //! Build the 5D groups
                vector<vector<float> > group_5D(kHard_2, vector<float>(chnls * nSx_r * asize, 0.0f));
                for (unsigned c = 0; c < chnls; c++)
                    for (unsigned n = 0; n < nSx_r; n++)
                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            for (unsigned st = 0; st < asize; st++)
                                group_5D[pq][n + st * nSx_r + c * asize * nSx_r] =
                                    group_4D[n][pq * asize + st + c * asize * kHard_2];

                //! HT filtering of the 5D groups
                vector<float> weight_table(chnls, 0.0f);
                if(!use_sadct)
                {
                    if(tau_5D == HADAMARD) //! Hadamard-Walsh transform
                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            ht_filtering_hadamard_5d(group_5D[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, lambdaHard5D, weight_table);
                    else if(tau_5D == HAAR)
                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            ht_filtering_haar_5d(group_5D[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, lambdaHard5D, weight_table);
                    else if(tau_5D == DCT)
                    {
                        preProcess_5d(coef_norm_5d, coef_norm_inv_5d, nSx_r);
                        allocate_plan_1d(plan_5d,     nSx_r, FFTW_REDFT10, asize * chnls);
                        allocate_plan_1d(plan_5d_inv, nSx_r, FFTW_REDFT01, asize * chnls);

                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            ht_filtering_dct_5d(group_5D[pq], plan_5d, plan_5d_inv, coef_norm_5d, coef_norm_inv_5d, nSx_r, awidth, aheight, chnls, sigma_table, lambdaHard5D, weight_table);
                    }
                }
                else
                {
                    if(tau_5D == HADAMARD) //! Hadamard-Walsh transform
                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            ht_filtering_hadamard_5d(group_5D[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, lambdaHard5D, weight_table, shape_mask_dct);
                    else
                    if(tau_5D == HAAR)
                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            ht_filtering_haar_5d(group_5D[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, lambdaHard5D, weight_table, shape_mask_dct);
                    else if(tau_5D == DCT)
                    {
                        preProcess_5d(coef_norm_5d, coef_norm_inv_5d, nSx_r);
                        allocate_plan_1d(plan_5d,     nSx_r, FFTW_REDFT10, asize * chnls);
                        allocate_plan_1d(plan_5d_inv, nSx_r, FFTW_REDFT01, asize * chnls);

                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            ht_filtering_dct_5d(group_5D[pq], plan_5d, plan_5d_inv, coef_norm_5d, coef_norm_inv_5d,
                            nSx_r, awidth, aheight, chnls, sigma_table, lambdaHard5D, weight_table, shape_mask_dct);
                    }
                }

                //! 5D weighting using Standard Deviation
                if (useSD)
                    sd_weighting_5d(group_5D, kHard_2, nSx_r, awidth, aheight, chnls, weight_table);
                else
                {
                    //! Weight for aggregation
                    for (unsigned c = 0; c < chnls; c++)
                        weight_table[c] = (weight_table[c] > 0.0f ? (sigma_table[c] > 0.0 ? 1.0f / (float)(sigma_table[c] * sigma_table[c] * weight_table[c]) : 
                                                                                            1.0f / (float)(weight_table[c])) : 1.0f);
                }

                //! Return filtered 5D group to 4D group
                for (unsigned c = 0; c < chnls; c++)
                    for (unsigned n = 0; n < nSx_r; n++)
                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            for (unsigned st = 0; st < asize; st++)
                                group_4D[n][pq * asize + st + c * asize * kHard_2] = 
                                    group_5D[pq][n + st * nSx_r + c * asize * nSx_r];

                //! Inverse 4D DCT
                if(tau_4D == ID)
                {
                    vector<vector<float> > tmp_group_4D(nSx_r, vector<float>(chnls * asize * kHard_2, 0.0f));
                    for (unsigned n = 0; n < nSx_r; n++)
                        for (unsigned c = 0; c < chnls; c++)
                        {
                            const unsigned dc = c * asize * kHard_2;
                            for (unsigned pq = 0; pq < kHard_2; pq++)
                                for (unsigned st = 0; st < asize; st++)
                                    tmp_group_4D[n][dc + st * kHard_2 + pq] = group_4D[n][dc + pq * asize + st];
                        }
                    group_4D = tmp_group_4D;
                }
                else if (tau_4D == DCT || (tau_4D == SADCT && !use_sadct))
                    for (unsigned n = 0; n < nSx_r; n++)
                        dct_4d_inverse(group_4D[n], plan_4d_inv, awidth, aheight, chnls, kHard, coef_norm_inv_4d);
                else if(tau_4D == SADCT || use_sadct)
                    for (unsigned n = 0; n < nSx_r; n++)
                        sadct_4d_inverse(group_4D[n], shape_mask, shape_idx, shape_mask_col, shape_idx_col,
                                         plan_4d_sa_inv, awidth, aheight, chnls, kHard, coef_norm_inv_4d_sa);


                //! Add patches mean after SADCT
                if(use_sadct && sa_use_mean_sub)
                {
                    for (unsigned n = 0; n < nSx_r; n++)
                        for (unsigned c = 0; c < chnls; c++)
                            for (unsigned k = 0; k < kHard_2; k++)
                            {
                                //! Get mean
                                float patch_mean = group_4D_mean[n][k + c * kHard_2];

                                //! Add mean
                                for (unsigned st = 0; st < asize; st++)
                                    group_4D[n][k + st * kHard_2 + c * kHard_2 * asize] += patch_mean;
                            }
                }
                
                //! Save the 4D group. The DCT 2D inverse will be done after.
                for (unsigned st = 0; st < asize; st++)
                    if(LF_SAI_mask[st])
                        for (unsigned c = 0; c < chnls; c++)
                            for (unsigned n = 0; n < nSx_r; n++)
                                for (unsigned k = 0; k < kHard_2; k++)
                                    group_4D_table[st].push_back(group_4D[n][k + st * kHard_2 + c * kHard_2 * asize]);

                //! Save weighting
                for (unsigned c = 0; c < chnls; c++)
                    wx_r_table.push_back(weight_table[c]);

            } //! End of loop on j_r

            for (unsigned st = 0; st < asize; st++)
            {
                if(!procSAI[st])
                {
                    //! Apply 2D inverse transform, if tau_2D == ID, no need to inverse
                    if (tau_2D == DCT)
                        dct_2d_inverse(group_4D_table[st], kHard, NHard * chnls * column_ind.size(),
                                    coef_norm_inv, plan_2d_inv);
                    else if (tau_2D == BIOR)
                        bior_2d_inverse(group_4D_table[st], kHard, lpr, hpr);

                    //! Registration of the weighted estimation
                    unsigned dec = 0;
                    for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
                    {
                        const unsigned j_r   = column_ind[ind_j];
                        const unsigned k_r   = i_r * width + j_r;
                        const unsigned nSx_r = patch_table_LF[pst][k_r].size();

                        if(tau_4D != SADCT || st == pst || shape_table_LF[st][k_r])
                        {
                            for (unsigned c = 0; c < chnls; c++)
                            {
                                for (unsigned n = 0; n < nSx_r; n++)
                                {
                                    const unsigned ind_pst = patch_table_LF[pst][k_r][n];
                                    const unsigned ind_st  = ((st == pst) ? ind_pst : patch_table_LF[st][ind_pst][0]) + c * width * height;

                                    for (unsigned p = 0; p < kHard; p++)
                                        for (unsigned q = 0; q < kHard; q++)
                                        {
                                            const unsigned ind = ind_st + p * width + q;
                                            LF_basic_num[st][ind] += kaiser_window[p * kHard + q]
                                                            * wx_r_table[c + ind_j * chnls]
                                                            * group_4D_table[st][p * kHard + q + n * kHard_2 + c * kHard_2 * nSx_r + dec];
                                            LF_basic_den[st][ind] += kaiser_window[p * kHard + q]
                                                            * wx_r_table[c + ind_j * chnls];
                                        }
                                }
                            }
                        }
                        dec += nSx_r * chnls * kHard_2;
                    }
                }
            }
        } //! End of loop on i_r
    }
    else //! SAI being processed is not central, many patches do not need to be processed
    {
        //! Loop on i_r
        for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
        {
            const unsigned i_r = row_ind[ind_i];

            column_ind = column_ind_per_row[ind_i];

            //! Update row variables
            wx_r_table.clear();
            for(unsigned st = 0; st < asize; st++)
                group_4D_table[st].clear();
            
            //! Loop on j_r
            for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
            {
                //! Initialization
                const unsigned j_r = column_ind[ind_j];
                const unsigned k_r = i_r * width + j_r;

                //! Update of table_2D
                if (tau_2D == ID)
                {
                    for(unsigned st = 0; st < asize; st++)
                        if(LF_SAI_mask[st])
                            id_2d_process(table_2D[st], LF_noisy[st], nHard, width, height, chnls, kHard, i_r, j_r);
                }
                else if (tau_2D == DCT)
                {
                    for(unsigned st = 0; st < asize; st++)
                        if(LF_SAI_mask[st])
                            dct_2d_process(table_2D[st], LF_noisy[st], plan_2d_for_3, plan_2d_for_2, nHard,
                                           width, height, chnls, kHard, i_r, j_r, coef_norm);
                }
                else if (tau_2D == BIOR)
                {
                    for(unsigned st = 0; st < asize; st++)
                        if(LF_SAI_mask[st])
                            bior_2d_process(table_2D[st], LF_noisy[st], nHard, width, height, chnls, kHard, i_r, j_r, lpd, hpd);
                }

                //! Number of similar patches
                const unsigned nSx_r = patch_table_LF[pst][k_r].size();
                //! Build the 4D groups
                vector<vector<float> > group_4D(nSx_r, vector<float>(chnls * asize * kHard_2, 0.0f));
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned ind_pst = patch_table_LF[pst][k_r][n];
                    for (unsigned c = 0; c < chnls; c++)
                        for(unsigned st = 0; st < asize; st++)
                            if(LF_SAI_mask[st])
                            {
                                const unsigned ind_st = ((st == pst) ? ind_pst : patch_table_LF[st][ind_pst][0]) + (nHard - i_r) * width; //! index is in Table_2D, not SAI !
                                for (unsigned k = 0; k < kHard_2; k++)
                                    group_4D[n][st + k * asize + c * kHard_2 * asize] =
                                        table_2D[st][k + ind_st * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                }

                //! Get mask for SADCT
                vector<unsigned> shape_mask(asize), shape_idx(asize), shape_mask_col(asize, 0), shape_idx_col(asize), shape_mask_dct(asize, 0);
                bool use_sadct = false;
                if(tau_4D == SADCT)
                {
                    unsigned shape_size = 0;
                    //! Fill mask
                    for (unsigned st = 0; st < asize; st++)
                    {
                        shape_mask[st] = st == pst ? 1 : LF_SAI_mask[st] ? shape_table_LF[st][k_r] : 0;
                        shape_size += shape_mask[st];
                    }
                    //! Get corresponding row index
                    for(unsigned s = 0; s < aheight; s++)
                    {
                        unsigned r_idx = 0;
                        for(unsigned t = 0; t < awidth; t++)
                            if(shape_mask[s * awidth + t])
                                shape_idx[s * awidth + r_idx++] = t;
                    }

                    use_sadct = shape_size == asize ? false : true; //! Only use sadct when shape is not equal to the patch (square)
                }

                //! Remove patches mean before SADCT
                vector<vector<float> > group_4D_mean(nSx_r, vector<float>(chnls * kHard_2, 0.0f));
                if(use_sadct && sa_use_mean_sub)
                {
                    for (unsigned n = 0; n < nSx_r; n++)
                        for (unsigned c = 0; c < chnls; c++)
                            for (unsigned k = 0; k < kHard_2; k++)
                            {
                                //! Compute mean
                                float patch_mean = 0.0f;
                                float patch_size = 0.0f;
                                for (unsigned st = 0; st < asize; st++)
                                    if(shape_mask[st])
                                    {
                                        patch_mean += group_4D[n][k * asize + c * kHard_2 * asize + st];
                                        patch_size++;
                                    }
                                patch_mean /= patch_size;

                                //! Remove mean
                                for (unsigned st = 0; st < asize; st++)
                                    group_4D[n][k * asize + c * kHard_2 * asize + st] -= patch_mean;

                                //! Store mean
                                group_4D_mean[n][k + c * kHard_2] = patch_mean;
                            }
                }

                //! Compute 4D DCT, if tau_4D == ID, no need to process
                if (tau_4D == DCT || (tau_4D == SADCT && !use_sadct))
                    for (unsigned n = 0; n < nSx_r; n++)
                        dct_4d_process(group_4D[n], plan_4d, awidth, aheight, chnls, kHard, coef_norm_4d);
                else if(tau_4D == SADCT || use_sadct)
                    for (unsigned n = 0; n < nSx_r; n++)
                        sadct_4d_process(group_4D[n], shape_mask, shape_idx, shape_mask_col, shape_idx_col, shape_mask_dct,
                                         plan_4d_sa, awidth, aheight, chnls, kHard, coef_norm_4d_sa);

                //! Build the 5D groups
                vector<vector<float> > group_5D(kHard_2, vector<float>(chnls * nSx_r * asize, 0.0f));
                for (unsigned c = 0; c < chnls; c++)
                    for (unsigned n = 0; n < nSx_r; n++)
                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            for (unsigned st = 0; st < asize; st++)
                                group_5D[pq][n + st * nSx_r + c * asize * nSx_r] =
                                    group_4D[n][pq * asize + st + c * asize * kHard_2];

                //! HT filtering of the 5D groups
                vector<float> weight_table(chnls, 0.0f);
                if(!use_sadct)
                {
                    if(tau_5D == HADAMARD) //! Hadamard-Walsh transform
                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            ht_filtering_hadamard_5d(group_5D[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, lambdaHard5D, weight_table);
                    else if(tau_5D == HAAR)
                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            ht_filtering_haar_5d(group_5D[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, lambdaHard5D, weight_table);
                    else if(tau_5D == DCT)
                    {
                        preProcess_5d(coef_norm_5d, coef_norm_inv_5d, nSx_r);
                        allocate_plan_1d(plan_5d,     nSx_r, FFTW_REDFT10, asize * chnls);
                        allocate_plan_1d(plan_5d_inv, nSx_r, FFTW_REDFT01, asize * chnls);

                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            ht_filtering_dct_5d(group_5D[pq], plan_5d, plan_5d_inv, coef_norm_5d, coef_norm_inv_5d, nSx_r, awidth, aheight, chnls, sigma_table, lambdaHard5D, weight_table);
                    }
                }
                else
                {
                    if(tau_5D == HADAMARD) //! Hadamard-Walsh transform
                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            ht_filtering_hadamard_5d(group_5D[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, lambdaHard5D, weight_table, shape_mask_dct);
                    else
                    if(tau_5D == HAAR)
                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            ht_filtering_haar_5d(group_5D[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, lambdaHard5D, weight_table, shape_mask_dct);
                    else if(tau_5D == DCT)
                    {
                        preProcess_5d(coef_norm_5d, coef_norm_inv_5d, nSx_r);
                        allocate_plan_1d(plan_5d,     nSx_r, FFTW_REDFT10, asize * chnls);
                        allocate_plan_1d(plan_5d_inv, nSx_r, FFTW_REDFT01, asize * chnls);

                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            ht_filtering_dct_5d(group_5D[pq], plan_5d, plan_5d_inv, coef_norm_5d, coef_norm_inv_5d, 
                            nSx_r, awidth, aheight, chnls, sigma_table, lambdaHard5D, weight_table, shape_mask_dct);
                    }
                }

                //! 5D weighting using Standard Deviation
                if (useSD)
                    sd_weighting_5d(group_5D, kHard_2, nSx_r, awidth, aheight, chnls, weight_table);
                else
                {
                    //! Weight for aggregation
                    for (unsigned c = 0; c < chnls; c++)
                        weight_table[c] = (weight_table[c] > 0.0f ? (sigma_table[c] > 0.0 ? 1.0f / (float)(sigma_table[c] * sigma_table[c] * weight_table[c]) : 
                                                                                            1.0f / (float)(weight_table[c])) : 1.0f);
                }
                
                //! Return filtered 5D group to 4D group
                for (unsigned c = 0; c < chnls; c++)
                    for (unsigned n = 0; n < nSx_r; n++)
                        for(unsigned pq = 0; pq < kHard_2; pq++)
                            for (unsigned st = 0; st < asize; st++)
                                group_4D[n][pq * asize + st + c * asize * kHard_2] = 
                                    group_5D[pq][n + st * nSx_r + c * asize * nSx_r];


                //! Inverse 4D DCT
                if(tau_4D == ID)
                {
                    vector<vector<float> > tmp_group_4D(nSx_r, vector<float>(chnls * asize * kHard_2, 0.0f));
                    for (unsigned n = 0; n < nSx_r; n++)
                        for (unsigned c = 0; c < chnls; c++)
                        {
                            const unsigned dc = c * asize * kHard_2;
                            for (unsigned pq = 0; pq < kHard_2; pq++)
                                for (unsigned st = 0; st < asize; st++)
                                    tmp_group_4D[n][dc + st * kHard_2 + pq] = group_4D[n][dc + pq * asize + st];
                        }
                    group_4D = tmp_group_4D;
                }
                else if (tau_4D == DCT || (tau_4D == SADCT && !use_sadct))
                    for (unsigned n = 0; n < nSx_r; n++)
                        dct_4d_inverse(group_4D[n], plan_4d_inv, awidth, aheight, chnls, kHard, coef_norm_inv_4d);
                else if(tau_4D == SADCT || use_sadct)
                    for (unsigned n = 0; n < nSx_r; n++)
                        sadct_4d_inverse(group_4D[n], shape_mask, shape_idx, shape_mask_col, shape_idx_col,
                                         plan_4d_sa_inv, awidth, aheight, chnls, kHard, coef_norm_inv_4d_sa);

                //! Add patches mean after SADCT
                if(use_sadct && sa_use_mean_sub)
                {
                    for (unsigned n = 0; n < nSx_r; n++)
                        for (unsigned c = 0; c < chnls; c++)
                            for (unsigned k = 0; k < kHard_2; k++)
                            {
                                //! Get mean
                                float patch_mean = group_4D_mean[n][k + c * kHard_2];

                                //! Add mean
                                for (unsigned st = 0; st < asize; st++)
                                    group_4D[n][k + st * kHard_2 + c * kHard_2 * asize] += patch_mean;
                            }
                }
                    
                //! Save the 4D group. The DCT 2D inverse will be done after.
                for (unsigned st = 0; st < asize; st++)
                    if(LF_SAI_mask[st])
                        for (unsigned c = 0; c < chnls; c++)
                            for (unsigned n = 0; n < nSx_r; n++)
                                for (unsigned k = 0; k < kHard_2; k++)
                                    group_4D_table[st].push_back(group_4D[n][k + st * kHard_2 + c * kHard_2 * asize]);

                //! Save weighting
                for (unsigned c = 0; c < chnls; c++)
                    wx_r_table.push_back(weight_table[c]);

                
            } //! End of loop on j_r
    
            for (unsigned st = 0; st < asize; st++)
            {
                if(!procSAI[st])
                {
                    //!  Apply 2D inverse transform, if tau_2D == ID, no need to inverse
                    if (tau_2D == DCT)
                        dct_2d_inverse(group_4D_table[st], kHard, NHard * chnls * max_nb_cols,
                                    coef_norm_inv, plan_2d_inv);
                    else if (tau_2D == BIOR)
                        bior_2d_inverse(group_4D_table[st], kHard, lpr, hpr);

                    //! Registration of the weighted estimation
                    unsigned dec = 0;
                    for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
                    {
                        const unsigned j_r   = column_ind[ind_j];
                        const unsigned k_r   = i_r * width + j_r;
                        const unsigned nSx_r = patch_table_LF[pst][k_r].size();

                        if(tau_4D != SADCT || st == pst || shape_table_LF[st][k_r])
                        {
                            for (unsigned c = 0; c < chnls; c++)
                            {
                                for (unsigned n = 0; n < nSx_r; n++)
                                {
                                    const unsigned ind_pst = patch_table_LF[pst][k_r][n];
                                    const unsigned ind_st  = ((st == pst) ? ind_pst : patch_table_LF[st][ind_pst][0]) + c * width * height;

                                    for (unsigned p = 0; p < kHard; p++)
                                        for (unsigned q = 0; q < kHard; q++)
                                        {
                                            const unsigned ind = ind_st + p * width + q;
                                            LF_basic_num[st][ind] += kaiser_window[p * kHard + q]
                                                            * wx_r_table[c + ind_j * chnls]
                                                            * group_4D_table[st][p * kHard + q + n * kHard_2 + c * kHard_2 * nSx_r + dec];
                                            LF_basic_den[st][ind] += kaiser_window[p * kHard + q]
                                                            * wx_r_table[c + ind_j * chnls];
                                        }
                                }
                            }
                        }
                        dec += nSx_r * chnls * kHard_2;
                    }
                }
            }
        } //! End of loop on i_r
    }
}


/**
 * @brief Run the final process of LFBM5D (Wiener step). The results
 *        numerator and denominator are contained in LF_denoised_num and LF_denoised_den respectively.
 *        Each image in the light field has boundary, which
 *        are here only for block-matching and do not need to be
 *        denoised.
 *
 * @param sigma: value of assumed noise of the image to denoise;
 * @param LF_noisy: noisy light field;
 * @param LF_basic: light field estimate after 1st step;
 * @param LF_denoised_num: will contain the numerator (accumulation buffer of estimates) of the denoised light field after the 2nd step;
 * @param LF_denoised_den: will contain the denominator (aggregation weights) of the denoised light field after the 2nd step;
 * @param LF_SAI_mask: indicate if sub-aperture image is empty (0) or not (1);
 * @param procSAI: keep track of processed SAIs;
 * @param cst, pst: s, t indexes of the center SAI (c*) and the SAI being processed (p*);
 * @param awidth, aheight: angular size of the LF;
 * @param width, height, chnls : size of noisy SAI, with borders;
 * @param nSimWien: size of the boundary around the center noisy SAI;
 * @param nDispWien: size of the boundary around the non-center noisy SAI;
 * @param kWien: patch size;
 * @param NWien: maximum number of similar patches for self-similarities search (center SAI);
 * @param pWien: processing step;
 * @param useSD: if true, use weight based on the standard variation
 *        of the 3D group for the first step, otherwise use the number
 *        of non-zero coefficients after Hard-thresholding;
 * @param color_space: Transformation from RGB to YUV. Allowed
 *        values are RGB (do nothing), YUV, YCBCR and OPP;
 * @param tau_2D, tau_4D, tau_5D: successive transform to apply on every 5D group
 *        Allowed values are ID, DCT and BIOR / ID, DCT, SADCT / and HADAMARD, HAAR and DCT respectively;
 * @param plan_*d*: for convenience. Used by fftw for dct transform;
 * @param BM_elapsed_secs: Block matching processing time.
 *
 * @return none.
 **/
void bm5d_2nd_step(
    const float sigma
,   vector<vector<float> > &LF_noisy
,   vector<vector<float> > &LF_basic
,   vector<vector<float> > &LF_denoised_num
,   vector<vector<float> > &LF_denoised_den
,   vector<unsigned> &LF_SAI_mask
,   const vector<unsigned> &procSAI
,   unsigned cst
,   unsigned pst
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned nSimWien
,   const unsigned nDispWien
,   const unsigned kWien
,   const unsigned NWien
,   const unsigned pWien
,   const bool     useSD
,   const unsigned color_space
,   const unsigned tau_2D
,   const unsigned tau_4D
,   const unsigned tau_5D
,   fftwf_plan * plan_2d_for_1
,   fftwf_plan * plan_2d_for_2
,   fftwf_plan * plan_2d_for_3
,   fftwf_plan * plan_2d_inv
,   fftwf_plan * plan_4d
,   fftwf_plan * plan_4d_inv
,   fftwf_plan * plan_4d_sa
,   fftwf_plan * plan_4d_sa_inv
,   fftwf_plan * plan_5d
,   fftwf_plan * plan_5d_inv
,   float &BM_elapsed_secs
){
        //! Check if OpenMP is used or if number of cores of the computer is > 1
    unsigned nb_threads = 1;

#ifdef _OPENMP
    nb_threads = omp_get_num_procs();

    //! In case where the number of processors isn't a power of 2
    if (!power_of_2(nb_threads))
        nb_threads = closest_power_of_2(nb_threads);
#endif

    //! Estimatation of sigma on each channel
    vector<float> sigma_table(chnls);
    if (estimate_sigma(sigma, sigma_table, chnls, color_space) != EXIT_SUCCESS)
        return;

    //! Parameters initialization
    const unsigned asize = awidth * aheight;
    const unsigned nWien = nSimWien + nDispWien;
    const float    tauMatch = (chnls == 1 ? 3.f : 1.f) * (sigma_table[0] < 35.0f ? 2000 : 5000); //! threshold used to determinate similarity between patches

    //! Find row and column index of patches to be processed
    vector<unsigned> row_ind, column_ind, column_ind_all_row; 
    vector<vector<unsigned> > column_ind_per_row;
    const unsigned max_nb_cols = ind_size(width - kWien + 1, nWien, pWien);
    if(pst == cst)
    {
        ind_initialize(row_ind,            height - kWien + 1, nWien, pWien);
        ind_initialize(column_ind_all_row, width  - kWien + 1, nWien, pWien);
    }
    else
        ind_initialize(row_ind, column_ind_per_row, height - kWien + 1, width  - kWien + 1, width, nWien, pWien, kWien, LF_denoised_den[pst]);

    if(row_ind.size() == 0) //! check if there are any patches to process
    {
        BM_elapsed_secs = 0.0f;
        return;
    }

    //! Compute current iteration basic estimate
    vector<vector<float> > LF_denoised;
    if(compute_LF_estimate(LF_SAI_mask, LF_denoised_num, LF_denoised_den, LF_basic, LF_denoised, asize) != EXIT_SUCCESS)
        return;


    //! Initializations
    const unsigned kWien_2 = kWien * kWien;
    vector<vector<float> > group_4D_table(asize, vector<float>(chnls * kWien_2 * NWien * column_ind.size()));
    vector<float> wx_r_table;
    wx_r_table.reserve(chnls * column_ind.size());
    vector<unsigned> skip_table(column_ind.size(), 0);
    vector<float> tmp_5d(NWien);

    //! Preprocessing (KaiserWindow, Threshold, DCT normalization, ...)
    vector<float> kaiser_window(kWien_2);
    vector<float> coef_norm(kWien_2);
    vector<float> coef_norm_inv(kWien_2);
    preProcess(kaiser_window, coef_norm, coef_norm_inv, kWien);

    vector<float> coef_norm_4d(asize);
    vector<float> coef_norm_inv_4d(asize);
    if(tau_4D == DCT || tau_4D == SADCT)
        preProcess_4d(coef_norm_4d, coef_norm_inv_4d, awidth, aheight);

    const unsigned max_dct_size = awidth > aheight ? awidth : aheight;
    vector<vector<float> > coef_norm_4d_sa(max_dct_size-1);
    vector<vector<float> > coef_norm_inv_4d_sa(max_dct_size-1);
    if(tau_4D == SADCT)
        preProcess_4d_sadct(coef_norm_4d_sa, coef_norm_inv_4d_sa, max_dct_size);
    const bool sa_use_mean_sub = false;

    vector<float> coef_norm_5d(NWien), coef_norm_inv_5d(NWien);
    
    //! Preprocessing of Bior table
    vector<float> lpd, hpd, lpr, hpr;
    bior15_coef(lpd, hpd, lpr, hpr);

    //! Precompute Bloc-Matching between the SAI being processed and all other SAIs
    timestamp_t start_BM = get_timestamp();
    vector<vector<vector<unsigned> > > patch_table_LF(asize);
    vector<vector<unsigned> > shape_table_LF(asize); //! To store sadct shapes
    if(pst == cst)
    {
        for(unsigned st = 0; st < asize; st++)
        {
            if(st == pst)
                precompute_BM(patch_table_LF[pst], LF_denoised[pst], width, height, kWien, NWien, nWien, nSimWien, pWien, tauMatch);
            else if(LF_SAI_mask[st])
                if(precompute_BM_stereo(patch_table_LF[st], shape_table_LF[st], LF_denoised[pst], LF_denoised[st], width, height, kWien, nWien, nDispWien, 1, tauMatch) != EXIT_SUCCESS)
                    return;
        }
    }
    else
    {
        for(unsigned st = 0; st < asize; st++)
        {
            if(st == pst)
                precompute_BM(patch_table_LF[pst], LF_denoised[pst], width, height, kWien, NWien, nWien, nSimWien, tauMatch, row_ind, column_ind_per_row);
            else if(LF_SAI_mask[st])
                if(precompute_BM_stereo(patch_table_LF[st], shape_table_LF[st], LF_denoised[pst], LF_denoised[st], width, height, kWien, nWien, nDispWien, tauMatch, row_ind, column_ind_per_row) != EXIT_SUCCESS)
                    return;
        }
    }
    timestamp_t end_BM = get_timestamp();
    BM_elapsed_secs = float(end_BM-start_BM) / 1000000.0f;
    
    vector<vector<float> > table_2D_org(asize, vector<float>((2 * nWien + 1) * width * chnls * kWien_2, 0.0f));
    vector<vector<float> > table_2D_est(asize, vector<float>((2 * nWien + 1) * width * chnls * kWien_2, 0.0f));

    if(pst == cst)
    {
        //! Loop on i_r
        for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
        {
            const unsigned i_r = row_ind[ind_i];

            //! Update of table_2D
            if (tau_2D == ID)
            {
                for(unsigned st = 0; st < asize; st++)
                    if(LF_SAI_mask[st])
                    {
                        id_2d_process(table_2D_org[st], LF_noisy[st], nWien, width, height, chnls, kWien, i_r);
                        id_2d_process(table_2D_est[st], LF_basic[st], nWien, width, height, chnls, kWien, i_r);
                    }
            }
            else if (tau_2D == DCT)
            {
                for(unsigned st = 0; st < asize; st++)
                    if(LF_SAI_mask[st])
                    {
                        dct_2d_process(table_2D_org[st], LF_noisy[st], plan_2d_for_1, plan_2d_for_2, nWien,
                                       width, height, chnls, kWien, i_r, pWien, coef_norm,
                                       row_ind[0], row_ind.back());
                        dct_2d_process(table_2D_est[st], LF_basic[st], plan_2d_for_1, plan_2d_for_2, nWien,
                                       width, height, chnls, kWien, i_r, pWien, coef_norm,
                                       row_ind[0], row_ind.back());
                    }
            }
            else if (tau_2D == BIOR)
            {
                for(unsigned st = 0; st < asize; st++)
                    if(LF_SAI_mask[st])
                    {
                        bior_2d_process(table_2D_org[st], LF_noisy[st], nWien, width, height, chnls,
                                        kWien, i_r, pWien, row_ind[0], row_ind.back(), lpd, hpd);
                        bior_2d_process(table_2D_est[st], LF_basic[st], nWien, width, height, chnls,
                                        kWien, i_r, pWien, row_ind[0], row_ind.back(), lpd, hpd);
                    }
            }           
            column_ind = column_ind_all_row;

            //! Update row variables
            wx_r_table.clear();
            for(unsigned st = 0; st < asize; st++)
                group_4D_table[st].clear();
            
            //! Loop on j_r
            for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
            {
                //! Initialization
                const unsigned j_r = column_ind[ind_j];
                const unsigned k_r = i_r * width + j_r;

                //! Number of similar patches
                const unsigned nSx_r = patch_table_LF[pst][k_r].size();
                //! Build the 4D groups
                vector<vector<float> > group_4D_org(nSx_r, vector<float>(chnls * asize * kWien_2, 0.0f));
                vector<vector<float> > group_4D_est(nSx_r, vector<float>(chnls * asize * kWien_2, 0.0f));
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned ind_pst = patch_table_LF[pst][k_r][n];
                    for (unsigned c = 0; c < chnls; c++)
                        for(unsigned st = 0; st < asize; st++)
                            if(LF_SAI_mask[st])
                            {
                                const unsigned ind_st = ((st == pst) ? ind_pst : patch_table_LF[st][ind_pst][0]) + (nWien - i_r) * width; //! index is in Table_2D, not SAI !
                                for (unsigned k = 0; k < kWien_2; k++)
                                {
                                    group_4D_org[n][st + k * asize + c * kWien_2 * asize] =
                                        table_2D_org[st][k + ind_st * kWien_2 + c * kWien_2 * (2 * nWien + 1) * width];
                                    group_4D_est[n][st + k * asize + c * kWien_2 * asize] =
                                        table_2D_est[st][k + ind_st * kWien_2 + c * kWien_2 * (2 * nWien + 1) * width];
                                }
                            }
                }

                //! Get mask for SADCT
                vector<unsigned> shape_mask(asize), shape_idx(asize), shape_mask_col(asize, 0), shape_idx_col(asize), shape_mask_dct(asize, 0);
                bool use_sadct = false;
                if(tau_4D == SADCT)
                {
                    unsigned shape_size = 0;
                    //! Fill mask
                    for (unsigned st = 0; st < asize; st++)
                    {
                        shape_mask[st] = st == pst ? 1 : LF_SAI_mask[st] ? shape_table_LF[st][k_r] : 0;
                        shape_size += shape_mask[st];
                    }
                    for(unsigned s = 0; s < aheight; s++)
                    {
                        unsigned r_idx = 0;
                        for(unsigned t = 0; t < awidth; t++)
                            if(shape_mask[s * awidth + t])
                                shape_idx[s * awidth + r_idx++] = t;
                    }

                    use_sadct = shape_size == asize ? false : true; //! Only use sadct when shape is not equal to the patch (square)
                }

                //! Remove patches mean before SADCT
                vector<vector<float> > group_4D_mean_est(nSx_r, vector<float>(chnls * kWien_2, 0.0f));
                if(use_sadct && sa_use_mean_sub)
                {
                    for (unsigned n = 0; n < nSx_r; n++)
                        for (unsigned c = 0; c < chnls; c++)
                            for (unsigned k = 0; k < kWien_2; k++)
                            {
                                //! Compute mean
                                float patch_mean_org = 0.0f;
                                float patch_mean_est = 0.0f;
                                float patch_size = 0.0f;
                                for (unsigned st = 0; st < asize; st++)
                                    if(shape_mask[st])
                                    {
                                        patch_mean_org += group_4D_org[n][k * asize + c * kWien_2 * asize + st];
                                        patch_mean_est += group_4D_est[n][k * asize + c * kWien_2 * asize + st];
                                        patch_size++;
                                    }
                                patch_mean_org /= patch_size;
                                patch_mean_est /= patch_size;

                                //! Remove mean
                                for (unsigned st = 0; st < asize; st++)
                                {
                                    group_4D_org[n][k * asize + c * kWien_2 * asize + st] -= patch_mean_org;
                                    group_4D_est[n][k * asize + c * kWien_2 * asize + st] -= patch_mean_est;
                                }

                                //! Store mean
                                group_4D_mean_est[n][k + c * kWien_2] = patch_mean_est;
                            }
                }
                                 
                //! Compute 4D DCT, if tau_4D == ID, no need to process
                if (tau_4D == DCT || (tau_4D == SADCT && !use_sadct))
                    for (unsigned n = 0; n < nSx_r; n++)
                    {
                        dct_4d_process(group_4D_org[n], plan_4d, awidth, aheight, chnls, kWien, coef_norm_4d);
                        dct_4d_process(group_4D_est[n], plan_4d, awidth, aheight, chnls, kWien, coef_norm_4d);
                    }
                else if(tau_4D == SADCT || use_sadct)
                    for (unsigned n = 0; n < nSx_r; n++)
                    {
                        sadct_4d_process(group_4D_org[n], shape_mask, shape_idx, shape_mask_col, shape_idx_col, shape_mask_dct,
                                         plan_4d_sa, awidth, aheight, chnls, kWien, coef_norm_4d_sa);
                        sadct_4d_process(group_4D_est[n], shape_mask, shape_idx, shape_mask_col, shape_idx_col, shape_mask_dct,
                                         plan_4d_sa, awidth, aheight, chnls, kWien, coef_norm_4d_sa);
                    }

                //! Build the 5D groups
                vector<vector<float> > group_5D_org(kWien_2, vector<float>(chnls * nSx_r * asize, 0.0f));
                vector<vector<float> > group_5D_est(kWien_2, vector<float>(chnls * nSx_r * asize, 0.0f));
                for (unsigned c = 0; c < chnls; c++)
                    for (unsigned n = 0; n < nSx_r; n++)
                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            for (unsigned st = 0; st < asize; st++)
                            {
                                group_5D_org[pq][n + st * nSx_r + c * asize * nSx_r] =
                                    group_4D_org[n][pq * asize + st + c * asize * kWien_2];
                                group_5D_est[pq][n + st * nSx_r + c * asize * nSx_r] =
                                    group_4D_est[n][pq * asize + st + c * asize * kWien_2];
                            }

                //! HT filtering of the 5D groups
                vector<float> weight_table(chnls, 0.0f);
                if(!use_sadct)
                {
                    if(tau_5D == HADAMARD) //! Hadamard-Walsh transform
                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            wiener_filtering_hadamard_5d(group_5D_org[pq], group_5D_est[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, weight_table);
                    else if(tau_5D == HAAR)
                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            wiener_filtering_haar_5d(group_5D_org[pq], group_5D_est[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, weight_table);
                    else if(tau_5D == DCT)
                    {
                        preProcess_5d(coef_norm_5d, coef_norm_inv_5d, nSx_r);
                        allocate_plan_1d(plan_5d,     nSx_r, FFTW_REDFT10, asize * chnls);
                        allocate_plan_1d(plan_5d_inv, nSx_r, FFTW_REDFT01, asize * chnls);

                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            wiener_filtering_dct_5d(group_5D_org[pq], group_5D_est[pq], plan_5d, plan_5d_inv, 
                                                    coef_norm_5d, coef_norm_inv_5d, nSx_r, awidth, aheight, chnls, sigma_table, weight_table);
                    }
                }
                else
                {
                    if(tau_5D == HADAMARD) //! Hadamard-Walsh transform
                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            wiener_filtering_hadamard_5d(group_5D_org[pq], group_5D_est[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, weight_table, shape_mask_dct);
                    else
                    if(tau_5D == HAAR)
                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            wiener_filtering_haar_5d(group_5D_org[pq], group_5D_est[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, weight_table, shape_mask_dct);
                    else if(tau_5D == DCT)
                    {
                        preProcess_5d(coef_norm_5d, coef_norm_inv_5d, nSx_r);
                        allocate_plan_1d(plan_5d,     nSx_r, FFTW_REDFT10, asize * chnls);
                        allocate_plan_1d(plan_5d_inv, nSx_r, FFTW_REDFT01, asize * chnls);

                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            wiener_filtering_dct_5d(group_5D_org[pq], group_5D_est[pq], plan_5d, plan_5d_inv, 
                                                    coef_norm_5d, coef_norm_inv_5d, nSx_r, awidth, aheight, chnls, sigma_table, weight_table, shape_mask_dct);
                    }
                }

                //! 5D weighting using Standard Deviation
                if (useSD)
                    sd_weighting_5d(group_5D_est, kWien_2, nSx_r, awidth, aheight, chnls, weight_table);
                else
                {
                    //! Weight for aggregation
                    for (unsigned c = 0; c < chnls; c++)
                        weight_table[c] = (weight_table[c] > 0.0f ? (sigma_table[c] > 0.0 ? 1.0f / (float)(sigma_table[c] * sigma_table[c] * weight_table[c]) : 
                                                                                            1.0f / (float)(weight_table[c])) : 1.0f);
                }

                //! Return filtered 5D group to 4D group
                for (unsigned c = 0; c < chnls; c++)
                    for (unsigned n = 0; n < nSx_r; n++)
                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            for (unsigned st = 0; st < asize; st++)
                                group_4D_est[n][pq * asize + st + c * asize * kWien_2] = 
                                    group_5D_est[pq][n + st * nSx_r + c * asize * nSx_r];


                //! Inverse 4D DCT
                if(tau_4D == ID)
                {
                    vector<vector<float> > tmp_group_4D(nSx_r, vector<float>(chnls * asize * kWien_2, 0.0f));
                    for (unsigned n = 0; n < nSx_r; n++)
                        for (unsigned c = 0; c < chnls; c++)
                        {
                            const unsigned dc = c * asize * kWien_2;
                            for (unsigned pq = 0; pq < kWien_2; pq++)
                                for (unsigned st = 0; st < asize; st++)
                                    tmp_group_4D[n][dc + st * kWien_2 + pq] = group_4D_est[n][dc + pq * asize + st];
                        }
                    group_4D_est = tmp_group_4D;
                }
                else if (tau_4D == DCT || (tau_4D == SADCT && !use_sadct))
                    for (unsigned n = 0; n < nSx_r; n++)
                        dct_4d_inverse(group_4D_est[n], plan_4d_inv, awidth, aheight, chnls, kWien, coef_norm_inv_4d);
                else if(tau_4D == SADCT || use_sadct)
                    for (unsigned n = 0; n < nSx_r; n++)
                        sadct_4d_inverse(group_4D_est[n], shape_mask, shape_idx, shape_mask_col, shape_idx_col,
                                         plan_4d_sa_inv, awidth, aheight, chnls, kWien, coef_norm_inv_4d_sa);


                //! Add patches mean after SADCT
                if(use_sadct && sa_use_mean_sub)
                {
                    for (unsigned n = 0; n < nSx_r; n++)
                        for (unsigned c = 0; c < chnls; c++)
                            for (unsigned k = 0; k < kWien_2; k++)
                            {
                                //! Get mean
                                float patch_mean = group_4D_mean_est[n][k + c * kWien_2];

                                //! Add mean
                                for (unsigned st = 0; st < asize; st++)
                                    group_4D_est[n][k + st * kWien_2 + c * kWien_2 * asize] += patch_mean;
                            }
                }
                
                //! Save the 4D group. The DCT 2D inverse will be done after.
                for (unsigned st = 0; st < asize; st++)
                    if(LF_SAI_mask[st])
                        for (unsigned c = 0; c < chnls; c++)
                            for (unsigned n = 0; n < nSx_r; n++)
                                for (unsigned k = 0; k < kWien_2; k++)
                                    group_4D_table[st].push_back(group_4D_est[n][k + st * kWien_2 + c * kWien_2 * asize]);

                //! Save weighting
                for (unsigned c = 0; c < chnls; c++)
                    wx_r_table.push_back(weight_table[c]);

            } //! End of loop on j_r

            for (unsigned st = 0; st < asize; st++)
            {
                if(!procSAI[st])
                {
                    //! Apply 2D inverse transform, if tau_2D == ID, no need to inverse
                    if (tau_2D == DCT)
                        dct_2d_inverse(group_4D_table[st], kWien, NWien * chnls * column_ind.size(),
                                    coef_norm_inv, plan_2d_inv);
                    else if (tau_2D == BIOR)
                        bior_2d_inverse(group_4D_table[st], kWien, lpr, hpr);

                    //! Registration of the weighted estimation
                    unsigned dec = 0;
                    for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
                    {
                        const unsigned j_r   = column_ind[ind_j];
                        const unsigned k_r   = i_r * width + j_r;
                        const unsigned nSx_r = patch_table_LF[pst][k_r].size();

                        if(tau_4D != SADCT || st == pst || shape_table_LF[st][k_r])
                        {
                            for (unsigned c = 0; c < chnls; c++)
                            {
                                for (unsigned n = 0; n < nSx_r; n++)
                                {
                                    const unsigned ind_pst = patch_table_LF[pst][k_r][n];
                                    const unsigned ind_st  = ((st == pst) ? ind_pst : patch_table_LF[st][ind_pst][0]) + c * width * height;

                                    for (unsigned p = 0; p < kWien; p++)
                                        for (unsigned q = 0; q < kWien; q++)
                                        {
                                            const unsigned ind = ind_st + p * width + q;
                                            LF_denoised_num[st][ind] += kaiser_window[p * kWien + q]
                                                            * wx_r_table[c + ind_j * chnls]
                                                            * group_4D_table[st][p * kWien + q + n * kWien_2 + c * kWien_2 * nSx_r + dec];
                                            LF_denoised_den[st][ind] += kaiser_window[p * kWien + q]
                                                            * wx_r_table[c + ind_j * chnls];
                                        }
                                }
                            }
                        }
                        dec += nSx_r * chnls * kWien_2;
                    }
                }
            }
        } //! End of loop on i_r
    }
    else //! SAI being processed is not central, many patches do not need to be processed
    {
        //! Loop on i_r
        for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
        {
            const unsigned i_r = row_ind[ind_i];

            column_ind = column_ind_per_row[ind_i];

            //! Update row variables
            wx_r_table.clear();
            for(unsigned st = 0; st < asize; st++)
                group_4D_table[st].clear();
            
            //! Loop on j_r
            for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
            {
                //! Initialization
                const unsigned j_r = column_ind[ind_j];
                const unsigned k_r = i_r * width + j_r;

                //! Update of table_2D
                if (tau_2D == ID)
                {
                    for(unsigned st = 0; st < asize; st++)
                        if(LF_SAI_mask[st])
                        {
                            id_2d_process(table_2D_org[st], LF_noisy[st], nWien, width, height, chnls, kWien, i_r, j_r);
                            id_2d_process(table_2D_est[st], LF_basic[st], nWien, width, height, chnls, kWien, i_r, j_r);
                        }
                }
                else if (tau_2D == DCT)
                {
                    for(unsigned st = 0; st < asize; st++)
                        if(LF_SAI_mask[st])
                        {
                            dct_2d_process(table_2D_org[st], LF_noisy[st], plan_2d_for_3, plan_2d_for_2, nWien,
                                           width, height, chnls, kWien, i_r, j_r, coef_norm);
                            dct_2d_process(table_2D_est[st], LF_basic[st], plan_2d_for_3, plan_2d_for_2, nWien,
                                           width, height, chnls, kWien, i_r, j_r, coef_norm);
                        }
                }
                else if (tau_2D == BIOR)
                {
                    for(unsigned st = 0; st < asize; st++)
                        if(LF_SAI_mask[st])
                        {
                            bior_2d_process(table_2D_org[st], LF_noisy[st], nWien, width, height, chnls, kWien, i_r, j_r, lpd, hpd);
                            bior_2d_process(table_2D_est[st], LF_basic[st], nWien, width, height, chnls, kWien, i_r, j_r, lpd, hpd);
                        }
                }

                //! Number of similar patches
                const unsigned nSx_r = patch_table_LF[pst][k_r].size();
                //! Build the 4D groups
                vector<vector<float> > group_4D_org(nSx_r, vector<float>(chnls * asize * kWien_2, 0.0f));
                vector<vector<float> > group_4D_est(nSx_r, vector<float>(chnls * asize * kWien_2, 0.0f));
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned ind_pst = patch_table_LF[pst][k_r][n];
                    for (unsigned c = 0; c < chnls; c++)
                        for(unsigned st = 0; st < asize; st++)
                            if(LF_SAI_mask[st])
                            {
                                const unsigned ind_st = ((st == pst) ? ind_pst : patch_table_LF[st][ind_pst][0]) + (nWien - i_r) * width; //! index is in Table_2D, not SAI !
                                for (unsigned k = 0; k < kWien_2; k++)
                                {
                                    group_4D_org[n][st + k * asize + c * kWien_2 * asize] =
                                        table_2D_org[st][k + ind_st * kWien_2 + c * kWien_2 * (2 * nWien + 1) * width];
                                    group_4D_est[n][st + k * asize + c * kWien_2 * asize] =
                                        table_2D_est[st][k + ind_st * kWien_2 + c * kWien_2 * (2 * nWien + 1) * width];
                                }
                            }
                }

                //! Get mask for SADCT
                vector<unsigned> shape_mask(asize), shape_idx(asize), shape_mask_col(asize, 0), shape_idx_col(asize), shape_mask_dct(asize, 0);
                bool use_sadct = false;
                if(tau_4D == SADCT)
                {
                    unsigned shape_size = 0;
                    //! Fill mask
                    for (unsigned st = 0; st < asize; st++)
                    {
                        shape_mask[st] = st == pst ? 1 : LF_SAI_mask[st] ? shape_table_LF[st][k_r] : 0;
                        shape_size += shape_mask[st];
                    }
                    //! Get corresponding row index
                    for(unsigned s = 0; s < aheight; s++)
                    {
                        unsigned r_idx = 0;
                        for(unsigned t = 0; t < awidth; t++)
                            if(shape_mask[s * awidth + t])
                                shape_idx[s * awidth + r_idx++] = t;
                    }

                    use_sadct = shape_size == asize ? false : true; //! Only use sadct when shape is not equal to the patch (square)
                }

                //! Remove patches mean before SADCT
                vector<vector<float> > group_4D_mean_est(nSx_r, vector<float>(chnls * kWien_2, 0.0f));
                if(use_sadct && sa_use_mean_sub)
                {
                    for (unsigned n = 0; n < nSx_r; n++)
                        for (unsigned c = 0; c < chnls; c++)
                            for (unsigned k = 0; k < kWien_2; k++)
                            {
                                //! Compute mean
                                float patch_mean_org = 0.0f;
                                float patch_mean_est = 0.0f;
                                float patch_size = 0.0f;
                                for (unsigned st = 0; st < asize; st++)
                                    if(shape_mask[st])
                                    {
                                        patch_mean_org += group_4D_org[n][k * asize + c * kWien_2 * asize + st];
                                        patch_mean_est += group_4D_est[n][k * asize + c * kWien_2 * asize + st];
                                        patch_size++;
                                    }
                                patch_mean_org /= patch_size;
                                patch_mean_est /= patch_size;

                                //! Remove mean
                                for (unsigned st = 0; st < asize; st++)
                                {
                                    group_4D_org[n][k * asize + c * kWien_2 * asize + st] -= patch_mean_org;
                                    group_4D_est[n][k * asize + c * kWien_2 * asize + st] -= patch_mean_est;
                                }

                                //! Store mean
                                group_4D_mean_est[n][k + c * kWien_2] = patch_mean_est;
                            }
                }
                
                //! Compute 4D DCT, if tau_4D == ID, no need to process
                if (tau_4D == DCT || (tau_4D == SADCT && !use_sadct))
                    for (unsigned n = 0; n < nSx_r; n++)
                    {
                        dct_4d_process(group_4D_org[n], plan_4d, awidth, aheight, chnls, kWien, coef_norm_4d);
                        dct_4d_process(group_4D_est[n], plan_4d, awidth, aheight, chnls, kWien, coef_norm_4d);
                    }
                else if(tau_4D == SADCT || use_sadct)
                    for (unsigned n = 0; n < nSx_r; n++)
                    {
                        sadct_4d_process(group_4D_org[n], shape_mask, shape_idx, shape_mask_col, shape_idx_col, shape_mask_dct,
                                         plan_4d_sa, awidth, aheight, chnls, kWien, coef_norm_4d_sa);
                        sadct_4d_process(group_4D_est[n], shape_mask, shape_idx, shape_mask_col, shape_idx_col, shape_mask_dct,
                                         plan_4d_sa, awidth, aheight, chnls, kWien, coef_norm_4d_sa);
                    }

                //! Build the 5D groups
                vector<vector<float> > group_5D_org(kWien_2, vector<float>(chnls * nSx_r * asize, 0.0f));
                vector<vector<float> > group_5D_est(kWien_2, vector<float>(chnls * nSx_r * asize, 0.0f));
                for (unsigned c = 0; c < chnls; c++)
                    for (unsigned n = 0; n < nSx_r; n++)
                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            for (unsigned st = 0; st < asize; st++)
                            {
                                group_5D_org[pq][n + st * nSx_r + c * asize * nSx_r] =
                                    group_4D_org[n][pq * asize + st + c * asize * kWien_2];
                                group_5D_est[pq][n + st * nSx_r + c * asize * nSx_r] =
                                    group_4D_est[n][pq * asize + st + c * asize * kWien_2];
                            }

                //! HT filtering of the 5D groups
                vector<float> weight_table(chnls, 0.0f);
                if(!use_sadct)
                {
                    if(tau_5D == HADAMARD) //! Hadamard-Walsh transform
                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            wiener_filtering_hadamard_5d(group_5D_org[pq], group_5D_est[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, weight_table);
                    else if(tau_5D == HAAR)
                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            wiener_filtering_haar_5d(group_5D_org[pq], group_5D_est[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, weight_table);
                    else if(tau_5D == DCT)
                    {
                        preProcess_5d(coef_norm_5d, coef_norm_inv_5d, nSx_r);
                        allocate_plan_1d(plan_5d,     nSx_r, FFTW_REDFT10, asize * chnls);
                        allocate_plan_1d(plan_5d_inv, nSx_r, FFTW_REDFT01, asize * chnls);

                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            wiener_filtering_dct_5d(group_5D_org[pq], group_5D_est[pq], plan_5d, plan_5d_inv, 
                                                    coef_norm_5d, coef_norm_inv_5d, nSx_r, awidth, aheight, chnls, sigma_table, weight_table);
                    }
                }
                else
                {
                    if(tau_5D == HADAMARD) //! Hadamard-Walsh transform
                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            wiener_filtering_hadamard_5d(group_5D_org[pq], group_5D_est[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, weight_table, shape_mask_dct);
                    else
                    if(tau_5D == HAAR)
                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            wiener_filtering_haar_5d(group_5D_org[pq], group_5D_est[pq], tmp_5d, nSx_r, awidth, aheight, chnls, sigma_table, weight_table, shape_mask_dct);
                    else if(tau_5D == DCT)
                    {
                        preProcess_5d(coef_norm_5d, coef_norm_inv_5d, nSx_r);
                        allocate_plan_1d(plan_5d,     nSx_r, FFTW_REDFT10, asize * chnls);
                        allocate_plan_1d(plan_5d_inv, nSx_r, FFTW_REDFT01, asize * chnls);

                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            wiener_filtering_dct_5d(group_5D_org[pq], group_5D_est[pq], plan_5d, plan_5d_inv, 
                                                    coef_norm_5d, coef_norm_inv_5d, nSx_r, awidth, aheight, chnls, sigma_table, weight_table, shape_mask_dct);
                    }
                }

                //! 5D weighting using Standard Deviation
                if (useSD)
                    sd_weighting_5d(group_5D_est, kWien_2, nSx_r, awidth, aheight, chnls, weight_table);
                else
                {
                    //! Weight for aggregation
                    for (unsigned c = 0; c < chnls; c++)
                        weight_table[c] = (weight_table[c] > 0.0f ? (sigma_table[c] > 0.0 ? 1.0f / (float)(sigma_table[c] * sigma_table[c] * weight_table[c]) : 
                                                                                            1.0f / (float)(weight_table[c])) : 1.0f);
                }


                //! Return filtered 5D group to 4D group
                for (unsigned c = 0; c < chnls; c++)
                    for (unsigned n = 0; n < nSx_r; n++)
                        for(unsigned pq = 0; pq < kWien_2; pq++)
                            for (unsigned st = 0; st < asize; st++)
                                group_4D_est[n][pq * asize + st + c * asize * kWien_2] = 
                                    group_5D_est[pq][n + st * nSx_r + c * asize * nSx_r];


                //! Inverse 4D DCT
                if(tau_4D == ID)
                {
                    vector<vector<float> > tmp_group_4D(nSx_r, vector<float>(chnls * asize * kWien_2, 0.0f));
                    for (unsigned n = 0; n < nSx_r; n++)
                        for (unsigned c = 0; c < chnls; c++)
                        {
                            const unsigned dc = c * asize * kWien_2;
                            for (unsigned pq = 0; pq < kWien_2; pq++)
                                for (unsigned st = 0; st < asize; st++)
                                    tmp_group_4D[n][dc + st * kWien_2 + pq] = group_4D_est[n][dc + pq * asize + st];
                        }
                    group_4D_est = tmp_group_4D;
                }
                else if (tau_4D == DCT || (tau_4D == SADCT && !use_sadct))
                    for (unsigned n = 0; n < nSx_r; n++)
                        dct_4d_inverse(group_4D_est[n], plan_4d_inv, awidth, aheight, chnls, kWien, coef_norm_inv_4d);
                else if(tau_4D == SADCT || use_sadct)
                    for (unsigned n = 0; n < nSx_r; n++)
                        sadct_4d_inverse(group_4D_est[n], shape_mask, shape_idx, shape_mask_col, shape_idx_col,
                                         plan_4d_sa_inv, awidth, aheight, chnls, kWien, coef_norm_inv_4d_sa);


                //! Add patches mean after SADCT
                if(use_sadct && sa_use_mean_sub)
                {
                    for (unsigned n = 0; n < nSx_r; n++)
                        for (unsigned c = 0; c < chnls; c++)
                            for (unsigned k = 0; k < kWien_2; k++)
                            {
                                //! Get mean
                                float patch_mean = group_4D_mean_est[n][k + c * kWien_2];

                                //! Add mean
                                for (unsigned st = 0; st < asize; st++)
                                    group_4D_est[n][k + st * kWien_2 + c * kWien_2 * asize] += patch_mean;
                            }
                }
                
                //! Save the 4D group. The DCT 2D inverse will be done after.
                for (unsigned st = 0; st < asize; st++)
                    if(LF_SAI_mask[st])
                        for (unsigned c = 0; c < chnls; c++)
                            for (unsigned n = 0; n < nSx_r; n++)
                                for (unsigned k = 0; k < kWien_2; k++)
                                    group_4D_table[st].push_back(group_4D_est[n][k + st * kWien_2 + c * kWien_2 * asize]);

                //! Save weighting
                for (unsigned c = 0; c < chnls; c++)
                    wx_r_table.push_back(weight_table[c]);

                
            } //! End of loop on j_r
    
            for (unsigned st = 0; st < asize; st++)
            {
                if(!procSAI[st])
                {
                    //!  Apply 2D inverse transform, if tau_2D == ID, no need to inverse
                    if (tau_2D == DCT)
                        dct_2d_inverse(group_4D_table[st], kWien, NWien * chnls * max_nb_cols,
                                    coef_norm_inv, plan_2d_inv);
                    else if (tau_2D == BIOR)
                        bior_2d_inverse(group_4D_table[st], kWien, lpr, hpr);

                    //! Registration of the weighted estimation
                    unsigned dec = 0;
                    for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
                    {
                        const unsigned j_r   = column_ind[ind_j];
                        const unsigned k_r   = i_r * width + j_r;
                        const unsigned nSx_r = patch_table_LF[pst][k_r].size();

                        if(tau_4D != SADCT || st == pst || shape_table_LF[st][k_r])
                        {
                            for (unsigned c = 0; c < chnls; c++)
                            {
                                for (unsigned n = 0; n < nSx_r; n++)
                                {
                                    const unsigned ind_pst = patch_table_LF[pst][k_r][n];
                                    const unsigned ind_st  = ((st == pst) ? ind_pst : patch_table_LF[st][ind_pst][0]) + c * width * height;

                                    for (unsigned p = 0; p < kWien; p++)
                                        for (unsigned q = 0; q < kWien; q++)
                                        {
                                            const unsigned ind = ind_st + p * width + q;
                                            LF_denoised_num[st][ind] += kaiser_window[p * kWien + q]
                                                            * wx_r_table[c + ind_j * chnls]
                                                            * group_4D_table[st][p * kWien + q + n * kWien_2 + c * kWien_2 * nSx_r + dec];
                                            LF_denoised_den[st][ind] += kaiser_window[p * kWien + q]
                                                            * wx_r_table[c + ind_j * chnls];
                                        }
                                }
                            }
                        }
                        dec += nSx_r * chnls * kWien_2;
                    }
                }
            }
        } //! End of loop on i_r
    }
}

/*********************************************************************************************************************************************************************************/
/*********************************************************************************************************************************************************************************/
/** -------------- **/
/** - Transforms - **/
/** -------------- **/
/*********************************************************************************************************************************************************************************/
/*********************************************************************************************************************************************************************************/

/**
 * @brief Just collect patches in table without any transform
 *
 * @param table_2D : will contain all chosen patches;
 * @param img : image on which the 2d DCT will be processed;
 * @param nHW : size of the boundary around img;
 * @param width, height, chnls: size of img;
 * @param kHW : size of patches (kHW x kHW);
 * @param i_r: current index of the reference patches;
 **/
void id_2d_process(
    vector<float> &table_2D
,   vector<float> const& img
,   const unsigned nHW
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned kHW
,   const unsigned i_r
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;

    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * width * height;
        const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
        for (unsigned i = 0; i < 2 * nHW + 1; i++)
            for (unsigned j = 0; j < width - kHW; j++)
                for (unsigned p = 0; p < kHW; p++)
                    for (unsigned q = 0; q < kHW; q++)
                        table_2D[p * kHW + q + dc_p + (i * width + j) * kHW_2] =
                            img[dc + (i_r + i - nHW + p) * width + j + q];
    }
}

/**
 * @brief Just collect patches in table without any transform
 *
 * @param table_2D : will contain all chosen patches;
 * @param img : image on which the 2d DCT will be processed;
 * @param nHW : size of the boundary around img;
 * @param width, height, chnls: size of img;
 * @param kHW : size of patches (kHW x kHW);
 * @param i_r, j_r: current index of the reference patches;
 **/
void id_2d_process(
    vector<float> &table_2D
,   vector<float> const& img
,   const unsigned nHW
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned kHW
,   const unsigned i_r
,   const unsigned j_r
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;

    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * width * height;
        const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
        for (unsigned i = 0; i < 2 * nHW + 1; i++)
            for (unsigned j = 0; j < 2 * nHW + 1; j++)
                for (unsigned p = 0; p < kHW; p++)
                    for (unsigned q = 0; q < kHW; q++)
                        table_2D[p * kHW + q + dc_p + (i * width + j_r + j - nHW) * kHW_2] =
                            img[dc + (i_r + i - nHW + p) * width + j_r + j - nHW + q];
    }
}

/**
 * @brief Precompute a 2D DCT transform on all patches contained in
 *        a part of the image.
 *
 * @param DCT_table_2D : will contain the 2d DCT transform for all
 *        chosen patches;
 * @param img : image on which the 2d DCT will be processed;
 * @param plan_1, plan_2 : for convenience. Used by fftw;
 * @param nHW : size of the boundary around img;
 * @param width, height, chnls: size of img;
 * @param kHW : size of patches (kHW x kHW);
 * @param i_r, j_r: current index of the reference patches;
 * @param coef_norm : normalization coefficients of the 2D DCT.
 **/
void dct_2d_process(
    vector<float> &DCT_table_2D
,   vector<float> const& img
,   fftwf_plan * plan_1
,   fftwf_plan * plan_2
,   const unsigned nHW
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned kHW
,   const unsigned i_r
,   const unsigned j_r
,   vector<float> const& coef_norm
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;
    const unsigned size = chnls * kHW_2 * (2 * nHW + 1) * (2 * nHW + 1);

    //! Allocating Memory
    float* vec = (float*) fftwf_malloc(size * sizeof(float));
    float* dct = (float*) fftwf_malloc(size * sizeof(float));

    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * width * height;
        const unsigned dc_p = c * kHW_2 * (2 * nHW + 1) * (2 * nHW + 1);
        for (unsigned i = 0; i < 2 * nHW + 1; i++)
            for (unsigned j = 0; j < 2 * nHW + 1; j++)
                for (unsigned p = 0; p < kHW; p++)
                    for (unsigned q = 0; q < kHW; q++)
                        vec[p * kHW + q + dc_p + (i * (2 * nHW + 1) + j) * kHW_2] =
                            img[dc + (i_r + i - nHW + p) * width + j_r + j - nHW + q];
    }

    //! Process of all DCTs
    fftwf_execute_r2r(*plan_1, vec, dct);
    fftwf_free(vec);

    //! Getting the result
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc   = c * kHW_2 * width * (2 * nHW + 1);
        const unsigned dc_p = c * kHW_2 * (2 * nHW + 1) * (2 * nHW + 1);
        for (unsigned i = 0; i < 2 * nHW + 1; i++)
            for (unsigned j = 0; j < 2 * nHW + 1; j++)
                for (unsigned k = 0; k < kHW_2; k++)
                    DCT_table_2D[dc + (i * width + j_r + j - nHW) * kHW_2 + k] =
                        dct[dc_p + (i * (2 * nHW + 1) + j) * kHW_2 + k] * coef_norm[k];
    }
    fftwf_free(dct);
}

/**
 * @brief Precompute a 2D bior1.5 transform on all patches contained in
 *        a part of the image.
 *
 * @param bior_table_2D : will contain the 2d bior1.5 transform for all
 *        chosen patches;
 * @param img : image on which the 2d transform will be processed;
 * @param nHW : size of the boundary around img;
 * @param width, height, chnls: size of img;
 * @param kHW : size of patches (kHW x kHW). MUST BE A POWER OF 2 !!!
 * @param i_r, j_r: current index of the reference patches;
 * @param lpd : low pass filter of the forward bior1.5 2d transform;
 * @param hpd : high pass filter of the forward bior1.5 2d transform.
 **/
void bior_2d_process(
    vector<float> &bior_table_2D
,   vector<float> const& img
,   const unsigned nHW
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned kHW
,   const unsigned i_r
,   const unsigned j_r
,   vector<float> &lpd
,   vector<float> &hpd
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;

    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * width * height;
        const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
        for (unsigned i = 0; i < 2 * nHW + 1; i++)
            for (unsigned j = 0; j < 2 * nHW + 1; j++)
            {
                bior_2d_forward(img, bior_table_2D, kHW, dc +
                            (i_r + i - nHW) * width + j_r + j - nHW, width,
                            dc_p + (i * width + j_r + j - nHW) * kHW_2, lpd, hpd);
            }
    }
}

/**
 * @brief Compute the 4D DCT transform on a 4D group (2D table containing 2D tranformed patches)
 *
 * @param group_4D : contain all patches to be tranformed and will contain the 4D DCT transform;
 * @param plan : for convenience. Used by fftw;
 * @param awidth, aheight : size of the angular patches;
 * @param chnls: number of color channels;
 * @param kHW : size of patches (kHW x kHW);
 * @param coef_norm : normalization coefficients of the 2D DCT.
 **/
 void dct_4d_process(
    vector<float> &group_4D
,   fftwf_plan * plan
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   const unsigned kHW
,   vector<float> const& coef_norm
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;
    const unsigned asize = awidth * aheight;
    const unsigned size = chnls * kHW_2 * asize;

    //! Allocating Memory
    float* vec = (float*) fftwf_malloc(size * sizeof(float));
    float* dct = (float*) fftwf_malloc(size * sizeof(float));

    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * asize * kHW_2;
        for (unsigned pq = 0; pq < kHW_2; pq++)
            for (unsigned st = 0; st < asize; st++)
                vec[dc + pq * asize + st] = group_4D[dc + pq * asize + st];
    }
    
    //! Process of all DCTs
    fftwf_execute_r2r(*plan, vec, dct);
    fftwf_free(vec);

    //! Getting the result
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * asize * kHW_2;
        for (unsigned pq = 0; pq < kHW_2; pq++)
            for (unsigned st = 0; st < asize; st++)
                group_4D[dc + pq * asize + st] = dct[dc + pq * asize + st] * coef_norm[st];
    }
    fftwf_free(dct);
}

/**
 * @brief Compute the 4D inverse DCT transform on a 4D group (2D table containing 2D tranformed patches)
 *
 * @param group_4D : contain all patches to be tranformed and will contain the 4D DCT transform;
 * @param plan : for convenience. Used by fftw;
 * @param awidth, aheight : size of the angular patches;
 * @param chnls: number of color channels;
 * @param kHW : size of patches (kHW x kHW);
 * @param coef_norm_inv : normalization coefficients of the 2D DCT.
 **/
void dct_4d_inverse(
    vector<float> &group_4D
,   fftwf_plan * plan
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   const unsigned kHW
,   vector<float> const& coef_norm_inv
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;
    const unsigned asize = awidth * aheight;
    const unsigned size = chnls * kHW_2 * asize;

    //! Allocating Memory
    float* vec = (float*) fftwf_malloc(size * sizeof(float));
    float* dct = (float*) fftwf_malloc(size * sizeof(float));

    //! Normalization
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * asize * kHW_2;
        for (unsigned pq = 0; pq < kHW_2; pq++)
            for (unsigned st = 0; st < asize; st++)
                dct[dc + pq * asize + st] = group_4D[dc + pq * asize + st] * coef_norm_inv[st];
    }

    //! 2D dct inverse
    fftwf_execute_r2r(*plan, dct, vec);
    fftwf_free(dct);

    //! Getting the result + normalization
    const float coef = 1.0f / (sqrt((float) awidth) * sqrt((float) aheight) * 2.0f); //(float)(awidth * 2);
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * asize * kHW_2;
        for (unsigned pq = 0; pq < kHW_2; pq++)
            for (unsigned st = 0; st < asize; st++)
                group_4D[dc + st * kHW_2 + pq] = vec[dc + pq * asize + st] * coef;
    }
    fftwf_free(vec);
}

/**
 * @brief Compute the 4D SADCT transform on a 4D group (2D table containing 2D tranformed patches)
 *
 * @param group_4D : contain all patches to be tranformed and will contain the 4D DCT transform;
 * @param shape_mask, shape_idx : mask indicating patch pixels which have to be denoised (1) or not (0), and corresponding index for convenience;
 * @param shape_mask_col, shape_idx_col : mask indicating patch pixels which have to be denoised (1) or not (0) after applying 1D DCT on rows, and corresponding index for convenience;
 * @param shape_mask_dct : mask indicating patch pixels which have to be denoised (1) or not (0) after applying SADCT;
 * @param plan : for convenience. Used by fftw, contains all 1d transform of possible sizes;
 * @param awidth, aheight : size of the angular patches;
 * @param chnls: number of color channels;
 * @param kHW : size of patches (kHW x kHW);
 * @param coef_norm : normalization coefficients of the 2D DCT.
 **/
  void sadct_4d_process(
    vector<float> &group_4D
,   const vector<unsigned> &shape_mask
,   const vector<unsigned> &shape_idx
,   vector<unsigned> &shape_mask_col
,   vector<unsigned> &shape_idx_col
,   vector<unsigned> &shape_mask_dct
,   fftwf_plan * plan
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   const unsigned kHW
,   const vector<vector<float> > &coef_norm
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;
    const unsigned asize = awidth * aheight;
    
    //! Processing 1D DCT on rows
    for(unsigned s = 0; s < aheight; s++)
    {
        //! Get dct size
        unsigned dct_size = 0;
        for(unsigned t = 0; t < awidth; t++)
            dct_size += shape_mask[s * awidth + t];

        if(dct_size == 1) //! No transform needed
        {
            for (unsigned c = 0; c < chnls; c++)
            {
                const unsigned dc = c * asize * kHW_2;
                for (unsigned pq = 0; pq < kHW_2; pq++)
                    group_4D[dc + pq * asize + s * awidth] = group_4D[dc + pq * asize + s * awidth + shape_idx[s * awidth]];
            }
        }
        else if(dct_size > 1)
        {
            const unsigned size = chnls * kHW_2 * dct_size;

            //! Allocating Memory
            float* vec = (float*) fftwf_malloc(size * sizeof(float));
            float* dct = (float*) fftwf_malloc(size * sizeof(float));

            for (unsigned c = 0; c < chnls; c++)
            {
                const unsigned dc_sa = c * dct_size * kHW_2;
                const unsigned dc    = c * asize    * kHW_2;
                for (unsigned pq = 0; pq < kHW_2; pq++)
                    for (unsigned t = 0; t < dct_size; t++)
                        vec[dc_sa + pq * dct_size + t] = group_4D[dc + pq * asize + s * awidth + shape_idx[s * awidth + t]];
            }

            //! Process of all DCTs
            fftwf_execute_r2r(plan[dct_size-2], vec, dct);
            fftwf_free(vec);

            //! Getting the result
            for (unsigned c = 0; c < chnls; c++)
            {
                const unsigned dc_sa = c * dct_size * kHW_2;
                const unsigned dc    = c * asize    * kHW_2;
                for (unsigned pq = 0; pq < kHW_2; pq++)
                    for (unsigned t = 0; t < dct_size; t++)
                        group_4D[dc + pq * asize + s * awidth + t] = dct[dc_sa + pq * dct_size + t] * coef_norm[dct_size-2][t]; //! Coeff now start at row idx 0
            }
            fftwf_free(dct);
        }
        //! Update shape for processing on columns
        for(unsigned t = 0; t < dct_size; t++)
            shape_mask_col[s * awidth + t] = 1;
    }


    //! Update shape index for processing on columns
    for(unsigned t = 0; t < awidth; t++)
    {
        unsigned s_idx = 0;
        for(unsigned s = 0; s < aheight; s++)
            if(shape_mask_col[s * awidth + t])
                shape_idx_col[(s_idx++) * awidth + t] = s;
    }


    //! Processing 1D DCT on columns
    for(unsigned t = 0; t < awidth; t++)
    {
        //! Get dct size
        unsigned dct_size = 0;
        for(unsigned s = 0; s < aheight; s++)
            dct_size += shape_mask_col[s * awidth + t];

        if(dct_size == 1) //! No transform needed
        {
            unsigned s = 0;
            for (unsigned c = 0; c < chnls; c++)
            {
                const unsigned dc = c * asize * kHW_2;
                for (unsigned pq = 0; pq < kHW_2; pq++)
                    group_4D[dc + pq * asize + s * awidth + t] = group_4D[dc + pq * asize + shape_idx_col[s * awidth + t] * awidth + t];
            }
        }
        if(dct_size > 1)
        {
            const unsigned size = chnls * kHW_2 * dct_size;

            //! Allocating Memory
            float* vec = (float*) fftwf_malloc(size * sizeof(float));
            float* dct = (float*) fftwf_malloc(size * sizeof(float));

            for (unsigned c = 0; c < chnls; c++)
            {
                const unsigned dc_sa = c * dct_size * kHW_2;
                const unsigned dc    = c * asize    * kHW_2;
                for (unsigned pq = 0; pq < kHW_2; pq++)
                    for (unsigned s = 0; s < dct_size; s++)
                        vec[dc_sa + pq * dct_size + s] = group_4D[dc + pq * asize + shape_idx_col[s * awidth + t] * awidth + t];
            }

            //! Process of all DCTs
            fftwf_execute_r2r(plan[dct_size-2], vec, dct);
            fftwf_free(vec);

            //! Getting the result
            for (unsigned c = 0; c < chnls; c++)
            {
                const unsigned dc_sa = c * dct_size * kHW_2;
                const unsigned dc    = c * asize    * kHW_2;
                for (unsigned pq = 0; pq < kHW_2; pq++)
                    for (unsigned s = 0; s < dct_size; s++)
                        group_4D[dc + pq * asize + s * awidth + t] = dct[dc_sa + pq * dct_size + s] * coef_norm[dct_size-2][s]; //! Coeff now start at col idx 0
            }
            fftwf_free(dct);
        }   
        //! Update shape of final dct
        for(unsigned s = 0; s < dct_size; s++)
            shape_mask_dct[s * awidth + t] = 1;
    }

    //! Put non shape value to zero and normalize for hard thresholding
    const float coef = 0.5 * (float) SQRT2_INV;
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * asize * kHW_2;
        for (unsigned pq = 0; pq < kHW_2; pq++)
            for (unsigned st = 0; st < asize; st++)
                group_4D[dc + pq * asize + st] *= (float) shape_mask_dct[st] * coef;
    }
}

/**
 * @brief Compute the inverse 4D SADCT transform on a 4D group (2D table containing 2D tranformed patches)
 *
 * @param group_4D : contain all patches to be tranformed and will contain the 4D DCT transform;
 * @param shape_mask, shape_idx : mask indicating patch pixels which have to be denoised (1) or not (0), and corresponding index for convenience;
 * @param shape_mask_col, shape_idx_col : mask indicating patch pixels which have to be denoised (1) or not (0) after applying 1D DCT on rows, and corresponding index for convenience;
 * @param plan : for convenience. Used by fftw, contains all 1d transform of possible sizes;
 * @param awidth, aheight : size of the angular patches;
 * @param chnls: number of color channels;
 * @param kHW : size of patches (kHW x kHW);
 * @param coef_norm_inv : normalization coefficients of the 2D DCT.
 **/

void sadct_4d_inverse(
    vector<float> &group_4D
,   const vector<unsigned> &shape_mask
,   const vector<unsigned> &shape_idx
,   const vector<unsigned> &shape_mask_col
,   const vector<unsigned> &shape_idx_col
,   fftwf_plan * plan
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   const unsigned kHW
,   const vector<vector<float> > &coef_norm_inv
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;
    const unsigned asize = awidth * aheight;
    
    //! Processing inverse 1D DCT on columns
    for(unsigned t = 0; t < awidth; t++)
    {
        //! Get dct size
        unsigned dct_size = 0;
        for(unsigned s = 0; s < aheight; s++)
            dct_size += shape_mask_col[s * awidth + t];

        if(dct_size == 1) //! No transform needed
        {
            float coef = 2.0 * (float) SQRT2;
            for (unsigned c = 0; c < chnls; c++)
            {
                unsigned s = 0;
                const unsigned dc = c * asize * kHW_2;
                for (unsigned pq = 0; pq < kHW_2; pq++)
                    group_4D[dc + pq * asize + shape_idx_col[s * awidth + t] * awidth + t] = group_4D[dc + pq * asize + s * awidth + t] * coef;
            }
        }
        if(dct_size > 1)
        {
            const unsigned size = chnls * kHW_2 * dct_size;

            //! Allocating Memory
            float* vec = (float*) fftwf_malloc(size * sizeof(float));
            float* dct = (float*) fftwf_malloc(size * sizeof(float));

            float coef = 2.0 * (float) SQRT2;
            for (unsigned c = 0; c < chnls; c++)
            {
                const unsigned dc_sa = c * dct_size * kHW_2;
                const unsigned dc    = c * asize    * kHW_2;
                for (unsigned pq = 0; pq < kHW_2; pq++)
                    for (unsigned s = 0; s < dct_size; s++)
                        dct[dc_sa + pq * dct_size + s] = group_4D[dc + pq * asize + s * awidth + t] * coef_norm_inv[dct_size-2][s] * coef;
            }

            //! Process of all DCTs
            fftwf_execute_r2r(plan[dct_size-2], dct, vec);
            fftwf_free(dct);

            //! Getting the result
            coef = 0.5f * (float)(SQRT2_INV) / sqrt((float) dct_size);
            for (unsigned c = 0; c < chnls; c++)
            {
                const unsigned dc_sa = c * dct_size * kHW_2;
                const unsigned dc    = c * asize    * kHW_2;
                for (unsigned pq = 0; pq < kHW_2; pq++)
                    for (unsigned s = 0; s < dct_size; s++)
                        group_4D[dc + pq * asize + shape_idx_col[s * awidth + t] * awidth + t] = vec[dc_sa + pq * dct_size + s] * coef;
            }
            fftwf_free(vec);
        }   
    }

    //! Processing 1D DCT on rows
    for(unsigned s = 0; s < aheight; s++)
    {
        //! Get dct size
        unsigned dct_size = 0;
        for(unsigned t = 0; t < awidth; t++)
            dct_size += shape_mask_col[s * awidth + t];

        if(dct_size == 1) //! No transform needed
        {
            for (unsigned c = 0; c < chnls; c++)
            {
                const unsigned dc = c * asize * kHW_2;
                for (unsigned pq = 0; pq < kHW_2; pq++)
                    group_4D[dc + pq * asize + s * awidth + shape_idx[s * awidth]] = group_4D[dc + pq * asize + s * awidth];
            }
        }
        if(dct_size > 1)
        {
            const unsigned size = chnls * kHW_2 * dct_size;

            //! Allocating Memory
            float* vec = (float*) fftwf_malloc(size * sizeof(float));
            float* dct = (float*) fftwf_malloc(size * sizeof(float));

            for (unsigned c = 0; c < chnls; c++)
            {
                const unsigned dc_sa = c * dct_size * kHW_2;
                const unsigned dc    = c * asize    * kHW_2;
                for (unsigned pq = 0; pq < kHW_2; pq++)
                    for (unsigned t = 0; t < dct_size; t++)
                        dct[dc_sa + pq * dct_size + t] = group_4D[dc + pq * asize + s * awidth + t] * coef_norm_inv[dct_size-2][t];
            }

            //! Process of all DCTs
            fftwf_execute_r2r(plan[dct_size-2], dct, vec);
            fftwf_free(dct);

            //! Getting the result
            const float coef = 0.5f * (float)(SQRT2_INV) / sqrt((float) dct_size);
            for (unsigned c = 0; c < chnls; c++)
            {
                const unsigned dc_sa = c * dct_size * kHW_2;
                const unsigned dc    = c * asize    * kHW_2;
                for (unsigned pq = 0; pq < kHW_2; pq++)
                    for (unsigned t = 0; t < dct_size; t++)
                        group_4D[dc + pq * asize + s * awidth + shape_idx[s * awidth + t]] = vec[dc_sa + pq * dct_size + t] * coef;
            }
            fftwf_free(vec);
        }
    }

    //! Put non shape value to zero and reorganize pixels
    vector<float> tmp_group_4D = group_4D;
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * asize * kHW_2;
        for (unsigned pq = 0; pq < kHW_2; pq++)
            for (unsigned st = 0; st < asize; st++)
                group_4D[dc + st * kHW_2 + pq] = tmp_group_4D[dc + pq * asize + st] * (float) shape_mask[st];
    }
}

/**
 * @brief HT filtering using Welsh-Hadamard transform (do only fifth
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_5D : contains the 5D blocks;
 * @param tmp: allocated vector used in Hadamard transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param awdith, aheight : size of angular patches;
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param lambdaHard5D : value of thresholding;
 * @param weight_table: the weighting of this 5D group for each channel.
 *
 * @return none.
 **/
void ht_filtering_hadamard_5d(
    vector<float> &group_5D
,   vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   vector<float> const& sigma_table
,   const float lambdaHard5D
,   vector<float> &weight_table
){
    //! Declarations
    const unsigned asize = awidth * aheight;
    const float coef_norm = sqrtf((float) nSx_r);
    const float coef = 1.0f / (float) nSx_r;

    //! Process the Welsh-Hadamard transform on the 3rd dimension
    if(nSx_r > 1)
        for (unsigned n = 0; n < asize * chnls; n++)
            hadamard_transform(group_5D, tmp, nSx_r, n * nSx_r);

    //! Hard Thresholding
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        const float T = lambdaHard5D * sigma_table[c] * coef_norm * (float)(SQRT2);
        for (unsigned k = 0; k < asize * nSx_r; k++)
        {
            if (fabs(group_5D[k + dc]) > T)
                weight_table[c]++;
            else
                group_5D[k + dc] = 0.0f;
        }
    }

    //! Process of the Welsh-Hadamard inverse transform
    if(nSx_r > 1)
    {
        for (unsigned n = 0; n < asize * chnls; n++)
            hadamard_transform(group_5D, tmp, nSx_r, n * nSx_r);

        for (unsigned k = 0; k < group_5D.size(); k++)
            group_5D[k] *= coef;
    }
}

/**
 * @brief HT filtering using Welsh-Hadamard transform (do only fifth
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_5D : contains the 5D blocks;
 * @param tmp: allocated vector used in Hadamard transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param awdith, aheight : size of angular patches;
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param lambdaHard5D : value of thresholding;
 * @param weight_table: the weighting of this 5D group for each channel;
 * @param shape_mask : mask indicating patch pixels which have to be denoised (1) or not (0).
 *
 * @return none.
 **/
void ht_filtering_hadamard_5d(
    vector<float> &group_5D
,   vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   vector<float> const& sigma_table
,   const float lambdaHard5D
,   vector<float> &weight_table
,   vector<unsigned> shape_mask_dct
){
    //! Declarations
    const unsigned asize = awidth * aheight;
    const float coef_norm = sqrtf((float) nSx_r);
    const float coef = 1.0f / (float) nSx_r;

    //! Process the Welsh-Hadamard transform on the 3rd dimension
    if(nSx_r > 1)
        for (unsigned n = 0; n < asize * chnls; n++)
            hadamard_transform(group_5D, tmp, nSx_r, n * nSx_r);

    //! Hard Thresholding
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        const float T = lambdaHard5D * sigma_table[c] * coef_norm * (float)(SQRT2);
        
        for (unsigned st = 0; st < asize ; st++)
            if(shape_mask_dct[st])
                for(unsigned n = 0; n < nSx_r; n++)
                {
                    if (fabs(group_5D[st * nSx_r + n + dc]) > T)
                        weight_table[c]++;
                    else
                        group_5D[st * nSx_r + n + dc] = 0.0f;
                }
    }

    //! Process of the Welsh-Hadamard inverse transform
    if(nSx_r > 1)
    {
        for (unsigned n = 0; n < asize * chnls; n++)
            hadamard_transform(group_5D, tmp, nSx_r, n * nSx_r);

        for (unsigned k = 0; k < group_5D.size(); k++)
            group_5D[k] *= coef;
    }
}

/**
 * @brief HT filtering using Haar wavelet transform (do only fifth
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_5D : contains the 5D block;
 * @param tmp: allocated vector used in Haar transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param awidth, aheight : size of angular patches;
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param lambdaHard5D : value of thresholding;
 * @param weight_table: the weighting of this 3D group for each channel.
 *
 * @return none.
 **/
void ht_filtering_haar_5d(
    vector<float> &group_5D
,   vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   vector<float> const& sigma_table
,   const float lambdaHard5D
,   vector<float> &weight_table
){
    //! Declarations
    const unsigned asize = awidth * aheight;
    
    //! Process the Welsh-Hadamard transform on the 3rd dimension
    if(nSx_r > 1)
        for (unsigned n = 0; n < asize * chnls; n++)
            haar_forward(group_5D, tmp, nSx_r, n * nSx_r);

    //! Hard Thresholding
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        const float T = lambdaHard5D * sigma_table[c] * (float)(SQRT2);
        for (unsigned k = 0; k < asize * nSx_r; k++)
        {
            if (fabs(group_5D[k + dc]) > T)
                weight_table[c]++;
            else
                group_5D[k + dc] = 0.0f;
        }
    }

    //! Process of the Welsh-Hadamard inverse transform
    if(nSx_r > 1)
        for (unsigned n = 0; n < asize * chnls; n++)
            haar_inverse(group_5D, tmp, 1, nSx_r, n * nSx_r);
}


/**
 * @brief HT filtering using Haar wavelet transform (do only fifth
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_5D : contains the 5D block;
 * @param tmp: allocated vector used in Haar transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param awidth, aheight : size of angular patches;
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param lambdaHard5D : value of thresholding;
 * @param weight_table: the weighting of this 3D group for each channel;
 * @param shape_mask : mask indicating patch pixels which have to be denoised (1) or not (0).
 *
 * @return none.
 **/
void ht_filtering_haar_5d(
    vector<float> &group_5D
,   vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   vector<float> const& sigma_table
,   const float lambdaHard5D
,   vector<float> &weight_table
,   vector<unsigned> shape_mask_dct
){
    //! Declarations
    const unsigned asize = awidth * aheight;
    
    //! Process the Welsh-Hadamard transform on the 3rd dimension
    if(nSx_r > 1)
        for (unsigned n = 0; n < asize * chnls; n++)
            haar_forward(group_5D, tmp, nSx_r, n * nSx_r);

    //! Hard Thresholding
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        const float T = lambdaHard5D * sigma_table[c] * (float)(SQRT2);
        
        for (unsigned st = 0; st < asize ; st++)
            if(shape_mask_dct[st])
                for(unsigned n = 0; n < nSx_r; n++)
                {
                    if (fabs(group_5D[st * nSx_r + n + dc]) > T)
                        weight_table[c]++;
                    else
                        group_5D[st * nSx_r + n + dc] = 0.0f;
                }
    }

    //! Process of the Welsh-Hadamard inverse transform
    if(nSx_r > 1)
        for (unsigned n = 0; n < asize * chnls; n++)
            haar_inverse(group_5D, tmp, 1, nSx_r, n * nSx_r);
}


/**
 * @brief HT filtering using dct transform (do only fifth
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_5D : contains the 5D block;
 * @param plan, plan_inv : for convenience. Used by fftw;
 * @param coef_norm, coef_norm_inv : normalization coefficients of the DCT;
 * @param nSx_r : number of similar patches to a reference one;
 * @param awidth, aheight : size of angular patches;
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param lambdaHard5D : value of thresholding;
 * @param weight_table: the weighting of this 3D group for each channel.
 *
 * @return none.
 **/
void ht_filtering_dct_5d(
    vector<float> &group_5D
,   fftwf_plan * plan
,   fftwf_plan * plan_inv
,   vector<float> &coef_norm
,   vector<float> &coef_norm_inv
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   vector<float> const& sigma_table
,   const float lambdaHard5D
,   vector<float> &weight_table
){
    //! Declarations
    const unsigned asize = awidth * aheight;
    const unsigned size = chnls * asize * nSx_r;

    //! Allocating Memory
    float* vec = (float*) fftwf_malloc(size * sizeof(float));
    float* dct = (float*) fftwf_malloc(size * sizeof(float));

    for (unsigned n = 0; n < size; n++)
        vec[n] = group_5D[n];

    //! Process of all DCTs
    fftwf_execute_r2r(*plan, vec, dct);

    //! Normalization
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for (unsigned st = 0; st < asize; st++)
            for(unsigned n = 0; n < nSx_r; n++)
                dct[n + st * nSx_r + dc] *= coef_norm[n];
    }
    
    //! Hard Thresholding
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for(unsigned n = 0; n < nSx_r; n++)
        {
            const float T = lambdaHard5D * sigma_table[c] * 2.0f * (float)(SQRT2); //* coef_norm[n] 
            for (unsigned st = 0; st < asize; st++)
            {
                if (fabs(dct[n + st * nSx_r + dc]) > T)
                    weight_table[c]++;
                else
                    dct[n + st * nSx_r + dc] = 0.0f;
            }
        }
    }

    //! Normalization
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for (unsigned st = 0; st < asize; st++)
            for(unsigned n = 0; n < nSx_r; n++)
                dct[n + st * nSx_r + dc] *= coef_norm_inv[n];
    }

    //! Process inverse dct
    fftwf_execute_r2r(*plan_inv, dct, vec);
    fftwf_free(dct);

    const float coef = 0.5f * (float)(SQRT2_INV) / sqrt((float) nSx_r);
    for (unsigned n = 0; n < size; n++)
        group_5D[n] = vec[n] * coef;

    fftwf_free(vec);
}

/**
 * @brief HT filtering using dct transform (do only fifth
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_5D : contains the 5D block;
 * @param plan, plan_inv : for convenience. Used by fftw;
 * @param coef_norm, coef_norm_inv : normalization coefficients of the DCT;
 * @param tmp: allocated vector used in Haar transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param awidth, aheight : size of angular patches;
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param lambdaHard5D : value of thresholding;
 * @param weight_table: the weighting of this 3D group for each channel;
 * @param shape_mask : mask indicating patch pixels which have to be denoised (1) or not (0).
 *
 * @return none.
 **/
void ht_filtering_dct_5d(
    vector<float> &group_5D
,   fftwf_plan * plan
,   fftwf_plan * plan_inv
,   vector<float> &coef_norm
,   vector<float> &coef_norm_inv
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   vector<float> const& sigma_table
,   const float lambdaHard5D
,   vector<float> &weight_table
,   vector<unsigned> shape_mask_dct
){
    //! Declarations
    const unsigned asize = awidth * aheight;
    const unsigned size = chnls * asize * nSx_r;

    //! Allocating Memory
    float* vec = (float*) fftwf_malloc(size * sizeof(float));
    float* dct = (float*) fftwf_malloc(size * sizeof(float));

    for (unsigned n = 0; n < size; n++)
        vec[n] = group_5D[n];

    //! Process of all DCTs
    fftwf_execute_r2r(*plan, vec, dct);

    //! Normalization
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for (unsigned st = 0; st < asize; st++)
            for(unsigned n = 0; n < nSx_r; n++)
                dct[n + st * nSx_r + dc] *= coef_norm[n];
    }
    
    //! Hard Thresholding
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for(unsigned n = 0; n < nSx_r; n++)
        {
            const float T = lambdaHard5D * sigma_table[c] * 2.0f * (float)(SQRT2); //* coef_norm[n] 
            for (unsigned st = 0; st < asize; st++)
                if(shape_mask_dct[st])
                {
                    if (fabs(dct[n + st * nSx_r + dc]) > T)
                        weight_table[c]++;
                    else
                        dct[n + st * nSx_r + dc] = 0.0f;
                }
        }
    }

    //! Normalization
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for (unsigned st = 0; st < asize; st++)
            for(unsigned n = 0; n < nSx_r; n++)
                dct[n + st * nSx_r + dc] *= coef_norm_inv[n];
    }

    //! Process inverse dct
    fftwf_execute_r2r(*plan_inv, dct, vec);
    fftwf_free(dct);

    const float coef = 0.5f * (float)(SQRT2_INV) / sqrt((float) nSx_r);
    for (unsigned n = 0; n < size; n++)
        group_5D[n] = vec[n] * coef;

    fftwf_free(vec);
}

/**
 * @brief Wiener filtering using Welsh-Hadamard transform (do only fifth
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_5D_org, group_5D_est : contains the 5D blocks from the noisy light field and basic estimation respectively;
 * @param tmp: allocated vector used in Hadamard transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param awdith, aheight : size of angular patches;
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param weight_table: the weighting of this 5D group for each channel.
 *
 * @return none.
 **/
void wiener_filtering_hadamard_5d(
    vector<float> &group_5D_org
,   vector<float> &group_5D_est
,   vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   vector<float> const& sigma_table
,   vector<float> &weight_table
){
    //! Declarations
    const unsigned asize = awidth * aheight;
    const float coef = 1.0f / (float) nSx_r;

    //! Process the Welsh-Hadamard transform on the 5th dimension
    if(nSx_r > 1)
        for (unsigned n = 0; n < asize * chnls; n++)
        {
            hadamard_transform(group_5D_org, tmp, nSx_r, n * nSx_r);
            hadamard_transform(group_5D_est, tmp, nSx_r, n * nSx_r);
        }

    //! Wiener Filtering
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for (unsigned k = 0; k < asize * nSx_r; k++)
        {
            float value = group_5D_est[dc + k] * group_5D_est[dc + k] * coef;
            value /= (value + sigma_table[c] * sigma_table[c]);
            group_5D_est[k + dc] = group_5D_org[k + dc] * value * coef;
            weight_table[c] += value;
        }
    }

    //! Process of the Welsh-Hadamard inverse transform
    if(nSx_r > 1)
    {
        for (unsigned n = 0; n < asize * chnls; n++)
            hadamard_transform(group_5D_est, tmp, nSx_r, n * nSx_r);
    }
}

/**
 * @brief Wiener filtering using Welsh-Hadamard transform (do only fifth
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_5D_org, group_5D_est : contains the 5D blocks from the noisy light field and basic estimation respectively;
 * @param tmp: allocated vector used in Hadamard transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param awdith, aheight : size of angular patches;
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param weight_table: the weighting of this 5D group for each channel;
 * @param shape_mask : mask indicating patch pixels which have to be denoised (1) or not (0).
 *
 * @return none.
 **/
void wiener_filtering_hadamard_5d(
    vector<float> &group_5D_org
,   vector<float> &group_5D_est
,   vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   vector<float> const& sigma_table
,   vector<float> &weight_table
,   vector<unsigned> shape_mask_dct
){
    //! Declarations
    const unsigned asize = awidth * aheight;
    const float coef = 1.0f / (float) nSx_r;

    //! Process the Welsh-Hadamard transform on the 5th dimension
    if(nSx_r > 1)
        for (unsigned n = 0; n < asize * chnls; n++)
        {
            hadamard_transform(group_5D_org, tmp, nSx_r, n * nSx_r);
            hadamard_transform(group_5D_est, tmp, nSx_r, n * nSx_r);
        }

    //! Wiener Filtering
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for (unsigned st = 0; st < asize ; st++)
            if(shape_mask_dct[st])
                for(unsigned n = 0; n < nSx_r; n++)
                {
                    float value = group_5D_est[n + st * nSx_r + dc] * group_5D_est[n + st * nSx_r + dc] * coef;
                    value /= (value + sigma_table[c] * sigma_table[c]);
                    group_5D_est[n + st * nSx_r + dc] = group_5D_org[n + st * nSx_r + dc] * value * coef;
                    weight_table[c] += value;
                }
    }

    //! Process of the Welsh-Hadamard inverse transform
    if(nSx_r > 1)
    {
        for (unsigned n = 0; n < asize * chnls; n++)
            hadamard_transform(group_5D_est, tmp, nSx_r, n * nSx_r);
    }
}

/**
 * @brief Wiener filtering using Haar wavelet transform (do only fifth
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_5D_org, group_5D_est : contains the 5D blocks from the noisy light field and basic estimation respectively;
 * @param tmp: allocated vector used in Haar transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param awidth, aheight : size of angular patches;
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param weight_table: the weighting of this 3D group for each channel.
 *
 * @return none.
 **/
void wiener_filtering_haar_5d(
    vector<float> &group_5D_org
,   vector<float> &group_5D_est
,   vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   vector<float> const& sigma_table
,   vector<float> &weight_table
){
    //! Declarations
    const unsigned asize = awidth * aheight;
    
    //! Process the Welsh-Hadamard transform on the 3rd dimension
    if(nSx_r > 1)
        for (unsigned n = 0; n < asize * chnls; n++)
        {
            haar_forward(group_5D_org, tmp, nSx_r, n * nSx_r);
            haar_forward(group_5D_est, tmp, nSx_r, n * nSx_r);
        }

    //! Wiener Filtering
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for (unsigned k = 0; k < asize * nSx_r; k++)
        {
            float value = group_5D_est[dc + k] * group_5D_est[dc + k];
            value /= (value + sigma_table[c] * sigma_table[c]);
            group_5D_est[k + dc] = group_5D_org[k + dc] * value;
            weight_table[c] += value;
        }
    }

    //! Process of the Welsh-Hadamard inverse transform
    if(nSx_r > 1)
        for (unsigned n = 0; n < asize * chnls; n++)
            haar_inverse(group_5D_est, tmp, 1, nSx_r, n * nSx_r);
}


/**
 * @brief Wiener filtering using Haar wavelet transform (do only fifth
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_5D_org, group_5D_est : contains the 5D blocks from the noisy light field and basic estimation respectively;
 * @param tmp: allocated vector used in Haar transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param awidth, aheight : size of angular patches;
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param weight_table: the weighting of this 3D group for each channel;
 * @param shape_mask : mask indicating patch pixels which have to be denoised (1) or not (0).
 *
 * @return none.
 **/
void wiener_filtering_haar_5d(
    vector<float> &group_5D_org
,   vector<float> &group_5D_est
,   vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   vector<float> const& sigma_table
,   vector<float> &weight_table
,   vector<unsigned> shape_mask_dct
){
    //! Declarations
    const unsigned asize = awidth * aheight;
    
    //! Process the Welsh-Hadamard transform on the 3rd dimension
    if(nSx_r > 1)
        for (unsigned n = 0; n < asize * chnls; n++)
        {
            haar_forward(group_5D_org, tmp, nSx_r, n * nSx_r);
            haar_forward(group_5D_est, tmp, nSx_r, n * nSx_r);
        }

    //! Wiener Filtering
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for (unsigned st = 0; st < asize ; st++)
            if(shape_mask_dct[st])
                for(unsigned n = 0; n < nSx_r; n++)
                {
                    float value = group_5D_est[n + st * nSx_r + dc] * group_5D_est[n + st * nSx_r + dc];
                    value /= (value + sigma_table[c] * sigma_table[c]);
                    group_5D_est[n + st * nSx_r + dc] = group_5D_org[n + st * nSx_r + dc] * value;
                    weight_table[c] += value;
                }
    }

    //! Process of the Welsh-Hadamard inverse transform
    if(nSx_r > 1)
        for (unsigned n = 0; n < asize * chnls; n++)
            haar_inverse(group_5D_est, tmp, 1, nSx_r, n * nSx_r);
}


/**
 * @brief Wiener filtering using dct transform (do only fifth
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_5D_org, group_5D_est : contains the 5D blocks from the noisy light field and basic estimation respectively;
 * @param plan, plan_inv : for convenience. Used by fftw;
 * @param coef_norm, coef_norm_inv : normalization coefficients of the DCT;
 * @param nSx_r : number of similar patches to a reference one;
 * @param awidth, aheight : size of angular patches;
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param weight_table: the weighting of this 3D group for each channel.
 *
 * @return none.
 **/
void wiener_filtering_dct_5d(
    vector<float> &group_5D_org
,   vector<float> &group_5D_est
,   fftwf_plan * plan
,   fftwf_plan * plan_inv
,   vector<float> &coef_norm
,   vector<float> &coef_norm_inv
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   vector<float> const& sigma_table
,   vector<float> &weight_table
){
    //! Declarations
    const unsigned asize = awidth * aheight;
    const unsigned size = chnls * asize * nSx_r;

    //! Allocating Memory
    float* vec_org = (float*) fftwf_malloc(size * sizeof(float));
    float* vec_est = (float*) fftwf_malloc(size * sizeof(float));
    float* dct_org = (float*) fftwf_malloc(size * sizeof(float));
    float* dct_est = (float*) fftwf_malloc(size * sizeof(float));

    for (unsigned n = 0; n < size; n++)
    {
        vec_org[n] = group_5D_org[n];
        vec_est[n] = group_5D_est[n];
    }

    //! Process of all DCTs
    fftwf_execute_r2r(*plan, vec_org, dct_org);
    fftwf_execute_r2r(*plan, vec_est, dct_est);
    fftwf_free(vec_org);

    //! Normalization
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for (unsigned st = 0; st < asize; st++)
            for(unsigned n = 0; n < nSx_r; n++)
            {
                dct_org[n + st * nSx_r + dc] *= coef_norm[n];
                dct_est[n + st * nSx_r + dc] *= coef_norm[n];
            }
    }
    
    //! Wiener Filtering
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for (unsigned st = 0; st < asize ; st++)
            for(unsigned n = 0; n < nSx_r; n++)
            {
                float value = dct_est[n + st * nSx_r + dc] * dct_est[n + st * nSx_r + dc];
                value /= (value + sigma_table[c] * sigma_table[c]);
                dct_est[n + st * nSx_r + dc] = dct_org[n + st * nSx_r + dc] * value;
                weight_table[c] += value;
            }
    }

    //! Normalization
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for (unsigned st = 0; st < asize; st++)
            for(unsigned n = 0; n < nSx_r; n++)
                dct_est[n + st * nSx_r + dc] *= coef_norm_inv[n];
    }

    //! Process inverse dct
    fftwf_execute_r2r(*plan_inv, dct_est, vec_est);
    fftwf_free(dct_est);
    fftwf_free(dct_org);

    const float coef = 0.5f * (float)(SQRT2_INV) / sqrt((float) nSx_r);
    for (unsigned n = 0; n < size; n++)
        group_5D_est[n] = vec_est[n] * coef;

    fftwf_free(vec_est);
}

/**
 * @brief Wiener filtering using dct transform (do only fifth
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_5D_org, group_5D_est : contains the 5D blocks from the noisy light field and basic estimation respectively;
 * @param plan, plan_inv : for convenience. Used by fftw;
 * @param coef_norm, coef_norm_inv : normalization coefficients of the DCT;
 * @param nSx_r : number of similar patches to a reference one;
 * @param awidth, aheight : size of angular patches;
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param weight_table: the weighting of this 3D group for each channel;
 * @param shape_mask : mask indicating patch pixels which have to be denoised (1) or not (0).
 *
 * @return none.
 **/
void wiener_filtering_dct_5d(
    vector<float> &group_5D_org
,   vector<float> &group_5D_est
,   fftwf_plan * plan
,   fftwf_plan * plan_inv
,   vector<float> &coef_norm
,   vector<float> &coef_norm_inv
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   vector<float> const& sigma_table
,   vector<float> &weight_table
,   vector<unsigned> shape_mask_dct
){
    //! Declarations
    const unsigned asize = awidth * aheight;
    const unsigned size = chnls * asize * nSx_r;

    //! Allocating Memory
    float* vec_org = (float*) fftwf_malloc(size * sizeof(float));
    float* vec_est = (float*) fftwf_malloc(size * sizeof(float));
    float* dct_org = (float*) fftwf_malloc(size * sizeof(float));
    float* dct_est = (float*) fftwf_malloc(size * sizeof(float));

    for (unsigned n = 0; n < size; n++)
    {
        vec_org[n] = group_5D_org[n];
        vec_est[n] = group_5D_est[n];
    }

    //! Process of all DCTs
    fftwf_execute_r2r(*plan, vec_org, dct_org);
    fftwf_execute_r2r(*plan, vec_est, dct_est);
    fftwf_free(vec_org);

    //! Normalization
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for (unsigned st = 0; st < asize; st++)
            for(unsigned n = 0; n < nSx_r; n++)
            {
                dct_org[n + st * nSx_r + dc] *= coef_norm[n];
                dct_est[n + st * nSx_r + dc] *= coef_norm[n];
            }
    }

    //! Wiener Filtering
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for (unsigned st = 0; st < asize ; st++)
            if(shape_mask_dct[st])
                for(unsigned n = 0; n < nSx_r; n++)
                {
                    float value = dct_est[n + st * nSx_r + dc] * dct_est[n + st * nSx_r + dc];
                    value /= (value + sigma_table[c] * sigma_table[c]);
                    dct_est[n + st * nSx_r + dc] = dct_org[n + st * nSx_r + dc] * value;
                    weight_table[c] += value;
                }
    }

    //! Normalization
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * asize;
        for (unsigned st = 0; st < asize; st++)
            for(unsigned n = 0; n < nSx_r; n++)
                dct_est[n + st * nSx_r + dc] *= coef_norm_inv[n];
    }

    //! Process inverse dct
    fftwf_execute_r2r(*plan_inv, dct_est, vec_est);
    fftwf_free(dct_est);
    fftwf_free(dct_org);

    const float coef = 0.5f * (float)(SQRT2_INV) / sqrt((float) nSx_r);
    for (unsigned n = 0; n < size; n++)
        group_5D_est[n] = vec_est[n] * coef;

    fftwf_free(vec_est);
}


/**
 * @brief Process of a weight dependent on the standard
 *        deviation, used during the weighted aggregation.
 *
 * @param group_5D : 5D group
 * @param kHW_2: size of square patches
 * @param nSx_r : number of similar patches along the 5th dimension
 * @param awidth, aheight : size of angular patches;
 * @param chnls: number of channels in the image;
 * @param weight_table: will contain the weighting for each
 *        channel.
 *
 * @return none.
 **/
void sd_weighting_5d(
    vector<vector<float> > const& group_5D
,   const unsigned kHW_2
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   vector<float> &weight_table
){
    const unsigned N = nSx_r * awidth * aheight;

    for (unsigned c = 0; c < chnls; c++)
    {
        //! Initialization
        float mean = 0.0f;
        float std  = 0.0f;

        unsigned dc = c * N;

        //! Compute the sum and the square sum
        for(unsigned pq = 0; pq < kHW_2; pq++)
            for (unsigned k = 0; k < N; k++)
            {
                mean += group_5D[pq][dc + k];
                std  += group_5D[pq][dc + k] * group_5D[pq][dc + k];
            }

        //! Sample standard deviation (Bessel's correction)
        float res = (std - mean * mean / (float) N) / (float) (N - 1);

        //! Return the weight as used in the aggregation
        weight_table[c] = (res > 0.0f ? 1.0f / sqrtf(res) : 0.0f);
    }
}

/*********************************************************************************************************************************************************************************/
/*********************************************************************************************************************************************************************************/
/** ----------------- **/
/** - Preprocessing - **/
/** ----------------- **/
/*********************************************************************************************************************************************************************************/
/*********************************************************************************************************************************************************************************/
/**
 * @brief Preprocess for 4d dct
 *
 * @param coef_norm: Will contain values used to normalize the 2D DCT;
 * @param coef_norm_inv: Will contain values used to normalize the 2D DCT;
 * @param awidth, aheight: size of angular patches.
 *
 * @return none.
 **/
void preProcess_4d(
    vector<float> &coef_norm
,   vector<float> &coef_norm_inv
,   const unsigned awidth
,   const unsigned aheight
){  
    //! Coefficient of normalization for DCT II and DCT II inverse
    const float coef = 0.5f / (sqrt((float) (awidth)) * sqrt((float) (aheight)));
    for (unsigned i = 0; i < aheight; i++)
        for (unsigned j = 0; j < awidth; j++)
        {
            if (i == 0 && j == 0)
            {
                coef_norm    [i * awidth + j] = (float) (0.5f * coef);
                coef_norm_inv[i * awidth + j] = (float) (2.0f);
            }
            else if (i * j == 0)
            {
                coef_norm    [i * awidth + j] = (float) (SQRT2_INV * coef);
                coef_norm_inv[i * awidth + j] = (float) (SQRT2);
            }
            else
            {
                coef_norm    [i * awidth + j] = (float) (1.0f * coef);
                coef_norm_inv[i * awidth + j] = (float) (1.0f);
            }
        }
}

/**
 * @brief Preprocess for 4d sadct
 *
 * @param coef_norm: Will contain values used to normalize the 1D DCT of variable sizes;
 * @param coef_norm_inv: Will contain values used to normalize the 1D DCT of variable sizes;
 * @param max_dct_size: maximum size of 1D DCT used in SADCT.
 *
 * @return none.
 **/
void preProcess_4d_sadct(
    vector<vector<float> > &coef_norm
,   vector<vector<float> > &coef_norm_inv
,   const unsigned max_dct_size
){  
    //! Coefficient of normalization for DCT II and DCT II inverse
    for(unsigned k = 0; k < (max_dct_size-1); k++)
    {
        const unsigned dct_size = k+2;
        if(coef_norm[k].size() != dct_size)
            coef_norm[k].resize(dct_size);
        if(coef_norm_inv[k].size() != dct_size)
            coef_norm_inv[k].resize(dct_size);

        const float coef = (float) (SQRT2) / sqrt(dct_size);
        coef_norm    [k][0]  = (float) (SQRT2_INV * coef);
        coef_norm_inv[k][0]  = (float) (SQRT2);
        for (unsigned i = 1; i < dct_size; i++)
        {
            coef_norm    [k][i] = coef;
            coef_norm_inv[k][i] = 1.0f;
        }
    }
}

/**
 * @brief Preprocess for 5d dct
 *
 * @param coef_norm, coef_norm_inv: Will contain the inverse values used to normalize the 1D DCT forward and backward respectively (inverse because its applied on HT theshold and not dct value directly);
 * @param NHW: size of 5d dct.
 *
 * @return none.
 **/
void preProcess_5d(
    vector<float> &coef_norm
,   vector<float> &coef_norm_inv
,   const unsigned NHW
){  
    //! Coefficient of normalization for DCT II and DCT II inverse
    const float coef = (float) (SQRT2) / sqrt(NHW);
    coef_norm[0]     = (float) (SQRT2_INV * coef);
    coef_norm_inv[0] = (float) (SQRT2);
    for (unsigned i = 1; i < NHW; i++)
    {
        coef_norm[i]     = coef;
        coef_norm_inv[i] = 1.0;
    }
}

/*********************************************************************************************************************************************************************************/
/*********************************************************************************************************************************************************************************/
/** ------------------ **/
/** - Block matching - **/
/** ------------------ **/
/*********************************************************************************************************************************************************************************/
/*********************************************************************************************************************************************************************************/
/**
 * @brief Precompute Bloc Matching (distance inter-patches)
 *
 * @param patch_table: for each patch in the image, will contain all coordonnate of its similar patches;
 * @param img: noisy image on which the distance is computed;
 * @param width, height: size of img;
 * @param kHW: size of patch;
 * @param NHW: maximum similar patches wanted;
 * @param nHW: size of the boundary of img;
 * @param nHW_sim: half of the search window for similarities;
 * @param pHW: Processing step on row and columns;
 * @param tauMatch: threshold used to determinate similarity between
 *        patches.
 *
 * @return none.
 **/
void precompute_BM(
    vector<vector<unsigned> > &patch_table
,   const vector<float> &img
,   const unsigned width
,   const unsigned height
,   const unsigned kHW
,   const unsigned NHW
,   const unsigned nHW
,   const unsigned nHW_sim
,   const unsigned pHW
,   const float    tauMatch
){
    //! Declarations
    const unsigned Ns = 2 * nHW_sim + 1;
    const float threshold = tauMatch * kHW * kHW;
    vector<float> diff_table(width * height);
    vector<vector<float> > sum_table((nHW_sim + 1) * Ns, vector<float> (width * height, 2 * threshold));
    if (patch_table.size() != width * height)
        patch_table.resize(width * height);
    vector<unsigned> row_ind;
    ind_initialize(row_ind, height - kHW + 1, nHW, pHW);
    vector<unsigned> column_ind;
    ind_initialize(column_ind, width - kHW + 1, nHW, pHW);

    if(NHW > 1)
    {
        //! For each possible distance, precompute inter-patches distance
        for (unsigned di = 0; di <= nHW_sim; di++)
            for (unsigned dj = 0; dj < Ns; dj++)
            {
                const int dk = (int) (di * width + dj) - (int) nHW_sim;
                const unsigned ddk = di * Ns + dj;

                //! Process the image containing the square distance between pixels
                for (unsigned i = nHW; i < height - nHW; i++)
                {
                    unsigned k = i * width + nHW;
                    for (unsigned j = nHW; j < width - nHW; j++, k++)
                        diff_table[k] = (img[k + dk] - img[k]) * (img[k + dk] - img[k]);
                }

                //! Compute the sum for each patches, using the method of the integral images
                const unsigned dn = nHW * width + nHW;
                //! 1st patch, top left corner
                float value = 0.0f;
                for (unsigned p = 0; p < kHW; p++)
                {
                    unsigned pq = p * width + dn;
                    for (unsigned q = 0; q < kHW; q++, pq++)
                        value += diff_table[pq];
                }
                sum_table[ddk][dn] = value;

                //! 1st row, top
                for (unsigned j = nHW + 1; j < width - nHW; j++)
                {
                    const unsigned ind = nHW * width + j - 1;
                    float sum = sum_table[ddk][ind];
                    for (unsigned p = 0; p < kHW; p++)
                        sum += diff_table[ind + p * width + kHW] - diff_table[ind + p * width];
                    sum_table[ddk][ind + 1] = sum;
                }

                //! General case
                for (unsigned i = nHW + 1; i < height - nHW; i++)
                {
                    const unsigned ind = (i - 1) * width + nHW;
                    float sum = sum_table[ddk][ind];
                    //! 1st column, left
                    for (unsigned q = 0; q < kHW; q++)
                        sum += diff_table[ind + kHW * width + q] - diff_table[ind + q];
                    sum_table[ddk][ind + width] = sum;

                    //! Other columns
                    unsigned k = i * width + nHW + 1;
                    unsigned pq = (i + kHW - 1) * width + kHW - 1 + nHW + 1;
                    for (unsigned j = nHW + 1; j < width - nHW; j++, k++, pq++)
                    {
                        sum_table[ddk][k] =
                            sum_table[ddk][k - 1]
                            + sum_table[ddk][k - width]
                            - sum_table[ddk][k - 1 - width]
                            + diff_table[pq]
                            - diff_table[pq - kHW]
                            - diff_table[pq - kHW * width]
                            + diff_table[pq - kHW - kHW * width];
                    }

                }
            }

        //! Precompute Bloc Matching
        vector<pair<float, unsigned> > table_distance;
        //! To avoid reallocation
        table_distance.reserve(Ns * Ns);

        for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
        {
            for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
            {
                //! Initialization
                const unsigned k_r = row_ind[ind_i] * width + column_ind[ind_j];
                table_distance.clear();
                patch_table[k_r].clear();

                //! Threshold distances in order to keep similar patches
                for (int dj = -(int) nHW_sim; dj <= (int) nHW_sim; dj++)
                {
                    for (int di = 0; di <= (int) nHW_sim; di++)
                        if (sum_table[dj + nHW_sim + di * Ns][k_r] < threshold)
                            table_distance.push_back(make_pair(
                                        sum_table[dj + nHW_sim + di * Ns][k_r]
                                    , k_r + di * width + dj));

                    for (int di = - (int) nHW_sim; di < 0; di++)
                        if (sum_table[-dj + nHW_sim + (-di) * Ns][k_r] < threshold)
                            table_distance.push_back(make_pair(
                                        sum_table[-dj + nHW_sim + (-di) * Ns][k_r + di * width + dj]
                                    , k_r + di * width + dj));
                }

                //! We need a power of 2 for the number of similar patches,
                //! because of the Welsh-Hadamard transform on the third dimension.
                //! We assume that NHW is already a power of 2
                const unsigned nSx_r = (NHW > table_distance.size() ?
                                        closest_power_of_2(table_distance.size()) : NHW);

                //! To avoid problem
                if (nSx_r == 1 && table_distance.size() == 0)
                {
                    table_distance.push_back(make_pair(0, k_r));
                }

                //! Sort patches according to their distance to the reference one
                partial_sort(table_distance.begin(), table_distance.begin() + nSx_r,
                                                table_distance.end(), ComparaisonFirst);

                //! Keep a maximum of NHW similar patches
                for (unsigned n = 0; n < nSx_r; n++)
                    patch_table[k_r].push_back(table_distance[n].second);

                //! To avoid problem
                if (nSx_r == 1)
                    patch_table[k_r].push_back(table_distance[0].second);
            }
        }
    }
    else
    {
        for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
        {
            for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
            {
                //! Initialization
                const unsigned k_r = row_ind[ind_i] * width + column_ind[ind_j];
                patch_table[k_r].clear();
                patch_table[k_r].push_back(k_r);
            }
        }
    }
}

/**
 * @brief Precompute Bloc Matching between a pair of images (distance inter-patches)
 *
 * @param patch_table: for each patch in img1, will contain all coordinnates of its similar patches in img2;
 * @param shape_table: mask indicating if found similar patch distance is below (1) or above (0) tauMatch;
 * @param img1, img2: pair of noisy image on which the distance is computed;
 * @param width, height: size of img (including borders);
 * @param kHW: size of patch;
 * @param nHW: size of the boundary of img;
 * @param nHW_disp: half of the search window for disparity, smaller or equal to nHW;
 * @param pHW: Processing step on row and columns;
 * @param tauMatch: threshold used to determinate similarity between
 *        patches.
 *
 * @return none.
 **/
int precompute_BM_stereo(
    vector<vector<unsigned> > &patch_table
,   vector<unsigned> &shape_table
,   const vector<float> &img1
,   const vector<float> &img2
,   const unsigned width
,   const unsigned height
,   const unsigned kHW
,   const unsigned nHW
,   const unsigned nHW_disp
,   const unsigned pHW
,   const float    tauMatch
){
    //! Check allocation memory
    if (img1.size() != img2.size())
    {
        cout << "The two images for stereo BM should have the same size: " << img1.size() << " != " << img2.size() << endl;
        return EXIT_FAILURE;
    }

    //! Declarations
    const unsigned Ns_disp = 2 * nHW_disp + 1;
    const float threshold = tauMatch * kHW * kHW;
    vector<float> diff_table(width * height);
    vector<vector<float> > sum_table(Ns_disp * Ns_disp, vector<float> (width * height));
    if (patch_table.size() != width * height)
        patch_table.resize(width * height);
    if (shape_table.size() != width * height)
        shape_table.resize(width * height);
    vector<unsigned> row_ind, column_ind;
    ind_initialize(row_ind,    height - kHW + 1, nHW_disp, pHW);
    ind_initialize(column_ind, width  - kHW + 1, nHW_disp, pHW);

    //! For each possible distance, precompute inter-patches distance
    for (unsigned di = 0; di < Ns_disp; di++)
        for (unsigned dj = 0; dj < Ns_disp; dj++)
        {
            const int dk = (int) (di * width + dj) - (int) (nHW_disp * (1 + width));
            const unsigned ddk = di * Ns_disp + dj;

            //! Process the image containing the square distance between pixels
            for (unsigned i = nHW_disp; i < height - nHW_disp; i++)
            {
                unsigned k = i * width + nHW_disp;
                for (unsigned j = nHW_disp; j < width - nHW_disp; j++, k++)
                    diff_table[k] = (img2[k + dk] - img1[k]) * (img2[k + dk] - img1[k]);
            }

            //! Compute the sum for each patches, using the method of the integral images
            const unsigned dn = nHW_disp * width + nHW_disp;
            //! 1st patch, top left corner
            float value = 0.0f;
            for (unsigned p = 0; p < kHW; p++)
            {
                unsigned pq = p * width + dn;
                for (unsigned q = 0; q < kHW; q++, pq++)
                    value += diff_table[pq];
            }
            sum_table[ddk][dn] = value;

            //! 1st row, top
            for (unsigned j = nHW_disp + 1; j < width - nHW_disp - kHW + 1; j++)
            {
                const unsigned ind = nHW_disp * width + j - 1;
                float sum = sum_table[ddk][ind];
                for (unsigned p = 0; p < kHW; p++)
                    sum += diff_table[ind + p * width + kHW] - diff_table[ind + p * width];
                sum_table[ddk][ind + 1] = sum;
            }

            //! General case
            for (unsigned i = nHW_disp + 1; i < height - nHW_disp - kHW + 1; i++)
            {
                const unsigned ind = (i - 1) * width + nHW_disp;
                float sum = sum_table[ddk][ind];
                //! 1st column, left
                for (unsigned q = 0; q < kHW; q++)
                    sum += diff_table[ind + kHW * width + q] - diff_table[ind + q];
                sum_table[ddk][ind + width] = sum;

                //! Other columns
                unsigned k = i * width + nHW_disp + 1;
                unsigned pq = (i + kHW - 1) * width + kHW - 1 + nHW_disp + 1;
                for (unsigned j = nHW_disp + 1; j < width - nHW_disp - kHW + 1; j++, k++, pq++)
                {
                    sum_table[ddk][k] =
                          sum_table[ddk][k - 1]
                        + sum_table[ddk][k - width]
                        - sum_table[ddk][k - 1 - width]
                        + diff_table[pq]
                        - diff_table[pq - kHW]
                        - diff_table[pq - kHW * width]
                        + diff_table[pq - kHW - kHW * width];
                }
            }
        }

    //! Precompute Bloc Matching
    vector<pair<float, unsigned> > table_distance;
    //! To avoid reallocation
    table_distance.reserve(Ns_disp * Ns_disp);

    for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
    {
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            //! Initialization
            const unsigned k_r = row_ind[ind_i] * width + column_ind[ind_j];
            table_distance.clear();
            patch_table[k_r].clear();

            //! Create pairs of distance / index
            for (int dj = -(int) nHW_disp; dj <= (int) nHW_disp; dj++)
            {
                for (int di = - (int) nHW_disp; di <= (int) nHW_disp; di++)
                    table_distance.push_back(make_pair(
                                sum_table[dj + nHW_disp + (di + nHW_disp) * Ns_disp][k_r]
                                , k_r + di * width + dj));
            }

            //! Sort patches according to their distance to the reference one
            sort(table_distance.begin(), table_distance.end(), ComparaisonFirst);

            //! Store sorted patches in table
            for (unsigned n = 0; n < table_distance.size(); n++)
                patch_table[k_r].push_back(table_distance[n].second);

            //! Evaluate shape
            shape_table[k_r] = table_distance[0].first < threshold ? 1 : 0;
        }
    }
    return EXIT_SUCCESS;
}



/**
 * @brief Precompute Bloc Matching (distance inter-patches) only for patches which need denoising
 *
 * @param patch_table: for each patch in the image, will contain
 * all coordonnate of its similar patches;
 * @param img: noisy image on which the distance is computed;
 * @param width, height: size of img;
 * @param kHW: size of patch;
 * @param NHW: maximum similar patches wanted;
 * @param nHW: size of the boundary of img;
 * @param nHW_sim: half of the search window for similarities;
 * @param tauMatch: threshold used to determinate similarity between patches;
 * @param row_ind, column_ind_per_row: contains indexes of patches which need denoising.
 *
 * @return none.
 **/
void precompute_BM(
    vector<vector<unsigned> > &patch_table
,   const vector<float> &img
,   const unsigned width
,   const unsigned height
,   const unsigned kHW
,   const unsigned NHW
,   const unsigned nHW
,   const unsigned nHW_sim
,   const float    tauMatch
,   const vector<unsigned> row_ind
,   const vector<vector<unsigned> > column_ind_per_row
){
    //! Declarations
    const unsigned Ns = 2 * nHW_sim + 1;
    const float threshold = tauMatch * kHW * kHW;
    vector<float> diff_table(width * height);
    vector<vector<float> > sum_table((nHW_sim + 1) * Ns, vector<float> (width * height, 2 * threshold));
    if (patch_table.size() != width * height)
        patch_table.resize(width * height);

    if(NHW > 1)
    {
        //! For each possible distance, precompute inter-patches distance
        for (unsigned di = 0; di <= nHW_sim; di++)
            for (unsigned dj = 0; dj < Ns; dj++)
            {
                const int dk = (int) (di * width + dj) - (int) nHW_sim;
                const unsigned ddk = di * Ns + dj;

                //! Process the image containing the square distance between pixels
                for (unsigned i = nHW; i < height - nHW; i++)
                {
                    unsigned k = i * width + nHW;
                    for (unsigned j = nHW; j < width - nHW; j++, k++)
                        diff_table[k] = (img[k + dk] - img[k]) * (img[k + dk] - img[k]);
                }

                //! Compute the sum for each patches, using the method of the integral images
                const unsigned dn = nHW * width + nHW;
                //! 1st patch, top left corner
                float value = 0.0f;
                for (unsigned p = 0; p < kHW; p++)
                {
                    unsigned pq = p * width + dn;
                    for (unsigned q = 0; q < kHW; q++, pq++)
                        value += diff_table[pq];
                }
                sum_table[ddk][dn] = value;

                //! 1st row, top
                for (unsigned j = nHW + 1; j < width - nHW; j++)
                {
                    const unsigned ind = nHW * width + j - 1;
                    float sum = sum_table[ddk][ind];
                    for (unsigned p = 0; p < kHW; p++)
                        sum += diff_table[ind + p * width + kHW] - diff_table[ind + p * width];
                    sum_table[ddk][ind + 1] = sum;
                }

                //! General case
                for (unsigned i = nHW + 1; i < height - nHW; i++)
                {
                    const unsigned ind = (i - 1) * width + nHW;
                    float sum = sum_table[ddk][ind];
                    //! 1st column, left
                    for (unsigned q = 0; q < kHW; q++)
                        sum += diff_table[ind + kHW * width + q] - diff_table[ind + q];
                    sum_table[ddk][ind + width] = sum;

                    //! Other columns
                    unsigned k = i * width + nHW + 1;
                    unsigned pq = (i + kHW - 1) * width + kHW - 1 + nHW + 1;
                    for (unsigned j = nHW + 1; j < width - nHW; j++, k++, pq++)
                    {
                        sum_table[ddk][k] =
                            sum_table[ddk][k - 1]
                            + sum_table[ddk][k - width]
                            - sum_table[ddk][k - 1 - width]
                            + diff_table[pq]
                            - diff_table[pq - kHW]
                            - diff_table[pq - kHW * width]
                            + diff_table[pq - kHW - kHW * width];
                    }

                }
            }

        //! Precompute Bloc Matching
        vector<pair<float, unsigned> > table_distance;
        //! To avoid reallocation
        table_distance.reserve(Ns * Ns);

        for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
        {
            for (unsigned ind_j = 0; ind_j < column_ind_per_row[ind_i].size(); ind_j++)
            {
                //! Initialization
                const unsigned k_r = row_ind[ind_i] * width + column_ind_per_row[ind_i][ind_j];
                table_distance.clear();
                patch_table[k_r].clear();

                //! Threshold distances in order to keep similar patches
                for (int dj = -(int) nHW_sim; dj <= (int) nHW_sim; dj++)
                {
                    for (int di = 0; di <= (int) nHW_sim; di++)
                        if (sum_table[dj + nHW_sim + di * Ns][k_r] < threshold)
                            table_distance.push_back(make_pair(
                                        sum_table[dj + nHW_sim + di * Ns][k_r]
                                    , k_r + di * width + dj));

                    for (int di = - (int) nHW_sim; di < 0; di++)
                        if (sum_table[-dj + nHW_sim + (-di) * Ns][k_r] < threshold)
                            table_distance.push_back(make_pair(
                                        sum_table[-dj + nHW_sim + (-di) * Ns][k_r + di * width + dj]
                                    , k_r + di * width + dj));
                }

                //! We need a power of 2 for the number of similar patches,
                //! because of the Welsh-Hadamard transform on the third dimension.
                //! We assume that NHW is already a power of 2
                const unsigned nSx_r = (NHW > table_distance.size() ?
                                        closest_power_of_2(table_distance.size()) : NHW);

                //! To avoid problem
                if (nSx_r == 1 && table_distance.size() == 0)
                {
                    table_distance.push_back(make_pair(0, k_r));
                }

                //! Sort patches according to their distance to the reference one
                partial_sort(table_distance.begin(), table_distance.begin() + nSx_r,
                                                table_distance.end(), ComparaisonFirst);

                //! Keep a maximum of NHW similar patches
                for (unsigned n = 0; n < nSx_r; n++)
                    patch_table[k_r].push_back(table_distance[n].second);

                //! To avoid problem
                if (nSx_r == 1)
                    patch_table[k_r].push_back(table_distance[0].second);
            }
        }
    }
    else
    {
        for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
        {
            for (unsigned ind_j = 0; ind_j < column_ind_per_row[ind_i].size(); ind_j++)
            {
                //! Initialization
                const unsigned k_r = row_ind[ind_i] * width + column_ind_per_row[ind_i][ind_j];
                patch_table[k_r].clear();
                patch_table[k_r].push_back(k_r);
            }
        }
    }
}

/**
 * @brief Precompute Bloc Matching between a pair of images (distance inter-patches) only for patches which need denoising
 *
 * @param patch_table: for each patch in img1, will contain all coordinnates of its similar patches in img2;
 * @param shape_table: mask indicating if found similar patch distance is below (1) or above (0) tauMatch;
 * @param img1, img2: pair of noisy image on which the distance is computed;
 * @param width, height: size of img (including borders);
 * @param kHW: size of patch;
 * @param NHW: maximum similar patches wanted;
 * @param nHW: size of the boundary of img;
 * @param nHW_disp: half of the search window for disparity, smaller or equal to nHW;
 * @param tauMatch: threshold used to determinate similarity between patches;
 * @param row_ind, column_ind_per_row: contains indexes of patches which need denoising.
 *
 * @return none.
 **/
int precompute_BM_stereo(
    vector<vector<unsigned> > &patch_table
,   vector<unsigned> &shape_table
,   const vector<float> &img1
,   const vector<float> &img2
,   const unsigned width
,   const unsigned height
,   const unsigned kHW
,   const unsigned nHW
,   const unsigned nHW_disp
,   const float    tauMatch
,   const vector<unsigned> row_ind
,   const vector<vector<unsigned> > column_ind_per_row
){
    //! Check allocation memory
    if (img1.size() != img2.size())
    {
        cout << "The two images for stereo BM should have the same size: " << img1.size() << " != " << img2.size() << endl;
        return EXIT_FAILURE;
    }

    //! Declarations
    const unsigned nHW_sim = nHW - nHW_disp;
    const float threshold = tauMatch * kHW * kHW;
    const unsigned Ns_disp = 2 * nHW_disp + 1;
    vector<float> diff_table(width * height);
    vector<vector<float> > sum_table(Ns_disp * Ns_disp, vector<float> (width * height));
    if (patch_table.size() != width * height)
        patch_table.resize(width * height);
    if (shape_table.size() != width * height)
        shape_table.resize(width * height);
    for(unsigned k = 0; k < width * height; k++)
        patch_table[k].clear();

    //! For each possible distance, precompute inter-patches distance
    for (unsigned di = 0; di < Ns_disp; di++)
        for (unsigned dj = 0; dj < Ns_disp; dj++)
        {
            const int dk = (int) (di * width + dj) - (int) (nHW_disp * (1 + width));
            const unsigned ddk = di * Ns_disp + dj;

            //! Process the image containing the square distance between pixels
            for (unsigned i = nHW_disp; i < height - nHW_disp; i++)
            {
                unsigned k = i * width + nHW_disp;
                for (unsigned j = nHW_disp; j < width - nHW_disp; j++, k++)
                    diff_table[k] = (img2[k + dk] - img1[k]) * (img2[k + dk] - img1[k]);
            }

            //! Compute the sum for each patches, using the method of the integral images
            const unsigned dn = nHW_disp * width + nHW_disp;
            //! 1st patch, top left corner
            float value = 0.0f;
            for (unsigned p = 0; p < kHW; p++)
            {
                unsigned pq = p * width + dn;
                for (unsigned q = 0; q < kHW; q++, pq++)
                    value += diff_table[pq];
            }
            sum_table[ddk][dn] = value;

            //! 1st row, top
            for (unsigned j = nHW_disp + 1; j < width - nHW_disp - kHW + 1; j++)
            {
                const unsigned ind = nHW_disp * width + j - 1;
                float sum = sum_table[ddk][ind];
                for (unsigned p = 0; p < kHW; p++)
                    sum += diff_table[ind + p * width + kHW] - diff_table[ind + p * width];
                sum_table[ddk][ind + 1] = sum;
            }

            //! General case
            for (unsigned i = nHW_disp + 1; i < height - nHW_disp - kHW + 1; i++)
            {
                const unsigned ind = (i - 1) * width + nHW_disp;
                float sum = sum_table[ddk][ind];
                //! 1st column, left
                for (unsigned q = 0; q < kHW; q++)
                    sum += diff_table[ind + kHW * width + q] - diff_table[ind + q];
                sum_table[ddk][ind + width] = sum;

                //! Other columns
                unsigned k = i * width + nHW_disp + 1;
                unsigned pq = (i + kHW - 1) * width + kHW - 1 + nHW_disp + 1;
                for (unsigned j = nHW_disp + 1; j < width - nHW_disp - kHW + 1; j++, k++, pq++)
                {
                    sum_table[ddk][k] =
                          sum_table[ddk][k - 1]
                        + sum_table[ddk][k - width]
                        - sum_table[ddk][k - 1 - width]
                        + diff_table[pq]
                        - diff_table[pq - kHW]
                        - diff_table[pq - kHW * width]
                        + diff_table[pq - kHW - kHW * width];
                }
            }
        }

    //! Precompute Bloc Matching
    vector<pair<float, unsigned> > table_distance;
    //! To avoid reallocation
    table_distance.reserve(Ns_disp * Ns_disp);

    for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
    {
        for (unsigned ind_j = 0; ind_j < column_ind_per_row[ind_i].size(); ind_j++)
        {
            //! Make sure that all possible kNN of the current patch have a matching block
            for(int i = -(int) nHW_sim; i <= (int) nHW_sim; i++)
                for(int j = -(int) nHW_sim; j <= (int) nHW_sim; j++)
                {
                    //! Initialization
                    const unsigned k_r = (row_ind[ind_i] + i) * width + column_ind_per_row[ind_i][ind_j] + j;
                    if(patch_table[k_r].size()) continue; //! Already computed
                    table_distance.clear();
                    

                    //! Create pairs of distance / index
                    for (int dj = -(int) nHW_disp; dj <= (int) nHW_disp; dj++)
                    {
                        for (int di = - (int) nHW_disp; di <= (int) nHW_disp; di++)
                            table_distance.push_back(make_pair(
                                        sum_table[dj + nHW_disp + (di + nHW_disp) * Ns_disp][k_r]
                                        , k_r + di * width + dj));
                    }

                    //! Sort patches according to their distance to the reference one
                    sort(table_distance.begin(), table_distance.end(), ComparaisonFirst);
                    
                    //! Store sorted patches in table
                    for (unsigned n = 0; n < table_distance.size(); n++)
                        patch_table[k_r].push_back(table_distance[n].second);

                    //! Evaluate shape
                    shape_table[k_r] = table_distance[0].first < threshold ? 1 : 0;
                }
        }
    }
    return EXIT_SUCCESS;
}



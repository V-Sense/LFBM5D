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
 * @file bm5d.cpp
 * @brief LFBM5D denoising functions (step 1 - Hard Thresholding and 2 - Wiener filtering) - main outer loop on SAIs
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

 #include "bm5d.h"
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
 * @brief run BM5D hard thresholding step on the noisy light field. Depending on if OpenMP is used or not,
 *        and on the number of available threads, it divides the noisy
 *        SAI in sub_images, to process them in parallel.
 *
 * @param sigma: value of assumed noise of the noisy image;
 * @param lambdaHard5D: Threshold for Hard Thresholding;
 * @param LF_noisy: noisy light field;
 * @param LF_SAI_mask: indicate if SAI are empty (0) or not (1);
 * @param LF_basic: will be the basic estimation after the 1st step;
 * @param ang_major: scanning order of SAI in LF (row or column);
 * @param awidth, aheight, width, height, chnls: size of the LF;
 * @param anHard: half size of the angular search window;
 * @param NHard: number of nearest neighbor for a patch;
 * @param nSim, nDisp: Half size of the search window for self-similarities and disparity respectively;
 * @param kHard: Patches size;
 * @param pHard: Processing step on row and columns;
 * @param useSD: if true, use weight based
 *        on the standard variation of the 5D group for the
 *        first (resp. second) step, otherwise use the number
 *        of non-zero coefficients after Hard Thresholding
 *        (resp. the norm of Wiener coefficients);
 * @param tau_2D, tau_4D, tau_5D: successive transform to apply on every 5D group
 *        Allowed values are ID, DCT and BIOR / ID, DCT, SADCT / and HADAMARD, HAAR and DCT respectively;
 * @param color_space: Transformation from RGB to YUV. Allowed
 *        values are RGB (do nothing), YUV, YCBCR and OPP;
 * @param nb_threads: number of threads for OpenMP parallel processing;
 *
 * @return EXIT_FAILURE if color_space has not expected
 *         type, otherwise return EXIT_SUCCESS.
 **/
int run_bm5d_1st_step(
    const float sigma
,   const float lambdaHard5D
,   vector<vector<float> > &LF_noisy
,   vector<unsigned> &LF_SAI_mask
,   vector<vector<float> > &LF_basic
,   const unsigned ang_major
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned anHard
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned NHard
,   const unsigned nSim
,   const unsigned nDisp
,   const unsigned kHard
,   const unsigned pHard
,   const bool     useSD
,   const unsigned tau_2D
,         unsigned tau_4D
,   const unsigned tau_5D
,   const unsigned color_space
,   const unsigned nb_threads
){
    //! Parameters
    //! Angular parameters
    const unsigned asize = awidth * aheight;
    const unsigned cs  = aheight/2; //! Index of center SAI
    const unsigned ct  = awidth /2; //! Index of center SAI
    const unsigned cst = ang_major==ROWMAJOR ? cs * awidth + ct : cs + ct * aheight;
    const unsigned asize_sw = 2*anHard + 1;
    if(asize_sw > aheight || asize_sw > awidth)
    {
        cout << "Wrong size of angular search window, the angular search window must be smaller than the light field angular size." << endl;
        return EXIT_FAILURE;
    }
    //! SAI search windows
    const unsigned nHard = nSim + nDisp;

    //! Check memory allocation
    if (LF_basic.size() != asize)
        LF_basic.resize(asize);

    //! Transformation to YUV color space
    if (color_space_transform_LF(LF_noisy, LF_SAI_mask, color_space, width, height, chnls, true)
        != EXIT_SUCCESS) return EXIT_FAILURE;

    cout << endl << "Number of threads which will be used: " << nb_threads;
#ifdef _OPENMP
    cout << " (real available cores: " << omp_get_num_procs() << ")" << endl;
#else
    cout << endl;
#endif
    cout << endl;

    //! Allocate plan for FFTW library
    fftwf_plan* plan_2d_for_1 = new fftwf_plan[nb_threads];
  	fftwf_plan* plan_2d_for_2 = new fftwf_plan[nb_threads];
  	fftwf_plan* plan_2d_for_3 = new fftwf_plan[nb_threads];
  	fftwf_plan* plan_2d_inv = new fftwf_plan[nb_threads];
  	fftwf_plan* plan_4d = new fftwf_plan[nb_threads];
  	fftwf_plan* plan_4d_inv = new fftwf_plan[nb_threads];
  	fftwf_plan** plan_4d_sa = new fftwf_plan*[nb_threads]; //! Assign a 1d transform for each possible size between 2 and angular patch size
  	fftwf_plan** plan_4d_sa_inv = new fftwf_plan*[nb_threads];
  	for (unsigned int i = 0; i < nb_threads; i++) {
  		plan_4d_sa[i] = new fftwf_plan[asize_sw - 1];
  		plan_4d_sa_inv[i] = new fftwf_plan[asize_sw - 1];
  	}
  	fftwf_plan* plan_5d = new fftwf_plan[nb_threads];
  	fftwf_plan* plan_5d_inv = new fftwf_plan[nb_threads];

    //! Declaration of results buffer
    vector<vector<float> >  LF_basic_num(asize, vector<float>(height * width * chnls, 0.0f)), //! Numerator (accumulation buffer of estimates)
                            LF_basic_den(asize, vector<float>(height * width * chnls, 0.0f)); //! Denominator (aggregation weights)

    //! In the simple case
    if (nb_threads == 1)
    {
        //! Denoising, 1st Step
        const unsigned h_b = height + 2 * nHard;
        const unsigned w_b = width  + 2 * nHard;

        vector<unsigned> procSAI(asize); //! To keep track of the processed SAIs - 0 = unprocessed, >0 = processed
        for(unsigned st = 0; st < asize; st++) //! Do not process empty SAI
            procSAI[st] = !LF_SAI_mask[st];

        unsigned countProcSAI = count(procSAI.begin(), procSAI.end(), 0);
        const unsigned maxProcSAI = count(procSAI.begin(), procSAI.end(), 0);
        unsigned ps,pt,pst; //! angular indexes of the SAI to process

        while(countProcSAI) //! Loop on non-processed SAI
        {
            //! Determine the SAI to process
            if(countProcSAI == maxProcSAI && LF_SAI_mask[cst]) //! Start with the center SAI (it is assumed that the center SAI is not empty)
            {
                ps = cs;
                pt = ct;
            }
            else //! Process the SAI with the most pixels to denoised
            {
                int countNoisyPix = -1;
                #pragma omp parallel for
                for(int st = 0; st < asize; st++)
                {
                    if(procSAI[st] == 0)
                    {
                        int countNoisyPixst = count(LF_basic_den[st].begin(), LF_basic_den[st].end(), 0.0);
                        if(countNoisyPixst >= countNoisyPix)
                        {
                            pst = st;
                            countNoisyPix = countNoisyPixst;
                        }
                    }
                }
                if(ang_major == ROWMAJOR)
                {
                    ps = pst / awidth;
                    pt = pst - ps * awidth;
                }
                else if(ang_major == COLMAJOR)
                {
                    pt = pst / aheight;
                    ps = pst - pt * aheight;
                }
            }

            //! Compute angular search window around current SAI
            int cs_asw, min_s_asw, max_s_asw, ct_asw, min_t_asw, max_t_asw;
            compute_LF_angular_search_window(cs_asw, min_s_asw, max_s_asw, ps, aheight, anHard);
            compute_LF_angular_search_window(ct_asw, min_t_asw, max_t_asw, pt, awidth , anHard);

            unsigned cst_asw;
            if(ang_major==ROWMAJOR)
                cst_asw = cs_asw * asize_sw + ct_asw;
            else if(ang_major==COLMAJOR)
                cst_asw = cs_asw + ct_asw * asize_sw;

            vector<unsigned> st_idx_asw(asize_sw * asize_sw);
            #pragma omp parallel for
            for(int s_asw = 0; s_asw < asize_sw; s_asw++)
            {
                unsigned s = s_asw + min_s_asw;
                for(unsigned  t_asw = 0; t_asw < asize_sw; t_asw++)
                {
                    unsigned t = t_asw + min_t_asw;
                    unsigned st, st_asw;
                    if(ang_major==ROWMAJOR)
                    {
                        st     = s * awidth + t;
                        st_asw = s_asw * asize_sw + t_asw;
                    }
                    else if(ang_major==COLMAJOR)
                    {
                        st     = s + t * aheight;
                        st_asw = s_asw + t_asw * asize_sw;
                    }
                    st_idx_asw[st_asw] = st;
                }
            }

            cout << "-> Angular search window = (" << min_t_asw << ", " << min_s_asw << ") to (" << max_t_asw << ", " << max_s_asw << ")" << endl;

            //! Add boundaries and symetrize them
            vector<unsigned> LF_SAI_mask_asw(asize_sw * asize_sw);
            vector<vector<float> > LF_sym_noisy_asw(asize_sw * asize_sw), LF_sym_basic_num_asw(asize_sw * asize_sw), LF_sym_basic_den_asw(asize_sw * asize_sw);
            #pragma omp parallel for
            for(int st_asw = 0; st_asw < st_idx_asw.size(); st_asw++)
            {
                unsigned st = st_idx_asw[st_asw];
                LF_SAI_mask_asw[st_asw] = LF_SAI_mask[st];
                if(LF_SAI_mask[st])
                {
                    symetrize(LF_noisy[st],     LF_sym_noisy_asw[st_asw],     width, height, chnls, nHard);
                    symetrize(LF_basic_num[st], LF_sym_basic_num_asw[st_asw], width, height, chnls, nHard);
                    symetrize(LF_basic_den[st], LF_sym_basic_den_asw[st_asw], width, height, chnls, nHard);
                }
            }

            //! Initialization of SAI to process
            vector<unsigned> procSAI_asw(asize_sw * asize_sw); //! To keep track of the processed SAIs - 0 = unprocessed, >0 = processed
            for(unsigned st_asw = 0; st_asw < asize_sw * asize_sw; st_asw++) //! Do not process empty SAI
                procSAI_asw[st_asw] = !LF_SAI_mask_asw[st_asw];

            unsigned countProcSAI_asw = count(procSAI_asw.begin(), procSAI_asw.end(), 0);
            const unsigned maxProcSAI_asw = count(procSAI_asw.begin(), procSAI_asw.end(), 0);
            unsigned ps_asw, pt_asw, pst_asw;

            if(maxProcSAI_asw != asize_sw * asize_sw && tau_4D == DCT)
            {
                cout << "Warning ! Some SAI are empty, 4D transform changed to SADCT." << endl;
                tau_4D = SADCT;
            }

            //! Allocating Plan for FFTW process
            if (tau_2D == DCT)
            {
                const unsigned nb_cols = ind_size(w_b - kHard + 1, nHard, pHard);
                allocate_plan_2d(&plan_2d_for_1[0], kHard, FFTW_REDFT10,
                                                            w_b * (2 * nHard + 1) * chnls);
                allocate_plan_2d(&plan_2d_for_2[0], kHard, FFTW_REDFT10,
                                                            w_b * pHard * chnls);
                allocate_plan_2d(&plan_2d_for_3[0], kHard, FFTW_REDFT10,
                                                            (2 * nHard + 1) * (2 * nHard + 1) * chnls);
                allocate_plan_2d(&plan_2d_inv[0],   kHard, FFTW_REDFT01,
                                                            NHard * nb_cols * chnls);
            }
            if (tau_4D == DCT || tau_4D == SADCT)
            {
                allocate_plan_2d(&plan_4d[0],     asize_sw, asize_sw, FFTW_REDFT10, kHard * kHard * chnls);
                allocate_plan_2d(&plan_4d_inv[0], asize_sw, asize_sw, FFTW_REDFT01, kHard * kHard * chnls);
            }
            if(tau_4D == SADCT)
            {
                //! Allocate 1d DCT for each possible size between 2 and angular patch size
                for(unsigned k = 0; k < (asize_sw-1); k++)
                {
                    allocate_plan_1d(&plan_4d_sa[0][k],     k+2, FFTW_REDFT10, kHard * kHard * chnls);
                    allocate_plan_1d(&plan_4d_sa_inv[0][k], k+2, FFTW_REDFT01, kHard * kHard * chnls);
                }
            }

            //! Sub-loop on non-processed SAI in angular search window
            while(countProcSAI_asw)
            {
                if(countProcSAI_asw == maxProcSAI_asw && LF_SAI_mask_asw[cst_asw]) //! Start with central SAI of angular search window
                {
                    ps_asw  = cs_asw;
                    pt_asw  = ct_asw;
                    pst_asw = cst_asw;
                }
                else //! Process the SAI with the most pixels to denoised
                {
                    int countNoisyPix = -1;
                    #pragma omp parallel for
                    for(int st = 0; st < (asize_sw * asize_sw); st++)
                    {
                        if(procSAI_asw[st] == 0)
                        {
                            int countNoisyPixst = count(LF_sym_basic_den_asw[st].begin(), LF_sym_basic_den_asw[st].end(), 0.0);
                            if(countNoisyPixst >= countNoisyPix)
                            {
                                pst_asw = st;
                                countNoisyPix = countNoisyPixst;
                            }
                        }
                    }
                    if(ang_major == ROWMAJOR)
                    {
                        ps_asw = pst_asw / asize_sw;
                        pt_asw = pst_asw - ps_asw * asize_sw;
                    }
                    else if(ang_major == COLMAJOR)
                    {
                        pt_asw = pst_asw / asize_sw;
                        ps_asw = pst_asw - pt_asw * asize_sw;
                    }
                }

                //! Run BM5D first step
                cout << "\t-> Processing SAI (t, s) = (" << (min_t_asw + pt_asw) << ", " << (min_s_asw + ps_asw) << ") ... " << flush; // endl; //
                float BM_elapsed_secs;
                timestamp_t start_step1_st = get_timestamp();
                bm5d_1st_step(sigma, lambdaHard5D, LF_sym_noisy_asw, LF_sym_basic_num_asw, LF_sym_basic_den_asw, LF_SAI_mask_asw, procSAI_asw, cst_asw, pst_asw,
                                asize_sw, asize_sw, w_b, h_b, chnls, nSim, nDisp,
                                kHard, NHard, pHard, useSD, color_space, tau_2D, tau_4D, tau_5D,
                                &plan_2d_for_1[0], &plan_2d_for_2[0], &plan_2d_for_3[0], &plan_2d_inv[0],
                                &plan_4d[0], &plan_4d_inv[0], plan_4d_sa[0], plan_4d_sa_inv[0], &plan_5d[0], &plan_5d_inv[0], BM_elapsed_secs);
                timestamp_t end_step1_st = get_timestamp();
                float step1_st_elapsed_secs = float(end_step1_st-start_step1_st) / 1000000.0f;
                cout << " done in " << step1_st_elapsed_secs << " secs. (BM time = " << BM_elapsed_secs << " secs)" << endl;

                //! Update non-processed SAIs count
                procSAI_asw[pst_asw] += 1;
                unsigned st;
                if(ang_major==ROWMAJOR)
                    st     = (min_s_asw + ps_asw) * awidth + (min_t_asw + pt_asw);
                else if(ang_major==COLMAJOR)
                    st     = (min_s_asw + ps_asw) + (min_t_asw + pt_asw) * aheight;
                procSAI[st] += 1;

                //! Break loop if all SAIs in search window are denoised
                float LF_dn_pct = LF_denoised_percent(LF_sym_basic_den_asw, LF_SAI_mask_asw, width, height, chnls, nHard, kHard);
                if(LF_dn_pct >= 100.0f)
                {
                    for(unsigned st_asw = 0; st_asw < st_idx_asw.size(); st_asw++)
                    {
                        unsigned st = st_idx_asw[st_asw];
                        if(procSAI_asw[st_asw] == 0)
                        {
                             procSAI_asw[st_asw] += 1;
                             procSAI[st] += 1;
                        }
                    }
                }

                countProcSAI_asw = count(procSAI_asw.begin(), procSAI_asw.end(), 0);
            }

            //! Remove boundaries
            for(unsigned st_asw = 0; st_asw < st_idx_asw.size(); st_asw++)
            {
                unsigned st = st_idx_asw[st_asw];
                if(LF_SAI_mask[st])
                {
                    unsymetrize(LF_basic_num[st], LF_sym_basic_num_asw[st_asw], width, height, chnls, nHard);
                    unsymetrize(LF_basic_den[st], LF_sym_basic_den_asw[st_asw], width, height, chnls, nHard);
                }
            }

            countProcSAI = count(procSAI.begin(), procSAI.end(), 0);

            //! Display progress
            cout << countProcSAI << " SAI(s) remaining to process." << endl;
        }

        //! Get final basic estimate
        if(compute_LF_estimate(LF_SAI_mask, LF_basic_num, LF_basic_den, LF_noisy, LF_basic, asize) != EXIT_SUCCESS)
            return EXIT_FAILURE;
    }
    //* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //! If more than 1 threads are used
    //* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    else
    {
        vector<unsigned> procSAI(asize); //! To keep track of the processed SAIs - 0 = unprocessed, >0 = processed
        for(unsigned st = 0; st < asize; st++) //! Do not process empty SAI
            procSAI[st] = !LF_SAI_mask[st];

        unsigned countProcSAI = count(procSAI.begin(), procSAI.end(), 0);
        const unsigned maxProcSAI = count(procSAI.begin(), procSAI.end(), 0);
        unsigned ps,pt,pst; //! angular indexes of the SAI to process

        //! Sub-divide LF noisy in nb_threads
        vector<vector<vector<float> > > LF_sub_noisy(nb_threads), LF_sub_basic_num(nb_threads), LF_sub_basic_den(nb_threads);
        vector<unsigned> h_table(nb_threads), w_table(nb_threads);
        #pragma omp parallel for
        for(int n = 0; n < nb_threads; n++)
        {
            LF_sub_noisy[n].resize(asize);
            LF_sub_basic_num[n].resize(asize);
            LF_sub_basic_den[n].resize(asize);
        }

        sub_divide_LF(LF_noisy, LF_SAI_mask, LF_sub_noisy, w_table, h_table, awidth, aheight, width, height, chnls, nHard);

        while(countProcSAI) //! Loop on non-processed SAI
        {
            //! Determine the SAI to process
            if(countProcSAI == maxProcSAI && LF_SAI_mask[cst]) //! Start with the center SAI (it is assumed that the center SAI is not empty)
            {
                ps = cs;
                pt = ct;
            }
            else //! Process the SAI with the most pixels to denoised
            {
                int countNoisyPix = -1;
                #pragma omp parallel for
                for(int st = 0; st < asize; st++)
                {
                    if(procSAI[st] == 0)
                    {
                        int countNoisyPixst = count(LF_basic_den[st].begin(), LF_basic_den[st].end(), 0.0);
                        if(countNoisyPixst >= countNoisyPix)
                        {
                            pst = st;
                            countNoisyPix = countNoisyPixst;
                        }
                    }
                }
                if(ang_major == ROWMAJOR)
                {
                    ps = pst / awidth;
                    pt = pst - ps * awidth;
                }
                else if(ang_major == COLMAJOR)
                {
                    pt = pst / aheight;
                    ps = pst - pt * aheight;
                }
            }

            //! Compute angular search window around current SAI
            int cs_asw, min_s_asw, max_s_asw, ct_asw, min_t_asw, max_t_asw;
            compute_LF_angular_search_window(cs_asw, min_s_asw, max_s_asw, ps, aheight, anHard);
            compute_LF_angular_search_window(ct_asw, min_t_asw, max_t_asw, pt, awidth , anHard);

            unsigned cst_asw;
            if(ang_major==ROWMAJOR)
                cst_asw = cs_asw * asize_sw + ct_asw;
            else if(ang_major==COLMAJOR)
                cst_asw = cs_asw + ct_asw * asize_sw;

            vector<unsigned> st_idx_asw(asize_sw * asize_sw);
            #pragma omp parallel for
            for(int s_asw = 0; s_asw < asize_sw; s_asw++)
            {
                unsigned s = s_asw + min_s_asw;
                for(unsigned  t_asw = 0; t_asw < asize_sw; t_asw++)
                {
                    unsigned t = t_asw + min_t_asw;
                    unsigned st, st_asw;
                    if(ang_major==ROWMAJOR)
                    {
                        st     = s * awidth + t;
                        st_asw = s_asw * asize_sw + t_asw;
                    }
                    else if(ang_major==COLMAJOR)
                    {
                        st     = s + t * aheight;
                        st_asw = s_asw + t_asw * asize_sw;
                    }
                    st_idx_asw[st_asw] = st;
                }
            }

            cout << "-> Angular search window = (" << min_t_asw << ", " << min_s_asw << ") to (" << max_t_asw << ", " << max_s_asw << ")" << endl;

            //! Sub-divide current estimate
            sub_divide_LF(LF_basic_num, LF_SAI_mask, LF_sub_basic_num, w_table, h_table, awidth, aheight, width, height, chnls, nHard);
            sub_divide_LF(LF_basic_den, LF_SAI_mask, LF_sub_basic_den, w_table, h_table, awidth, aheight, width, height, chnls, nHard);

            //! Get subdivision for angular search window
            vector<vector<float> > LF_basic_den_asw(asize_sw * asize_sw), LF_basic_num_asw(asize_sw * asize_sw); //! Convenience to update subdivision in the loop
            vector<vector<vector<float> > > LF_sub_noisy_asw(nb_threads), LF_sub_basic_num_asw(nb_threads), LF_sub_basic_den_asw(nb_threads);
            vector<unsigned> LF_SAI_mask_asw(asize_sw * asize_sw);

            for(unsigned n = 0; n < nb_threads; n++)
            {
                LF_sub_noisy_asw[n].resize(asize_sw * asize_sw);
                LF_sub_basic_num_asw[n].resize(asize_sw * asize_sw);
                LF_sub_basic_den_asw[n].resize(asize_sw * asize_sw);

                #pragma omp parallel for
                for(int st_asw = 0; st_asw < st_idx_asw.size(); st_asw++)
                {
                    unsigned st = st_idx_asw[st_asw];
                    LF_SAI_mask_asw[st_asw] = LF_SAI_mask[st];
                    if(LF_SAI_mask[st])
                    {
                        LF_sub_noisy_asw[n][st_asw] = LF_sub_noisy[n][st];
                        LF_sub_basic_num_asw[n][st_asw] = LF_sub_basic_num[n][st];
                        LF_sub_basic_den_asw[n][st_asw] = LF_sub_basic_den[n][st];
                    }
                }
            }

            //! Initialization of SAI to process
            vector<unsigned> procSAI_asw(asize_sw * asize_sw); //! To keep track of the processed SAIs - 0 = unprocessed, >0 = processed
            for(unsigned st_asw = 0; st_asw < asize_sw * asize_sw; st_asw++) //! Do not process empty SAI
                procSAI_asw[st_asw] = !LF_SAI_mask_asw[st_asw];

            unsigned countProcSAI_asw = count(procSAI_asw.begin(), procSAI_asw.end(), 0);
            const unsigned maxProcSAI_asw = count(procSAI_asw.begin(), procSAI_asw.end(), 0);
            unsigned ps_asw, pt_asw, pst_asw;

            if(maxProcSAI_asw != asize_sw * asize_sw && tau_4D == DCT)
            {
                cout << "Warning ! Some SAI are empty, 4D transform changed to SADCT." << endl;
                tau_4D = SADCT;
            }

            //! Allocating Plan for FFTW process
            if (tau_2D == DCT)
                for(unsigned n = 0; n < nb_threads; n++)
                {
                    const unsigned nb_cols = ind_size(w_table[n] - kHard + 1, nHard, pHard);
                    allocate_plan_2d(&plan_2d_for_1[n], kHard, FFTW_REDFT10,
                                                                w_table[n] * (2 * nHard + 1) * chnls);
                    allocate_plan_2d(&plan_2d_for_2[n], kHard, FFTW_REDFT10,
                                                                w_table[n] * pHard * chnls);
                    allocate_plan_2d(&plan_2d_for_3[n], kHard, FFTW_REDFT10,
                                                                (2 * nHard + 1) * (2 * nHard + 1) * chnls);
                    allocate_plan_2d(&plan_2d_inv[n],   kHard, FFTW_REDFT01,
                                                                NHard * nb_cols * chnls);
                }
            if (tau_4D == DCT || tau_4D == SADCT)
                for(unsigned n = 0; n < nb_threads; n++)
                {
                    allocate_plan_2d(&plan_4d[n],     asize_sw, asize_sw, FFTW_REDFT10, kHard * kHard * chnls);
                    allocate_plan_2d(&plan_4d_inv[n], asize_sw, asize_sw, FFTW_REDFT01, kHard * kHard * chnls);
                }
            if(tau_4D == SADCT)
            {
                //! Allocate 1d DCT for each possible size between 2 and angular patch size
                for(unsigned n = 0; n < nb_threads; n++)
                    for(unsigned k = 0; k < (asize_sw-1); k++)
                    {
                        allocate_plan_1d(&plan_4d_sa[n][k],     k+2, FFTW_REDFT10, kHard * kHard * chnls);
                        allocate_plan_1d(&plan_4d_sa_inv[n][k], k+2, FFTW_REDFT01, kHard * kHard * chnls);
                    }
            }

            //! Sub-loop on non-processed SAI in angular search window
            while(countProcSAI_asw)
            {
                if(countProcSAI_asw == maxProcSAI_asw && LF_SAI_mask_asw[cst_asw]) //! Start with central SAI of angular search window
                {
                    ps_asw  = cs_asw;
                    pt_asw  = ct_asw;
                    pst_asw = cst_asw;
                }
                else //! Process the SAI with the most pixels to denoised
                {
                    int countNoisyPix = -1;
                    #pragma omp parallel for
                    for(int st = 0; st < (asize_sw * asize_sw); st++)
                    {
                        if(procSAI_asw[st] == 0)
                        {
                            int countNoisyPixst = 0;
                            for(unsigned n = 0; n < nb_threads; n++)
                                countNoisyPixst += count(LF_sub_basic_den_asw[n][st].begin(), LF_sub_basic_den_asw[n][st].end(), 0.0);

                            if(countNoisyPixst >= countNoisyPix)
                            {
                                pst_asw = st;
                                countNoisyPix = countNoisyPixst;
                            }
                        }
                    }
                    if(ang_major == ROWMAJOR)
                    {
                        ps_asw = pst_asw / asize_sw;
                        pt_asw = pst_asw - ps_asw * asize_sw;
                    }
                    else if(ang_major == COLMAJOR)
                    {
                        pt_asw = pst_asw / asize_sw;
                        ps_asw = pst_asw - pt_asw * asize_sw;
                    }
                }

                //! Run BM5D first step
                cout << "\t-> Processing SAI (t, s) = (" << (min_t_asw + pt_asw) << ", " << (min_s_asw + ps_asw) << ") ... " << flush; //endl;
                vector<float> BM_elapsed_secs(nb_threads);
                timestamp_t start_step1_st = get_timestamp();
                #pragma omp parallel shared(LF_sub_noisy_asw, LF_sub_basic_num_asw, LF_sub_basic_den_asw, w_table, h_table, \
                                    plan_2d_for_1, plan_2d_for_2, plan_2d_for_3, plan_2d_inv, plan_4d, plan_4d_inv, plan_5d, plan_5d_inv, BM_elapsed_secs)
                {
                    #pragma omp for schedule(dynamic) nowait
                    for (int n = 0; n < nb_threads; n++)
                    {
                        bm5d_1st_step(sigma, lambdaHard5D, LF_sub_noisy_asw[n], LF_sub_basic_num_asw[n], LF_sub_basic_den_asw[n], LF_SAI_mask_asw, procSAI_asw, cst_asw, pst_asw,
                                        asize_sw, asize_sw, w_table[n], h_table[n], chnls, nSim, nDisp,
                                        kHard, NHard, pHard, useSD, color_space, tau_2D, tau_4D, tau_5D,
                                        &plan_2d_for_1[n], &plan_2d_for_2[n], &plan_2d_for_3[n], &plan_2d_inv[n],
                                        &plan_4d[n], &plan_4d_inv[n], plan_4d_sa[n], plan_4d_sa_inv[n], &plan_5d[n], &plan_5d_inv[n], BM_elapsed_secs[n]);
                    }
                }
                timestamp_t end_step1_st = get_timestamp();
                float step1_st_elapsed_secs = float(end_step1_st-start_step1_st) / 1000000.0f;
                float nb_BM = (float)(nb_threads - count(BM_elapsed_secs.begin(), BM_elapsed_secs.end(), 0.0f));
                float avg_BM_elapsed_secs = nb_BM > 0.0 ? accumulate(BM_elapsed_secs.begin(), BM_elapsed_secs.end(), 0.0f) / nb_BM : 0.0;
                cout << " done in " << step1_st_elapsed_secs << " secs. (BM average time = " << avg_BM_elapsed_secs << " secs)" << endl;

                //! Update overlap between subdivision
                undivide_LF(LF_sub_basic_den_asw, LF_sub_basic_num_asw, LF_basic_den_asw, LF_basic_num_asw, LF_SAI_mask_asw,
                            asize_sw, asize_sw, width, height, chnls, nHard);
                sub_divide_LF(LF_basic_den_asw, LF_SAI_mask_asw, LF_sub_basic_den_asw, w_table, h_table, asize_sw, asize_sw, width, height, chnls, nHard);
                sub_divide_LF(LF_basic_num_asw, LF_SAI_mask_asw, LF_sub_basic_num_asw, w_table, h_table, asize_sw, asize_sw, width, height, chnls, nHard);

                //! Update non-processed SAIs count
                procSAI_asw[pst_asw] += 1;
                unsigned st;
                if(ang_major==ROWMAJOR)
                    st     = (min_s_asw + ps_asw) * awidth + (min_t_asw + pt_asw);
                else if(ang_major==COLMAJOR)
                    st     = (min_s_asw + ps_asw) + (min_t_asw + pt_asw) * aheight;
                procSAI[st] += 1;

                //! Break loop if all SAIs in search window are denoised
                float LF_dn_pct = 0.0f;
                #pragma omp parallel for
                for (int n = 0; n < nb_threads; n++)
                    LF_dn_pct += LF_denoised_percent(LF_sub_basic_den_asw[n], LF_SAI_mask_asw, w_table[n] - 2 * nHard, h_table[n] - 2 * nHard, chnls, nHard, kHard);
                if(LF_dn_pct >= (100.0f * (float)(nb_threads)))
                {
                    for(unsigned st_asw = 0; st_asw < st_idx_asw.size(); st_asw++)
                    {
                        unsigned st = st_idx_asw[st_asw];
                        if(procSAI_asw[st_asw] == 0)
                        {
                             procSAI_asw[st_asw] += 1;
                             procSAI[st] += 1;
                        }
                    }
                }

                countProcSAI_asw = count(procSAI_asw.begin(), procSAI_asw.end(), 0);
            }

            //! Update sub-division
            for(unsigned n = 0; n < nb_threads; n++)
            {
                #pragma omp parallel for
                for(int st_asw = 0; st_asw < st_idx_asw.size(); st_asw++)
                {
                    unsigned st = st_idx_asw[st_asw];
                    LF_SAI_mask_asw[st_asw] = LF_SAI_mask[st];
                    if(LF_SAI_mask[st])
                    {
                        LF_sub_basic_num[n][st] = LF_sub_basic_num_asw[n][st_asw];
                        LF_sub_basic_den[n][st] = LF_sub_basic_den_asw[n][st_asw];
                    }
                }
            }

            //! Un-divide current estimate
            undivide_LF(LF_sub_basic_den, LF_sub_basic_num, LF_basic_den, LF_basic_num, LF_SAI_mask, awidth, aheight, width, height, chnls, nHard);

            countProcSAI = count(procSAI.begin(), procSAI.end(), 0);

            //! Display progress
            cout << countProcSAI << " SAI(s) remaining to process." << endl;
        }

        //! Get final basic estimate
        if(compute_LF_estimate(LF_SAI_mask, LF_basic_num, LF_basic_den, LF_noisy, LF_basic, asize) != EXIT_SUCCESS)
            return EXIT_FAILURE;
    }

    //! Inverse color space transform to RGB
    if (color_space_transform_LF(LF_basic, LF_SAI_mask, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform_LF(LF_noisy, LF_SAI_mask, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;

    //! Free Memory
    if (tau_2D == DCT)
        for (unsigned n = 0; n < nb_threads; n++)
        {
            fftwf_destroy_plan(plan_2d_for_1[n]);
            fftwf_destroy_plan(plan_2d_for_2[n]);
            fftwf_destroy_plan(plan_2d_for_3[n]);
            fftwf_destroy_plan(plan_2d_inv[n]);
        }
    if (tau_4D == DCT || tau_4D == SADCT)
        for (unsigned n = 0; n < nb_threads; n++)
        {
            fftwf_destroy_plan(plan_4d[n]);
            fftwf_destroy_plan(plan_4d_inv[n]);
        }
    if(tau_4D == SADCT)
        for (unsigned n = 0; n < nb_threads; n++)
            for(unsigned k = 0; k < (asize_sw-1); k++)
            {
                fftwf_destroy_plan(plan_4d_sa[n][k]);
                fftwf_destroy_plan(plan_4d_sa_inv[n][k]);
            }
    if(tau_5D == DCT)
        for (unsigned n = 0; n < nb_threads; n++)
        {
            fftwf_destroy_plan(plan_5d[n]);
            fftwf_destroy_plan(plan_5d_inv[n]);
        }
    fftwf_cleanup();

    return EXIT_SUCCESS;
}



/**
 * @brief run BM5D Wiener filtering step on the noisy light field. Depending on if OpenMP is used or not,
 *        and on the number of available threads, it divides the noisy
 *        SAI in sub_images, to process them in parallel.
 *
 * @param sigma: value of assumed noise of the noisy image;
 * @param LF_noisy: noisy light field;
 * @param LF_SAI_mask: indicate if SAI are empty (0) or not (1);
 * @param LF_basic: contains the basic estimation from the 1st step;
 * @param LF_denoised: will be the final estimation after the 2nd step;
 * @param ang_major: scanning order of SAI in LF (row or column);
 * @param awidth, aheight, width, height, chnls: size of the LF;
 * @param anWien: half size of the angular search window;
 * @param NWien: number of nearest neighbor for a patch;
 * @param nSim, nDisp: Half size of the search window for self-similarities and disparity respectively;
 * @param kWien: Patches size;
 * @param pWien: Processing step on row and columns;
 * @param useSD: if true, use weight based
 *        on the standard variation of the 5D group for the
 *        first (resp. second) step, otherwise use the number
 *        of non-zero coefficients after Hard Thresholding
 *        (resp. the norm of Wiener coefficients);
 * @param tau_2D, tau_4D, tau_5D: successive transform to apply on every 5D group
 *        Allowed values are ID, DCT and BIOR / ID, DCT, SADCT / and HADAMARD, HAAR and DCT respectively;
 * @param color_space: Transformation from RGB to YUV. Allowed
 *        values are RGB (do nothing), YUV, YCBCR and OPP;
 * @param nb_threads: number of threads for OpenMP parallel processing;
 *
 * @return EXIT_FAILURE if color_space has not expected
 *         type, otherwise return EXIT_SUCCESS.
 **/
int run_bm5d_2nd_step(
    const float sigma
,   vector<vector<float> > &LF_noisy
,   vector<unsigned> &LF_SAI_mask
,   vector<vector<float> > &LF_basic
,   vector<vector<float> > &LF_denoised
,   const unsigned ang_major
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned anWien
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned NWien
,   const unsigned nSim
,   const unsigned nDisp
,   const unsigned kWien
,   const unsigned pWien
,   const bool     useSD
,   const unsigned tau_2D
,         unsigned tau_4D
,   const unsigned tau_5D
,   const unsigned color_space
,   const unsigned nb_threads
){
    //! Parameters
    //! Angular parameters
    const unsigned asize = awidth * aheight;
    const unsigned cs  = aheight/2; //! Index of center SAI
    const unsigned ct  = awidth /2; //! Index of center SAI
    const unsigned cst = ang_major==ROWMAJOR ? cs * awidth + ct : cs + ct * aheight;
    const unsigned asize_sw = 2*anWien + 1;
    if(asize_sw > aheight || asize_sw > awidth)
    {
        cout << "Wrong size of angular search window, the angular search window must be smaller than the light field angular size." << endl;
        return EXIT_FAILURE;
    }
    //! SAI search windows
    const unsigned nWien = nSim + nDisp;

    //! Check memory allocation
    if (LF_denoised.size() != asize)
        LF_denoised.resize(asize);

    //! Transformation to YUV color space
    if (color_space_transform_LF(LF_noisy, LF_SAI_mask, color_space, width, height, chnls, true)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform_LF(LF_basic, LF_SAI_mask, color_space, width, height, chnls, true)
        != EXIT_SUCCESS) return EXIT_FAILURE;

    cout << endl << "Number of threads which will be used: " << nb_threads;
#ifdef _OPENMP
    cout << " (real available cores: " << omp_get_num_procs() << ")" << endl;
#else
    cout << endl;
#endif
    cout << endl;

    //! Allocate plan for FFTW library
    fftwf_plan* plan_2d_for_1 = new fftwf_plan[nb_threads];
  	fftwf_plan* plan_2d_for_2 = new fftwf_plan[nb_threads];
  	fftwf_plan* plan_2d_for_3 = new fftwf_plan[nb_threads];
  	fftwf_plan* plan_2d_inv = new fftwf_plan[nb_threads];
  	fftwf_plan* plan_4d = new fftwf_plan[nb_threads];
  	fftwf_plan* plan_4d_inv = new fftwf_plan[nb_threads];
  	fftwf_plan** plan_4d_sa = new fftwf_plan*[nb_threads]; //! Assign a 1d transform for each possible size between 2 and angular patch size
  	fftwf_plan** plan_4d_sa_inv = new fftwf_plan*[nb_threads];
  	for (unsigned int i = 0; i < nb_threads; i++) {
  		plan_4d_sa[i] = new fftwf_plan[asize_sw - 1];
  		plan_4d_sa_inv[i] = new fftwf_plan[asize_sw - 1];
  	}
  	fftwf_plan* plan_5d = new fftwf_plan[nb_threads];
  	fftwf_plan* plan_5d_inv = new fftwf_plan[nb_threads];

    //! Declaration of results buffer
    vector<vector<float> >  LF_denoised_num(asize, vector<float>(height * width * chnls, 0.0f)), //! Numerator (accumulation buffer of estimates)
                            LF_denoised_den(asize, vector<float>(height * width * chnls, 0.0f)); //! Denominator (aggregation weights)

    //! In the simple case
    if (nb_threads == 1)
    {
        //! Denoising, 1st Step
        const unsigned h_b = height + 2 * nWien;
        const unsigned w_b = width  + 2 * nWien;

        vector<unsigned> procSAI(asize); //! To keep track of the processed SAIs - 0 = unprocessed, >0 = processed
        for(unsigned st = 0; st < asize; st++) //! Do not process empty SAI
            procSAI[st] = !LF_SAI_mask[st];

        unsigned countProcSAI = count(procSAI.begin(), procSAI.end(), 0);
        const unsigned maxProcSAI = count(procSAI.begin(), procSAI.end(), 0);
        unsigned ps,pt,pst; //! angular indexes of the SAI to process

        while(countProcSAI) //! Loop on non-processed SAI
        {
            //! Determine the SAI to process
            if(countProcSAI == maxProcSAI && LF_SAI_mask[cst]) //! Start with the center SAI (it is assumed that the center SAI is not empty)
            {
                ps = cs;
                pt = ct;
            }
            else //! Process the SAI with the most pixels to denoised
            {
                int countNoisyPix = -1;
                #pragma omp parallel for
                for(int st = 0; st < asize; st++)
                {
                    if(procSAI[st] == 0)
                    {
                        int countNoisyPixst = count(LF_denoised_den[st].begin(), LF_denoised_den[st].end(), 0.0);
                        if(countNoisyPixst >= countNoisyPix)
                        {
                            pst = st;
                            countNoisyPix = countNoisyPixst;
                        }
                    }
                }
                if(ang_major == ROWMAJOR)
                {
                    ps = pst / awidth;
                    pt = pst - ps * awidth;
                }
                else if(ang_major == COLMAJOR)
                {
                    pt = pst / aheight;
                    ps = pst - pt * aheight;
                }
            }

            //! Compute angular search window around current SAI
            int cs_asw, min_s_asw, max_s_asw, ct_asw, min_t_asw, max_t_asw;
            compute_LF_angular_search_window(cs_asw, min_s_asw, max_s_asw, ps, aheight, anWien);
            compute_LF_angular_search_window(ct_asw, min_t_asw, max_t_asw, pt, awidth , anWien);

            unsigned cst_asw;
            if(ang_major==ROWMAJOR)
                cst_asw = cs_asw * asize_sw + ct_asw;
            else if(ang_major==COLMAJOR)
                cst_asw = cs_asw + ct_asw * asize_sw;

            vector<unsigned> st_idx_asw(asize_sw * asize_sw);
            #pragma omp parallel for
            for(int s_asw = 0; s_asw < asize_sw; s_asw++)
            {
                unsigned s = s_asw + min_s_asw;
                for(unsigned  t_asw = 0; t_asw < asize_sw; t_asw++)
                {
                    unsigned t = t_asw + min_t_asw;
                    unsigned st, st_asw;
                    if(ang_major==ROWMAJOR)
                    {
                        st     = s * awidth + t;
                        st_asw = s_asw * asize_sw + t_asw;
                    }
                    else if(ang_major==COLMAJOR)
                    {
                        st     = s + t * aheight;
                        st_asw = s_asw + t_asw * asize_sw;
                    }
                    st_idx_asw[st_asw] = st;
                }
            }

            cout << "-> Angular search window = (" << min_t_asw << ", " << min_s_asw << ") to (" << max_t_asw << ", " << max_s_asw << ")" << endl;

            //! Add boundaries and symetrize them
            //cout << "Symetrize LF." << endl;
            vector<unsigned> LF_SAI_mask_asw(asize_sw * asize_sw);
            vector<vector<float> > LF_sym_noisy_asw(asize_sw * asize_sw), LF_sym_basic_asw(asize_sw * asize_sw),
                                   LF_sym_denoised_num_asw(asize_sw * asize_sw), LF_sym_denoised_den_asw(asize_sw * asize_sw);
            #pragma omp parallel for
            for(int st_asw = 0; st_asw < st_idx_asw.size(); st_asw++)
            {
                unsigned st = st_idx_asw[st_asw];
                LF_SAI_mask_asw[st_asw] = LF_SAI_mask[st];
                if(LF_SAI_mask[st])
                {
                    symetrize(LF_noisy[st],        LF_sym_noisy_asw[st_asw],        width, height, chnls, nWien);
                    symetrize(LF_basic[st],        LF_sym_basic_asw[st_asw],        width, height, chnls, nWien);
                    symetrize(LF_denoised_num[st], LF_sym_denoised_num_asw[st_asw], width, height, chnls, nWien);
                    symetrize(LF_denoised_den[st], LF_sym_denoised_den_asw[st_asw], width, height, chnls, nWien);
                }
            }

            //! Initialization of SAI to process
            vector<unsigned> procSAI_asw(asize_sw * asize_sw); //! To keep track of the processed SAIs - 0 = unprocessed, >0 = processed
            for(unsigned st_asw = 0; st_asw < asize_sw * asize_sw; st_asw++) //! Do not process empty SAI
                procSAI_asw[st_asw] = !LF_SAI_mask_asw[st_asw];

            unsigned countProcSAI_asw = count(procSAI_asw.begin(), procSAI_asw.end(), 0);
            const unsigned maxProcSAI_asw = count(procSAI_asw.begin(), procSAI_asw.end(), 0);
            unsigned ps_asw, pt_asw, pst_asw;

            if(maxProcSAI_asw != asize_sw * asize_sw && tau_4D == DCT)
            {
                cout << "Warning ! Some SAI are empty, 4D transform changed to SADCT." << endl;
                tau_4D = SADCT;
            }

            //! Allocating Plan for FFTW process
            if (tau_2D == DCT)
            {
                const unsigned nb_cols = ind_size(w_b - kWien + 1, nWien, pWien);
                allocate_plan_2d(&plan_2d_for_1[0], kWien, FFTW_REDFT10,
                                                            w_b * (2 * nWien + 1) * chnls);
                allocate_plan_2d(&plan_2d_for_2[0], kWien, FFTW_REDFT10,
                                                            w_b * pWien * chnls);
                allocate_plan_2d(&plan_2d_for_3[0], kWien, FFTW_REDFT10,
                                                            (2 * nWien + 1) * (2 * nWien + 1) * chnls);
                allocate_plan_2d(&plan_2d_inv[0],   kWien, FFTW_REDFT01,
                                                            NWien * nb_cols * chnls);
            }
            if (tau_4D == DCT || tau_4D == SADCT)
            {
                allocate_plan_2d(&plan_4d[0],     asize_sw, asize_sw, FFTW_REDFT10, kWien * kWien * chnls);
                allocate_plan_2d(&plan_4d_inv[0], asize_sw, asize_sw, FFTW_REDFT01, kWien * kWien * chnls);
            }
            if(tau_4D == SADCT)
            {
                //! Allocate 1d DCT for each possible size between 2 and angular patch size
                for(unsigned k = 0; k < (asize_sw-1); k++)
                {
                    allocate_plan_1d(&plan_4d_sa[0][k],     k+2, FFTW_REDFT10, kWien * kWien * chnls);
                    allocate_plan_1d(&plan_4d_sa_inv[0][k], k+2, FFTW_REDFT01, kWien * kWien * chnls);
                }
            }

            //! Sub-loop on non-processed SAI in angular search window
            while(countProcSAI_asw)
            {
                if(countProcSAI_asw == maxProcSAI_asw && LF_SAI_mask_asw[cst_asw]) //! Start with central SAI of angular search window
                {
                    ps_asw  = cs_asw;
                    pt_asw  = ct_asw;
                    pst_asw = cst_asw;
                }
                else //! Process the SAI with the most pixels to denoised
                {
                    int countNoisyPix = -1;
                    #pragma omp parallel for
                    for(int st = 0; st < (asize_sw * asize_sw); st++)
                    {
                        if(procSAI_asw[st] == 0)
                        {
                            int countNoisyPixst = count(LF_sym_denoised_den_asw[st].begin(), LF_sym_denoised_den_asw[st].end(), 0.0);
                            if(countNoisyPixst >= countNoisyPix)
                            {
                                pst_asw = st;
                                countNoisyPix = countNoisyPixst;
                            }
                        }
                    }
                    if(ang_major == ROWMAJOR)
                    {
                        ps_asw = pst_asw / asize_sw;
                        pt_asw = pst_asw - ps_asw * asize_sw;
                    }
                    else if(ang_major == COLMAJOR)
                    {
                        pt_asw = pst_asw / asize_sw;
                        ps_asw = pst_asw - pt_asw * asize_sw;
                    }
                }

                //! Run BM5D first step
                cout << "\t-> Processing SAI (t, s) = (" << (min_t_asw + pt_asw) << ", " << (min_s_asw + ps_asw) << ") ... " << flush;
                float BM_elapsed_secs;
                timestamp_t start_step2_st = get_timestamp();
                bm5d_2nd_step(sigma, LF_sym_noisy_asw, LF_sym_basic_asw, LF_sym_denoised_num_asw, LF_sym_denoised_den_asw, LF_SAI_mask_asw, procSAI_asw, cst_asw, pst_asw,
                                asize_sw, asize_sw, w_b, h_b, chnls, nSim, nDisp,
                                kWien, NWien, pWien, useSD, color_space, tau_2D, tau_4D, tau_5D,
                                &plan_2d_for_1[0], &plan_2d_for_2[0], &plan_2d_for_3[0], &plan_2d_inv[0],
                                &plan_4d[0], &plan_4d_inv[0], plan_4d_sa[0], plan_4d_sa_inv[0], &plan_5d[0], &plan_5d_inv[0], BM_elapsed_secs);
                timestamp_t end_step2_st = get_timestamp();
                float step2_st_elapsed_secs = float(end_step2_st-start_step2_st) / 1000000.0f;
                cout << " done in " << step2_st_elapsed_secs << " secs. (BM time = " << BM_elapsed_secs << " secs)" << endl;

                //! Update non-processed SAIs count
                procSAI_asw[pst_asw] += 1;
                unsigned st;
                if(ang_major==ROWMAJOR)
                    st     = (min_s_asw + ps_asw) * awidth + (min_t_asw + pt_asw);
                else if(ang_major==COLMAJOR)
                    st     = (min_s_asw + ps_asw) + (min_t_asw + pt_asw) * aheight;
                procSAI[st] += 1;

                //! Break loop if all SAIs in search window are denoised
                float LF_dn_pct = LF_denoised_percent(LF_sym_denoised_den_asw, LF_SAI_mask_asw, width, height, chnls, nWien, kWien);
                if(LF_dn_pct >= 100.0f)
                {
                    for(unsigned st_asw = 0; st_asw < st_idx_asw.size(); st_asw++)
                    {
                        unsigned st = st_idx_asw[st_asw];
                        if(procSAI_asw[st_asw] == 0)
                        {
                             procSAI_asw[st_asw] += 1;
                             procSAI[st] += 1;
                        }
                    }
                }

                countProcSAI_asw = count(procSAI_asw.begin(), procSAI_asw.end(), 0);
            }

            //! Remove boundaries
            for(unsigned st_asw = 0; st_asw < st_idx_asw.size(); st_asw++)
            {
                unsigned st = st_idx_asw[st_asw];
                if(LF_SAI_mask[st])
                {
                    unsymetrize(LF_denoised_num[st], LF_sym_denoised_num_asw[st_asw], width, height, chnls, nWien);
                    unsymetrize(LF_denoised_den[st], LF_sym_denoised_den_asw[st_asw], width, height, chnls, nWien);
                }
            }

            countProcSAI = count(procSAI.begin(), procSAI.end(), 0);

            //! Display progress
            cout << countProcSAI << " SAI(s) remaining to process." << endl;
        }

        //! Get final basic estimate
        if(compute_LF_estimate(LF_SAI_mask, LF_denoised_num, LF_denoised_den, LF_basic, LF_denoised, asize) != EXIT_SUCCESS)
            return EXIT_FAILURE;
    }
    //* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //! If more than 1 threads are used
    //* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    else
    {
        vector<unsigned> procSAI(asize); //! To keep track of the processed SAIs - 0 = unprocessed, >0 = processed
        for(unsigned st = 0; st < asize; st++) //! Do not process empty SAI
            procSAI[st] = !LF_SAI_mask[st];

        unsigned countProcSAI = count(procSAI.begin(), procSAI.end(), 0);
        const unsigned maxProcSAI = count(procSAI.begin(), procSAI.end(), 0);
        unsigned ps,pt,pst; //! angular indexes of the SAI to process

        //! Sub-divide LF noisy in nb_threads
        vector<vector<vector<float> > > LF_sub_noisy(nb_threads), LF_sub_basic(nb_threads), LF_sub_denoised_num(nb_threads), LF_sub_denoised_den(nb_threads);
        vector<unsigned> h_table(nb_threads), w_table(nb_threads);
        #pragma omp parallel for
        for(int n = 0; n < nb_threads; n++)
        {
            LF_sub_noisy[n].resize(asize);
            LF_sub_basic[n].resize(asize);
            LF_sub_denoised_num[n].resize(asize);
            LF_sub_denoised_den[n].resize(asize);
        }

        sub_divide_LF(LF_noisy, LF_SAI_mask, LF_sub_noisy, w_table, h_table, awidth, aheight, width, height, chnls, nWien);
        sub_divide_LF(LF_basic, LF_SAI_mask, LF_sub_basic, w_table, h_table, awidth, aheight, width, height, chnls, nWien);

        while(countProcSAI) //! Loop on non-processed SAI
        {
            //! Determine the SAI to process
            if(countProcSAI == maxProcSAI && LF_SAI_mask[cst]) //! Start with the center SAI (it is assumed that the center SAI is not empty)
            {
                ps = cs;
                pt = ct;
            }
            else //! Process the SAI with the most pixels to denoised
            {
                int countNoisyPix = -1;
                #pragma omp parallel for
                for(int st = 0; st < asize; st++)
                {
                    if(procSAI[st] == 0)
                    {
                        int countNoisyPixst = count(LF_denoised_den[st].begin(), LF_denoised_den[st].end(), 0.0);
                        if(countNoisyPixst >= countNoisyPix)
                        {
                            pst = st;
                            countNoisyPix = countNoisyPixst;
                        }
                    }
                }
                if(ang_major == ROWMAJOR)
                {
                    ps = pst / awidth;
                    pt = pst - ps * awidth;
                }
                else if(ang_major == COLMAJOR)
                {
                    pt = pst / aheight;
                    ps = pst - pt * aheight;
                }
            }

            //! Compute angular search window around current SAI
            int cs_asw, min_s_asw, max_s_asw, ct_asw, min_t_asw, max_t_asw;
            compute_LF_angular_search_window(cs_asw, min_s_asw, max_s_asw, ps, aheight, anWien);
            compute_LF_angular_search_window(ct_asw, min_t_asw, max_t_asw, pt, awidth , anWien);

            unsigned cst_asw;
            if(ang_major==ROWMAJOR)
                cst_asw = cs_asw * asize_sw + ct_asw;
            else if(ang_major==COLMAJOR)
                cst_asw = cs_asw + ct_asw * asize_sw;

            vector<unsigned> st_idx_asw(asize_sw * asize_sw);
            #pragma omp parallel for
            for(int s_asw = 0; s_asw < asize_sw; s_asw++)
            {
                unsigned s = s_asw + min_s_asw;
                for(unsigned  t_asw = 0; t_asw < asize_sw; t_asw++)
                {
                    unsigned t = t_asw + min_t_asw;
                    unsigned st, st_asw;
                    if(ang_major==ROWMAJOR)
                    {
                        st     = s * awidth + t;
                        st_asw = s_asw * asize_sw + t_asw;
                    }
                    else if(ang_major==COLMAJOR)
                    {
                        st     = s + t * aheight;
                        st_asw = s_asw + t_asw * asize_sw;
                    }
                    st_idx_asw[st_asw] = st;
                }
            }

            cout << "-> Angular search window = (" << min_t_asw << ", " << min_s_asw << ") to (" << max_t_asw << ", " << max_s_asw << ")" << endl;

            //! Sub-divide current estimate
            sub_divide_LF(LF_denoised_num, LF_SAI_mask, LF_sub_denoised_num, w_table, h_table, awidth, aheight, width, height, chnls, nWien);
            sub_divide_LF(LF_denoised_den, LF_SAI_mask, LF_sub_denoised_den, w_table, h_table, awidth, aheight, width, height, chnls, nWien);

            //! Get subdivision for angular search window
            vector<vector<float> > LF_denoised_den_asw(asize_sw * asize_sw), LF_denoised_num_asw(asize_sw * asize_sw); //! Convenience to update subdivision in the loop
            vector<vector<vector<float> > > LF_sub_noisy_asw(nb_threads), LF_sub_basic_asw(nb_threads), LF_sub_denoised_num_asw(nb_threads), LF_sub_denoised_den_asw(nb_threads);
            vector<unsigned> LF_SAI_mask_asw(asize_sw * asize_sw);

            for(unsigned n = 0; n < nb_threads; n++)
            {
                LF_sub_noisy_asw[n].resize(asize_sw * asize_sw);
                LF_sub_basic_asw[n].resize(asize_sw * asize_sw);
                LF_sub_denoised_num_asw[n].resize(asize_sw * asize_sw);
                LF_sub_denoised_den_asw[n].resize(asize_sw * asize_sw);

                #pragma omp parallel for
                for(int st_asw = 0; st_asw < st_idx_asw.size(); st_asw++)
                {
                    unsigned st = st_idx_asw[st_asw];
                    LF_SAI_mask_asw[st_asw] = LF_SAI_mask[st];
                    if(LF_SAI_mask[st])
                    {
                        LF_sub_noisy_asw[n][st_asw] = LF_sub_noisy[n][st];
                        LF_sub_basic_asw[n][st_asw] = LF_sub_basic[n][st];
                        LF_sub_denoised_num_asw[n][st_asw] = LF_sub_denoised_num[n][st];
                        LF_sub_denoised_den_asw[n][st_asw] = LF_sub_denoised_den[n][st];
                    }
                }
            }

            //! Initialization of SAI to process
            vector<unsigned> procSAI_asw(asize_sw * asize_sw); //! To keep track of the processed SAIs - 0 = unprocessed, >0 = processed
            for(unsigned st_asw = 0; st_asw < asize_sw * asize_sw; st_asw++) //! Do not process empty SAI
                procSAI_asw[st_asw] = !LF_SAI_mask_asw[st_asw];

            unsigned countProcSAI_asw = count(procSAI_asw.begin(), procSAI_asw.end(), 0);
            const unsigned maxProcSAI_asw = count(procSAI_asw.begin(), procSAI_asw.end(), 0);
            unsigned ps_asw, pt_asw, pst_asw;

            if(maxProcSAI_asw != asize_sw * asize_sw && tau_4D == DCT)
            {
                cout << "Warning ! Some SAI are empty, 4D transform changed to SADCT." << endl;
                tau_4D = SADCT;
            }

            //! Allocating Plan for FFTW process
            if (tau_2D == DCT)
                for(unsigned n = 0; n < nb_threads; n++)
                {
                    const unsigned nb_cols = ind_size(w_table[n] - kWien + 1, nWien, pWien);
                    allocate_plan_2d(&plan_2d_for_1[n], kWien, FFTW_REDFT10,
                                                                w_table[n] * (2 * nWien + 1) * chnls);
                    allocate_plan_2d(&plan_2d_for_2[n], kWien, FFTW_REDFT10,
                                                                w_table[n] * pWien * chnls);
                    allocate_plan_2d(&plan_2d_for_3[n], kWien, FFTW_REDFT10,
                                                                (2 * nWien + 1) * (2 * nWien + 1) * chnls);
                    allocate_plan_2d(&plan_2d_inv[n],   kWien, FFTW_REDFT01,
                                                                NWien * nb_cols * chnls);
                }
            if (tau_4D == DCT || tau_4D == SADCT)
                for(unsigned n = 0; n < nb_threads; n++)
                {
                    allocate_plan_2d(&plan_4d[n],     asize_sw, asize_sw, FFTW_REDFT10, kWien * kWien * chnls);
                    allocate_plan_2d(&plan_4d_inv[n], asize_sw, asize_sw, FFTW_REDFT01, kWien * kWien * chnls);
                }
            if(tau_4D == SADCT)
            {
                //! Allocate 1d DCT for each possible size between 2 and angular patch size
                for(unsigned n = 0; n < nb_threads; n++)
                    for(unsigned k = 0; k < (asize_sw-1); k++)
                    {
                        allocate_plan_1d(&plan_4d_sa[n][k],     k+2, FFTW_REDFT10, kWien * kWien * chnls);
                        allocate_plan_1d(&plan_4d_sa_inv[n][k], k+2, FFTW_REDFT01, kWien * kWien * chnls);
                    }
            }

            //! Sub-loop on non-processed SAI in angular search window
            while(countProcSAI_asw)
            {
                if(countProcSAI_asw == maxProcSAI_asw && LF_SAI_mask_asw[cst_asw]) //! Start with central SAI of angular search window
                {
                    ps_asw  = cs_asw;
                    pt_asw  = ct_asw;
                    pst_asw = cst_asw;
                }
                else //! Process the SAI with the most pixels to denoised
                {
                    int countNoisyPix = -1;
                    #pragma omp parallel for
                    for(int st = 0; st < (asize_sw * asize_sw); st++)
                    {
                        if(procSAI_asw[st] == 0)
                        {
                            int countNoisyPixst = 0;
                            for(unsigned n = 0; n < nb_threads; n++)
                                countNoisyPixst += count(LF_sub_denoised_den_asw[n][st].begin(), LF_sub_denoised_den_asw[n][st].end(), 0.0);

                            if(countNoisyPixst >= countNoisyPix)
                            {
                                pst_asw = st;
                                countNoisyPix = countNoisyPixst;
                            }
                        }
                    }
                    if(ang_major == ROWMAJOR)
                    {
                        ps_asw = pst_asw / asize_sw;
                        pt_asw = pst_asw - ps_asw * asize_sw;
                    }
                    else if(ang_major == COLMAJOR)
                    {
                        pt_asw = pst_asw / asize_sw;
                        ps_asw = pst_asw - pt_asw * asize_sw;
                    }
                }

                //! Run BM5D first step
                cout << "\t-> Processing SAI (t, s) = (" << (min_t_asw + pt_asw) << ", " << (min_s_asw + ps_asw) << ") ... " << flush; //endl;
                vector<float> BM_elapsed_secs(nb_threads);
                timestamp_t start_step2_st = get_timestamp();
                #pragma omp parallel shared(LF_sub_noisy_asw, LF_sub_basic_asw, LF_sub_denoised_num_asw, LF_sub_denoised_den_asw, w_table, h_table, \
                                    plan_2d_for_1, plan_2d_for_2, plan_2d_for_3, plan_2d_inv, plan_4d, plan_4d_inv, plan_5d, plan_5d_inv, BM_elapsed_secs)
                {
                    #pragma omp for schedule(dynamic) nowait
                    for (int n = 0; n < nb_threads; n++)
                    {
                        bm5d_2nd_step(sigma, LF_sub_noisy_asw[n], LF_sub_basic_asw[n], LF_sub_denoised_num_asw[n], LF_sub_denoised_den_asw[n], LF_SAI_mask_asw, procSAI_asw, cst_asw, pst_asw,
                                        asize_sw, asize_sw, w_table[n], h_table[n], chnls, nSim, nDisp,
                                        kWien, NWien, pWien, useSD, color_space, tau_2D, tau_4D, tau_5D,
                                        &plan_2d_for_1[n], &plan_2d_for_2[n], &plan_2d_for_3[n], &plan_2d_inv[n],
                                        &plan_4d[n], &plan_4d_inv[n], plan_4d_sa[n], plan_4d_sa_inv[n], &plan_5d[n], &plan_5d_inv[n], BM_elapsed_secs[n]);
                    }
                }
                timestamp_t end_step2_st = get_timestamp();
                float step2_st_elapsed_secs = float(end_step2_st-start_step2_st) / 1000000.0f;
                float nb_BM = (float)(nb_threads - count(BM_elapsed_secs.begin(), BM_elapsed_secs.end(), 0.0f));
                float avg_BM_elapsed_secs = nb_BM > 0.0 ? accumulate(BM_elapsed_secs.begin(), BM_elapsed_secs.end(), 0.0f) / nb_BM : 0.0;
                cout << " done in " << step2_st_elapsed_secs << " secs. (BM average time = " << avg_BM_elapsed_secs << " secs)" << endl;

                //! Update subdivision
                undivide_LF(LF_sub_denoised_den_asw, LF_sub_denoised_num_asw, LF_denoised_den_asw, LF_denoised_num_asw, LF_SAI_mask_asw,
                            asize_sw, asize_sw, width, height, chnls, nWien);
                sub_divide_LF(LF_denoised_den_asw, LF_SAI_mask_asw, LF_sub_denoised_den_asw, w_table, h_table, asize_sw, asize_sw, width, height, chnls, nWien);
                sub_divide_LF(LF_denoised_num_asw, LF_SAI_mask_asw, LF_sub_denoised_num_asw, w_table, h_table, asize_sw, asize_sw, width, height, chnls, nWien);

                //! Update non-processed SAIs count
                procSAI_asw[pst_asw] += 1;
                unsigned st;
                if(ang_major==ROWMAJOR)
                    st     = (min_s_asw + ps_asw) * awidth + (min_t_asw + pt_asw);
                else if(ang_major==COLMAJOR)
                    st     = (min_s_asw + ps_asw) + (min_t_asw + pt_asw) * aheight;
                procSAI[st] += 1;

                //! Break loop if all SAIs in search window are denoised
                float LF_dn_pct = 0.0f;
                #pragma omp parallel for
                for (int n = 0; n < nb_threads; n++)
                    LF_dn_pct += LF_denoised_percent(LF_sub_denoised_den_asw[n], LF_SAI_mask_asw, w_table[n] - 2 * nWien, h_table[n] - 2 * nWien, chnls, nWien, kWien);
                if(LF_dn_pct >= (100.0f * (float)(nb_threads)))
                {
                    for(unsigned st_asw = 0; st_asw < st_idx_asw.size(); st_asw++)
                    {
                        unsigned st = st_idx_asw[st_asw];
                        if(procSAI_asw[st_asw] == 0)
                        {
                             procSAI_asw[st_asw] += 1;
                             procSAI[st] += 1;
                        }
                    }
                }

                countProcSAI_asw = count(procSAI_asw.begin(), procSAI_asw.end(), 0);
            }

            //! Update sub-division
            for(unsigned n = 0; n < nb_threads; n++)
            {
                #pragma omp parallel for
                for(int st_asw = 0; st_asw < st_idx_asw.size(); st_asw++)
                {
                    unsigned st = st_idx_asw[st_asw];
                    LF_SAI_mask_asw[st_asw] = LF_SAI_mask[st];
                    if(LF_SAI_mask[st])
                    {
                        LF_sub_denoised_num[n][st] = LF_sub_denoised_num_asw[n][st_asw];
                        LF_sub_denoised_den[n][st] = LF_sub_denoised_den_asw[n][st_asw];
                    }
                }
            }

            //! Un-divide current estimate
            undivide_LF(LF_sub_denoised_den, LF_sub_denoised_num, LF_denoised_den, LF_denoised_num, LF_SAI_mask, awidth, aheight, width, height, chnls, nWien);

            countProcSAI = count(procSAI.begin(), procSAI.end(), 0);

            //! Display progress
            cout << countProcSAI << " SAI(s) remaining to process." << endl;
        }

        //! Get final basic estimate
        if(compute_LF_estimate(LF_SAI_mask, LF_denoised_num, LF_denoised_den, LF_basic, LF_denoised, asize) != EXIT_SUCCESS)
            return EXIT_FAILURE;
    }

    //! Inverse color space transform to RGB
    if (color_space_transform_LF(LF_denoised, LF_SAI_mask, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform_LF(LF_basic, LF_SAI_mask, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform_LF(LF_noisy, LF_SAI_mask, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;

    //! Free Memory
    if (tau_2D == DCT)
        for (unsigned n = 0; n < nb_threads; n++)
        {
            fftwf_destroy_plan(plan_2d_for_1[n]);
            fftwf_destroy_plan(plan_2d_for_2[n]);
            fftwf_destroy_plan(plan_2d_for_3[n]);
            fftwf_destroy_plan(plan_2d_inv[n]);
        }
    if (tau_4D == DCT || tau_4D == SADCT)
        for (unsigned n = 0; n < nb_threads; n++)
        {
            fftwf_destroy_plan(plan_4d[n]);
            fftwf_destroy_plan(plan_4d_inv[n]);
        }
    if(tau_4D == SADCT)
        for (unsigned n = 0; n < nb_threads; n++)
            for(unsigned k = 0; k < (asize_sw-1); k++)
            {
                fftwf_destroy_plan(plan_4d_sa[n][k]);
                fftwf_destroy_plan(plan_4d_sa_inv[n][k]);
            }
    if(tau_5D == DCT)
        for (unsigned n = 0; n < nb_threads; n++)
        {
            fftwf_destroy_plan(plan_5d[n]);
            fftwf_destroy_plan(plan_5d_inv[n]);
        }
    fftwf_cleanup();

    return EXIT_SUCCESS;
}

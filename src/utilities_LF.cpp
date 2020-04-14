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
 * @file utilities_LF.cpp
 * @brief Utilities functions for light field
 *
 * @author Martin Alain <alainm@scss.tcd.ie>
 **/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>

#include "mt19937ar.h"
#include "io_png.h"
#include "utilities_LF.h"


#ifndef WINDOWS
#include <unistd.h>
//#include <dirent.h>
#include <sys/time.h>
#endif

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

 using namespace std;

 /**
  * @brief Load light field, check for empty SAIs
  *
  * @param name : path + name of the directory containing all sub-aperture images;
  * @param LF : 2D vector which will contain the light field : all sub-aperture images concatenated in the first vector, R, G and B concatenated in the second one;
  * @param LF_SAI_mask : indicate if sub-aperture image is empty (0) or not (1);
  * @param awidth, aheight, width, height, chnls : size of the light field.
  *
  * @return EXIT_SUCCESS if the light field has been loaded, EXIT_FAILURE otherwise
  **/
int load_LF(
    char* name
,   char* sub_img_name
,   char* sep
,   vector<vector<float> > &LF
,   vector<unsigned> &LF_SAI_mask
,   const unsigned ang_major
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned s_start
,   const unsigned t_start
,   unsigned * width
,   unsigned * height
,   unsigned * chnls
){
    // //! open directory
    // DIR *dp;
    // struct dirent *dirp;
    // if((dp = opendir(name)) == NULL)
    // {
    //     cout << "error :: " << name << " not found or not a correct directory" << endl;
    //     return EXIT_FAILURE;
    // }
    // closedir(dp);

    //! allocate memory for SAI mask
    LF_SAI_mask.assign(awidth*aheight, (unsigned) 0);

    //! loop on sub-aperture images
    cout << endl;
    #pragma omp parallel for
    for(int s = 0; s < aheight; s++)
    {
        stringstream strs;
        strs << setw(2) << setfill('0') << (s + s_start);
        for(unsigned t = 0; t < awidth; t++)
        {
            //! read input image
            stringstream strt;
            strt << setw(2) << setfill('0') << (t + t_start);
            string img_name = string(name) + "/" + string(sub_img_name) + string(sep) + strs.str() + string(sep) +  strt.str() + ".png";
            cout << "\rRead input image " << img_name << flush;
            size_t h, w, c;
            float *tmp = NULL;
            tmp = read_png_f32(img_name.c_str(), &w, &h, &c);
            if (!tmp)
            {
                cout << endl << "error :: " << img_name << " not found or not a correct png image." << string(name) << " folder might not exist." << endl;
            }

            //! test if image is really a color image and exclude the alpha channel
            if (c > 2)
            {
                unsigned k = 0;
                float acc = 0.0f;
                while (k < w * h && tmp[k] == tmp[w * h + k] && tmp[k] == tmp[2 * w * h + k])
                    acc += tmp[k] + tmp[w * h + k] + tmp[2 * w * h + k++];
                c = (k == w * h && acc > 0.0f ? 1 : 3);
            }

            //! Initializations (once)
            if(s==0 && t==0)
            {
                *width  = w;
                *height = h;
                *chnls  = c;
            }

            unsigned st;
            if(ang_major == ROWMAJOR)
                st = s * awidth + t;
            else if(ang_major==COLMAJOR)
                st = s + t * aheight;

            LF[st].resize(w * h * c); // angular dimension are allocated before calling this

            //! test if image is not empty
            for (unsigned k = 0; k < w * h * c; k++)
            {
                LF[st][k] = tmp[k];
                if(tmp[k] && LF_SAI_mask[st]==0)
                    LF_SAI_mask[st] = 1;
            }
            st++;
        }
    }
    //! Display LF informations
    cout << endl;
    cout << " Light field size :"  << endl;
    cout << " - awidth         = " << awidth  << endl;
    cout << " - aheight        = " << aheight << endl;
    cout << " - width          = " << *width  << endl;
    cout << " - height         = " << *height << endl;
    cout << " - nb of channels = " << *chnls  << endl;
    return EXIT_SUCCESS;
}


/**
 * @brief write light field
 *
 * @param LF_name : path+name of the directory where all the images are writen;
 * @param sub_img_name : generic name for the images (SAI or EPI);
 * @param LF : 2D vector which will contain the light field : all sub-aperture images concatenated in the first vector, R, G and B concatenated in the second one;
 * @param LF_SAI_mask : indicate if sub-aperture image is empty (0) or not (1);
 * @param ang_major : scanning order over angular dimensions (row or column major);
 * @param awidth, aheight, width, height, chnls : size of the light field.
 *
 * @return EXIT_SUCCESS if the image has been saved, EXIT_FAILURE otherwise
 **/
int save_LF(
    char* LF_name
,   char* sub_img_name
,   char* sep
,   vector<vector<float> > &LF
,   vector<unsigned> &LF_SAI_mask
,   const unsigned ang_major
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned s_start
,   const unsigned t_start
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
){
    // //! check directory
    // DIR *dp;
    // if((dp  = opendir(LF_name)) == NULL)
    // {
    //     cout << "error :: " << LF_name << " not found or not a correct directory" << endl;
    //     return EXIT_FAILURE;
    // }
    // closedir(dp);

    //! Loop over angular dimensions of LF
    #pragma omp parallel for
    for(int s = 0; s < aheight; s++)
    {
        for(unsigned t = 0; t < awidth; t++)
        {
            //! Compute angular index depending on angular scanning order
            unsigned st;
            if(ang_major==ROWMAJOR)
                st = s*awidth + t;
            else if(ang_major==COLMAJOR)
                st = s + t*aheight;

            string tmp_name(LF_name);
            ostringstream strs, strt;
            strs << setfill('0') << setw(2) << (s + s_start);
            strt << setfill('0') << setw(2) << (t + t_start);
            tmp_name = tmp_name + "/" + sub_img_name + string(sep) + strs.str() + string(sep) + strt.str() + ".png";
            cout << "\rWrite image " << tmp_name << flush;
            const char * img_name = tmp_name.c_str();
            save_image(img_name, LF[st], width, height, chnls);
        }
    }
    cout << endl;
    return EXIT_SUCCESS;
}


/**
 * @brief add noise to LF
 *
 * @param LF : original noise-free light field;
 * @param LF_SAI_mask : indicate if sub-aperture image is empty (0) or not (1);
 * @param LF_noisy = LF + noise;
 * @param sigma : standard deviation of the noise.
 *
 * @return none.
 **/
void add_noise_LF(
    const vector<vector<float> > &LF
,   vector<unsigned> &LF_SAI_mask
,   vector<vector<float> > &LF_noisy
,   const float sigma
){
    const unsigned asize = LF.size();
    if (LF_noisy.size() != asize)
        LF_noisy.resize(asize);

     //! Add noise to SAI
     #pragma omp parallel for
     for(int st = 0; st < asize; st++)
     {
        if(LF_SAI_mask[st])
            add_noise(LF[st], LF_noisy[st], sigma);
        else
            LF_noisy[st] = LF[st];
     }
}


/**
 * @brief Add boundaries by symetry
 *
 * @param LF : light field to symetrize;
 * @param LF_sym : will contain LF with symetrized boundaries;
 * @param LF_SAI_mask : indicate if sub-aperture image is empty (0) or not (1);
 * @param cs, ct : angular indexes of the central SAI;
 * @param ang_major : scanning order over angular dimensions (row or column major);
 * @param awidth, aheight, width, height, chnls : size of LF;
 * @param Nc : size of the boundary for the central SAI;
 * @param N  : size of the boundary for the non-central SAIs.
 *
 * @return none.
 **/
int symetrize_LF(
    vector<vector<float> > &LF
,   vector<vector<float> > &LF_sym
,   vector<unsigned> &LF_SAI_mask
,   const unsigned cs
,   const unsigned ct
,   const unsigned ang_major
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned Nc
,   const unsigned N
){
    const unsigned asize = awidth * aheight;
    if (LF_sym.size() != asize)
        LF_sym.resize(asize);

    //! Compute angular index of central SAI depending on angular scanning order
    unsigned cst;
    if(ang_major == ROWMAJOR)
        cst = cs * awidth + ct;
    else if(ang_major == COLMAJOR)
        cst = cs + ct * aheight;
    else
    {
        cout << "error :: wrong angular scanning order in save_LF function." << endl;
        return EXIT_FAILURE;
    }

    //! Symetrize each SAI
    #pragma omp parallel for
    for(int st = 0; st < asize; st++)
    {
        if(LF_SAI_mask[st])
        {
            const unsigned Nst = (st==cst ? Nc : N);
            symetrize(LF[st], LF_sym[st], width, height, chnls, Nst);
        }
    }
    return EXIT_SUCCESS;
}

/**
 * @brief Subdivide each SAI of a light field into small sub-images
 *
 * @param LF : light field to subdivide;
 * @param LF_SAI_mask : indicate if sub-aperture image is empty (0) or not (1);
 * @param sub_LF: will contain all sub_images;
 * @param w_table, h_table : size of sub-images contained in each SAI of sub_LF;
 * @param awidth, aheight, width, height, chnls : size of LF;
 * @param N  : size of the boundary;
 * @param divide: if true, sub-divides LF into sub_LF, else rebuild LF from sub_LF.
 *
 * @return none.
 **/
int sub_divide_LF(
    vector<vector<float> > &LF
,   vector<unsigned> &LF_SAI_mask
,   vector<vector<vector<float> > > &sub_LF
,   vector<unsigned> &w_table
,   vector<unsigned> &h_table
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned N
,   bool divide
){
    const unsigned asize = awidth * aheight;

    unsigned nb_threads = sub_LF.size();
    vector<vector<float> > sub_SAI(nb_threads);

    if(divide) //! Sub-divide each SAI
    {
        for(unsigned st = 0; st < asize; st++)
            if(LF_SAI_mask[st])
            {
                sub_divide(LF[st], sub_SAI, w_table, h_table, width, height, chnls, N, divide);

                //! Redistribute every sub SAI
                for(unsigned tr = 0; tr < nb_threads; tr++)
                    sub_LF[tr][st]  = sub_SAI[tr];
            }
    }
    else //! Reconstruction of each SAI
    {
        for(unsigned st = 0; st < asize; st++)
            if(LF_SAI_mask[st])
            {
                //! Redistribute every sub SAI
                for(unsigned tr = 0; tr < nb_threads; tr++)
                    sub_SAI[tr] = sub_LF[tr][st];

                sub_divide(LF[st], sub_SAI, w_table, h_table, width, height, chnls, N, divide);
            }
    }
    return EXIT_SUCCESS;
}

/**
 * @brief Subdivide each SAI of a light field into small sub-images
 *
 * @param LF : light field to subdivide;
 * @param LF_SAI_mask : indicate if sub-aperture image is empty (0) or not (1);
 * @param sub_LF: will contain all sub_images;
 * @param w_table, h_table : size of sub-images contained in each SAI of sub_LF;
 * @param awidth, aheight, width, height, chnls : size of LF;
 * @param N  : size of the boundary.
 *
 * @return none.
 **/
int sub_divide_LF(
    vector<vector<float> > &LF
,   vector<unsigned> &LF_SAI_mask
,   vector<vector<vector<float> > > &sub_LF
,   vector<unsigned> &w_table
,   vector<unsigned> &h_table
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned N
){
    const unsigned asize = awidth * aheight;

    unsigned nb_threads = sub_LF.size();
    vector<vector<float> > sub_SAI(nb_threads);

    for(unsigned st = 0; st < asize; st++)
        if(LF_SAI_mask[st])
        {
            sub_divide(LF[st], sub_SAI, w_table, h_table, width, height, chnls, N, true);

            //! Redistribute every sub SAI
            for(unsigned tr = 0; tr < nb_threads; tr++)
                sub_LF[tr][st]  = sub_SAI[tr];
        }

    return EXIT_SUCCESS;
}


/**
 * @brief Subdivide each SAI of a light field into small sub-images
 *
 * @param sub_LF_den, sub_LF_num: denominator and numerator respectively of subdivided light field;
 * @param LF_den, LF_num : will contain denominator and numerator respectively of light field rebuilt from subdivided light field;
 * @param LF_SAI_mask : indicate if sub-aperture image is empty (0) or not (1);
 * @param awidth, aheight, width, height, chnls : size of LF;
 * @param N  : size of the boundary.
 *
 * @return none.
 **/
int undivide_LF(
    const vector<vector<vector<float> > > &sub_LF_den
,   const vector<vector<vector<float> > > &sub_LF_num
,   vector<vector<float> > &LF_den
,   vector<vector<float> > &LF_num
,   vector<unsigned> &LF_SAI_mask
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned N
){
    const unsigned asize = awidth * aheight;

    for(unsigned st = 0; st < asize; st++)
        if(LF_SAI_mask[st])
        {
            //! Add by symetry boundaries to the img
            const unsigned h_b = height + 2 * N;
            const unsigned w_b = width  + 2 * N;

            //! Obtain nb of sub_images in row and column
            unsigned w_small = width;
            unsigned h_small = height;
            unsigned n = sub_LF_den.size();
            unsigned nw = 1;
            unsigned nh = 1;
            while (n > 1)
            {
                if (w_small > h_small)
                {
                    w_small = (unsigned) floor((float) w_small * 0.5f);
                    nw *= 2;
                }
                else
                {
                    h_small = (unsigned) floor((float) h_small * 0.5f);
                    nh *=2;
                }
                n /= 2;
            }

            //! As the image may not have power of 2 dimensions, it may exist a boundary
            const unsigned h_bound = (nh > 1 ? height - (nh - 1) * h_small : h_small);
            const unsigned w_bound = (nw > 1 ? width  - (nw - 1) * w_small : w_small);

            vector<float> SAI_den(width * height * chnls, 0.0f);
            vector<float> SAI_num(width * height * chnls, 0.0f);

            //! LF reconstruction
            for (unsigned i = 0; i < nh; i++)
                for (unsigned j = 0; j < nw; j++)
                {
                    const unsigned k = i * nw + j;
                    const unsigned h = (i == nh - 1 ? h_bound : h_small) + 2 * N;
                    const unsigned w = (j == nw - 1 ? w_bound : w_small) + 2 * N;
                    for (unsigned c = 0; c < chnls; c++)
                    {
                        unsigned dc = c * w * h + N * w + N;
                        unsigned dc_2 = c * width * height + i * h_small * width + j * w_small;
                        for (unsigned p = 0; p < h - 2 * N; p++)
                        {
                            unsigned dq = dc + p * w;
                            for (unsigned q = 0; q < w - 2 * N; q++, dq++)
                            {
                                SAI_den[dc_2 + p * width + q] += sub_LF_den[k][st][dq];
                                SAI_num[dc_2 + p * width + q] += sub_LF_num[k][st][dq];
                            }
                        }
                    }
                }
            LF_den[st] = SAI_den;
            LF_num[st] = SAI_num;
        }

    return EXIT_SUCCESS;
}

/**
 * @brief update subdivision of LF over overlapping boundaries
 *
 * @param sub_LF_den, sub_LF_num: will contain denominator and numerator respectively of subdivided light field;
 * @param LF_SAI_mask : indicate if sub-aperture image is empty (0) or not (1);
 * @param awidth, aheight, width, height, chnls : size of LF;
 * @param N  : size of the boundary.
 *
 * @return none.
 **/
int update_div_borders_LF(
    vector<vector<vector<float> > > &sub_LF_den
,   vector<vector<vector<float> > > &sub_LF_num
,   vector<unsigned> &LF_SAI_mask
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned N
){
    const unsigned asize = awidth * aheight;

    for(unsigned st = 0; st < asize; st++)
        if(LF_SAI_mask[st])
        {
            //! Add by symetry boundaries to the img
            const unsigned h_b = height + 2 * N;
            const unsigned w_b = width  + 2 * N;

            //! Obtain nb of sub_images in row and column
            unsigned w_small = width;
            unsigned h_small = height;
            unsigned n = sub_LF_den.size();
            unsigned nw = 1;
            unsigned nh = 1;
            while (n > 1)
            {
                if (w_small > h_small)
                {
                    w_small = (unsigned) floor((float) w_small * 0.5f);
                    nw *= 2;
                }
                else
                {
                    h_small = (unsigned) floor((float) h_small * 0.5f);
                    nh *=2;
                }
                n /= 2;
            }

            //! As the image may not have power of 2 dimensions, it may exist a boundary
            const unsigned h_bound = (nh > 1 ? height - (nh - 1) * h_small : h_small);
            const unsigned w_bound = (nw > 1 ? width  - (nw - 1) * w_small : w_small);

            vector<float> SAI_den(h_b * w_b * chnls, 0.0f);
            vector<float> SAI_num(h_b * w_b * chnls, 0.0f);

            //! Undivide reconstruction and update overlapping division
            for (unsigned i = 0; i < nh; i++)
                for (unsigned j = 0; j < nw; j++)
                {
                    const unsigned k = i * nw + j;
                    const unsigned h = (i == nh - 1 ? h_bound : h_small) + 2 * N;
                    const unsigned w = (j == nw - 1 ? w_bound : w_small) + 2 * N;
                    //! Undivide
                    for (unsigned c = 0; c < chnls; c++)
                    {
                        unsigned dc = c * w * h;
                        unsigned dc_2 = c * w_b * h_b + i * h_small * w_b + j * w_small;
                        for (unsigned p = 0; p < h; p++)
                        {
                            unsigned dq = dc + p * w;
                            for (unsigned q = 0; q < w; q++, dq++)
                            {
                                SAI_den[dc_2 + p * w_b + q] += sub_LF_den[k][st][dq];
                                SAI_num[dc_2 + p * w_b + q] += sub_LF_num[k][st][dq];
                            }
                        }
                    }
                }

            //! Divide update
            for (unsigned i = 0; i < nh; i++)
                for (unsigned j = 0; j < nw; j++)
                {
                    const unsigned k = i * nw + j;
                    const unsigned h = (i == nh - 1 ? h_bound : h_small) + 2 * N;
                    const unsigned w = (j == nw - 1 ? w_bound : w_small) + 2 * N;

                    for (unsigned c = 0; c < chnls; c++)
                    {
                        unsigned dc_2 = c * w_b * h_b + i * h_small * w_b + j * w_small;
                        for (unsigned p = 0; p < h; p++)
                        {
                            unsigned dq = c * w * h + p * w;
                            for (unsigned q = 0; q < w; q++, dq++)
                            {
                                sub_LF_den[k][st][dq] = SAI_den[dc_2 + p * w_b + q];
                                sub_LF_num[k][st][dq] = SAI_num[dc_2 + p * w_b + q];
                            }
                        }
                    }
                }
        }

    return EXIT_SUCCESS;
}


/**
 * @brief Compute PSNR and RMSE between each SAI of LF_1 and LF_2
 *
 * @param LF_1 : pointer to an allocated array of array of pixels;
 * @param LF_2 : pointer to an allocated array of array of pixels;
 * @param LF_SAI_mask : indicate if sub-aperture image is empty (0) or not (1);
 * @param psnr  : array that will contain the PSNR for each SAI;
 * @param rmse  : array that will contain the RMSE for each SAI;
 * @param avgpsnr, stdpnsr, avgrmse, stdrmse : contain the average and standard deviation of the psnr and rmse respectively.
 *
 * @return EXIT_FAILURE if both LF haven't the same size.
 **/
int compute_psnr_LF(
    const vector<vector<float> > &LF_1
,   const vector<vector<float> > &LF_2
,   vector<unsigned> &LF_SAI_mask
,   vector<float> &psnr
,   float *avg_psnr
,   float *std_psnr
,   vector<float> &rmse
,   float *avg_rmse
,   float *std_rmse
)
{
    if (LF_1.size() != LF_2.size())
    {
        cout << "Can't compute PSNR & RMSE, LF_1 and LF_2 don't have the same size" << endl;
        cout << "LF_1 : " << LF_1.size() << endl;
        cout << "LF_2 : " << LF_2.size() << endl;
        return EXIT_FAILURE;
    }

    const unsigned asize = LF_SAI_mask.size();

    if (psnr.size() != asize)
        psnr.resize(asize, 0.0f);
    if (rmse.size() != asize)
        rmse.resize(asize, 0.0f);

    //! Compute PSNR and RMSE for each SAI
    #pragma omp parallel for
    for(int st = 0; st < asize; st++)
        if(LF_SAI_mask[st])
            compute_psnr(LF_1[st], LF_2[st], &psnr[st], &rmse[st]);

    //! Compute mean of PSNR and RMSE
    float LF_size = (float) count(LF_SAI_mask.begin(), LF_SAI_mask.end(), 1);
    *avg_psnr = accumulate(psnr.begin(), psnr.end(), 0.0) / LF_size;
    *avg_rmse = accumulate(rmse.begin(), rmse.end(), 0.0) / LF_size;

    //! Compute standard deviation of PSNR and RMSE
    float tmp_stdpsnr = 0.0;
    float tmp_stdrmse = 0.0;

    #pragma omp parallel for
    for(int st = 0; st < asize; st++)
        if(LF_SAI_mask[st])
        {
            tmp_stdpsnr += (psnr[st]-*avg_psnr)*(psnr[st]-*avg_psnr);
            tmp_stdrmse += (rmse[st]-*avg_rmse)*(rmse[st]-*avg_rmse);
        }
    *std_psnr = sqrtf(tmp_stdpsnr/LF_size);
    *std_rmse = sqrtf(tmp_stdrmse/LF_size);

    return EXIT_SUCCESS;
}

/**
 * @brief Compute a difference for each SAI between LF_1 and LF_2
 *
 * @param LF_1 : pointer to an allocated array of array of pixels;
 * @param LF_2 : pointer to an allocated array of array of pixels;
 * @param LF_SAI_mask : indicate if sub-aperture image is empty (0) or not (1);
 * @param LF_diff  : will contain the difference between LF_1 and LF_2;
 * @param sigma : standard deviation of the noise.
 *
 * @return EXIT_FAILURE if both LF haven't the same size.
 **/
int compute_diff_LF(
    const vector<vector<float> > &LF_1
,   const vector<vector<float> > &LF_2
,   vector<unsigned> &LF_SAI_mask
,   vector<vector<float> > &LF_diff
,   const float sigma
){
    if (LF_1.size() != LF_2.size())
    {
        cout << "Can't compute difference, LF_1 and LF_2 don't have the same size" << endl;
        cout << "LF_1 : " << LF_1.size() << endl;
        cout << "LF_2 : " << LF_2.size() << endl;
        return EXIT_FAILURE;
    }

    const unsigned asize = LF_1.size();
    if (LF_diff.size() != asize)
        LF_diff.resize(asize);

    //! Compute difference for each SAI
    #pragma omp parallel for
    for(int st = 0; st < asize; st++)
        if(LF_SAI_mask[st])
            compute_diff(LF_1[st], LF_2[st], LF_diff[st], sigma);

    return EXIT_SUCCESS;
}

/**
 * @brief Transform the color space of each SAI of the light field
 *
 * @param LF: light field to transform;
 * @param LF_SAI_mask : indicate if sub-aperture image is empty (0) or not (1);
 * @param color_space: choice between OPP, YUV, YCbCr, RGB;
 * @param width, height, chnls: size of SAI;
 * @param rgb2yuv: if true, transform the color space
 *        from RGB to YUV, otherwise do the inverse.
 *
 * @return EXIT_FAILURE if color_space has not expected
 *         type, otherwise return EXIT_SUCCESS.
 **/
int color_space_transform_LF(
    vector<vector<float> > &LF
,   vector<unsigned> &LF_SAI_mask
,   const unsigned color_space
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const bool rgb2yuv
){
    if (chnls == 1 || color_space == RGB)
        return EXIT_SUCCESS;

    //#pragma omp parallel for
    for(unsigned st = 0; st < LF_SAI_mask.size(); st++)
    {
        if(LF_SAI_mask[st])
            if(color_space_transform(LF[st], color_space, width, height, chnls, rgb2yuv) != EXIT_SUCCESS)
                return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

/**
 * @brief Write PSNR and RMSE measures in file
 *
 * @param file_name : path+name of the file where the measures are writen;
 * @param LF_name : name of the light field;
 * @param LF_SAI_mask : indicate if sub-aperture image is empty (0) or not (1);
 * @param ang_major : scanning order over angular dimensions (row or column major);
 * @param awidth, aheight : angular size of the light field;
 * @param psnr  : array that will contain the PSNR for each SAI;
 * @param rmse  : array that will contain the RMSE for each SAI;
 * @param avg_psnr, std_psnr, avg_rmse, std_rmse : contain the average and standard deviation of the psnr and rmse respectively.
 *
 * @return EXIT_FAILURE if both LF haven't the same size.
 **/
int write_psnr_LF(
    const char *file_name
,   const char *LF_name
,   vector<unsigned> &LF_SAI_mask
,   const unsigned ang_major
,   const unsigned awidth
,   const unsigned aheight
,   const vector<float> &psnr
,   const float avg_psnr
,   const float std_psnr
,   const vector<float> &rmse
,   const float avg_rmse
,   const float std_rmse
)
{
    ofstream file(file_name, ios::out | ios::app);
    if(!file)
    {
        cout << "Can't open " << file_name << endl;
        return EXIT_FAILURE;
    }
    file << endl << "******************************************" << endl;
    file << "-> Average PSNR " << LF_name << " = " << avg_psnr << endl;
    file << "-> Standard deviation PSNR " << LF_name << " = " << std_psnr << endl;
    file << "PSNR for all " << LF_name << " SAIs:" << endl;

    //! Write PSNR values for each SAI
    for(unsigned s = 0; s < aheight; s++)
    {
        for(unsigned t = 0; t < awidth; t++)
        {
            //! Compute angular index depending on angular scanning order
            unsigned st;
            if(ang_major==ROWMAJOR)
            {
                st = s*awidth + t;
            }
            else if(ang_major==COLMAJOR)
            {
                st = s + t*aheight;
            }
            else
            {
                cout << "error :: wrong angular scanning order in write_psnr_LF function." << endl;
                return EXIT_FAILURE;
            }
            if(LF_SAI_mask[st])
                file << psnr[st] << " ";
            else
                file << "No SAI ";
        }
        file << endl;
    }
    file << endl;

    file << "-> Average RMSE " << LF_name << " = " << avg_rmse << endl;
    file << "-> Standard deviation RMSE " << LF_name << " = " << std_rmse << endl;
    file << "RMSE for all " << LF_name << " SAIs:" << endl;

    //! Write PSNR values for each SAI
    for(unsigned s = 0; s < aheight; s++)
    {
        for(unsigned t = 0; t < awidth; t++)
        {
            //! Compute angular index depending on angular scanning order
            unsigned st;
            if(ang_major==ROWMAJOR)
            {
                st = s*awidth + t;
            }
            else if(ang_major==COLMAJOR)
            {
                st = s + t*aheight;
            }
            else
            {
                cout << "error :: wrong angular scanning order in write_psnr_LF function." << endl;
                return EXIT_FAILURE;
            }

            file << rmse[st] << " ";
        }
        file << endl;
    }
    file << "******************************************" << endl;
    file.close();
    return EXIT_SUCCESS;
}

/**
 * @brief Compute angular search window min, max, and central SAI indexes
 *
 * @param c_asw: central SAI index of angular search window;
 * @param min_asw: smallest index of angular seach window;
 * @param max_asw: biggest index of angular seach window;
 * @param aidx: angular index around which the search window is computed;
 * @param asize: angular size of the light field;
 * @param asize_sw: angular size of the seach window.
 **/
void compute_LF_angular_search_window(
    int &c_asw
,   int &min_asw
,   int &max_asw
,   unsigned aidx
,   const unsigned asize
,   const unsigned asize_sw
)
{
    min_asw = aidx - asize_sw;
    max_asw = aidx + asize_sw;
    int s_shift = min_asw < 0 ? -min_asw : 0;
    min_asw += s_shift;
    max_asw += s_shift;
    c_asw = asize_sw - s_shift;

    s_shift = max_asw >= asize ? (asize-max_asw-1) : 0;
    min_asw += s_shift;
    max_asw += s_shift;
    c_asw -= s_shift;
}

/**
 * @brief Compute light field denoised estimate from numerator (accumulation buffer of estimates) and denominator (aggregation weights).
          If denominator is zero, replace with value in subsitute.
 *
 * @param LF_SAI_mask : indicate if sub-aperture image is empty (0) or not (1);
 * @param LF_num: LF numerator;
 * @param LF_den: LF denominator;
 * @param LF_sub: LF subsitute;
 * @param LF_est: will contain LF estimate.
 **/
int compute_LF_estimate(
    vector<unsigned> &LF_SAI_mask
,   vector<vector<float> > &LF_num
,   vector<vector<float> > &LF_den
,   vector<vector<float> > &LF_sub
,   vector<vector<float> > &LF_est
,   const unsigned asize
)
{
    if(LF_num.size() != LF_den.size() || LF_num.size() != LF_sub.size())
    {
        cout << "LF numerator and denominator should have the same size." << endl;
        return EXIT_FAILURE;
    }

    if(LF_est.size() != asize)
        LF_est.resize(asize);

    for(unsigned st = 0; st < asize; st++)
    {
        if(LF_SAI_mask[st])
        {
            if(LF_num[st].size() != LF_den[st].size() || LF_num[st].size() != LF_sub[st].size())
            {
                cout << "SAI numerator, denominator, and substitute should have the same size." << endl;
                return EXIT_FAILURE;
            }
            const unsigned size = LF_num[st].size();
            if(LF_est[st].size() != size)
                LF_est[st].resize(size);

            for(unsigned ij = 0; ij < size; ij++)
            {
                if(LF_den[st][ij])
                    LF_est[st][ij] = LF_num[st][ij] / LF_den[st][ij];
                else
                    LF_est[st][ij] = LF_sub[st][ij];
            }
        }
    }
    return EXIT_SUCCESS;
}

/**
 * @brief Compute percentage of light field which is denoised.
 *
 * @param LF_basic_den : image denominator of denoised light field;
 * @param LF_SAI_mask : indicate if sub-aperture image is empty (0) or not (1);
 * @param width, height: size of SAI;
 * @param N : boundary;
 * @param kHW : patch size.
 *
 * @return none.
 **/
float LF_denoised_percent(
    vector<vector<float> > LF_basic_den
,   vector<unsigned> &LF_SAI_mask
,   const unsigned width
,   const unsigned height
,	const unsigned chnls
,   const unsigned N
,   const unsigned kHW
)
{
    const unsigned w_b = width  + 2 * N;
    const unsigned h_b = height + 2 * N;

    float LF_dn_count = 0.0;
    const unsigned dc_b = N * w_b + N;
    const unsigned dc_c = w_b * h_b;
    for(unsigned st = 0; st < LF_basic_den.size(); st++)
        if(LF_SAI_mask[st])
            for (unsigned i = 0; i < (height - kHW + 1); i++)
                for (unsigned j = 0; j < (width - kHW + 1) ; j++)
					for (unsigned c = 0; c < chnls; c++)
						if (LF_basic_den[st][dc_b + i * w_b + j + c * dc_c] > 0.0)
						{
							LF_dn_count++;
							continue;
						}

    return LF_dn_count * 100.0f / (float) count(LF_SAI_mask.begin(), LF_SAI_mask.end(), 1) / (float) (height - kHW + 1) / (float) (width - kHW + 1);
}

/**
 * @brief Check if 2D patch is denoised (Utility for the following ind_initialize function).
 **/
bool is_patch_denoised(
    vector<float> SAI_weights
,   const unsigned p_idx
,   const unsigned width
,   const unsigned kHW
)
{
    bool denoised = true;
    for(unsigned p = 0; p < kHW; p++)
        for(unsigned q = 0; q < kHW; q++)
            if(SAI_weights[p_idx + p * width + q] == 0.0)
            {
                denoised = false;
                break;
            }
    return denoised;
}

/**
 * @brief Initialize a set of indices.
 *
 * @param row_ind, col_ind: will contain the set of indices for rows and columns;
 * @param max_height, max_width: indices can't go over this size;
 * @param width : image width;
 * @param N : boundary;
 * @param step: step between two indices;
 * @param kHW : patch size;
 * @param SAI_weights: indicate pixels already denoised (>0) or not (=0).
 *
 * @return none.
 **/
void ind_initialize(
    vector<unsigned> &row_ind
,   vector<vector<unsigned> > &col_ind
,   const unsigned max_height
,   const unsigned max_width
,   const unsigned width
,   const unsigned N
,   const unsigned step
,   const unsigned kHW
,   vector<float> SAI_weights
){
    row_ind.clear();
    for(unsigned j = 0; j < col_ind.size(); j++)
        col_ind[j].clear();
    col_ind.clear();
    vector<unsigned> tmp_col_ind;

    for (unsigned ind_i = N; ind_i < max_height - N; ind_i += step)
    {
        tmp_col_ind.clear();

        for (unsigned ind_j = N; ind_j < max_width - N; ind_j += step)
        {
            const unsigned p_idx = ind_i * width + ind_j;
            if(!is_patch_denoised(SAI_weights, p_idx, width, kHW))
                tmp_col_ind.push_back(ind_j);

        }

        const bool check_col_border = tmp_col_ind.size() == 0 ? true : (tmp_col_ind.back() < max_width - N - 1 ? true : false);
        if(check_col_border)
        {
            const unsigned p_idx = ind_i * width + max_width - N - 1;
            if(!is_patch_denoised(SAI_weights, p_idx, width, kHW))
                tmp_col_ind.push_back(max_width - N - 1);
        }
        if(tmp_col_ind.size() > 0)
        {
            row_ind.push_back(ind_i);
            col_ind.push_back(tmp_col_ind);
        }
    }

    const bool check_row_border = row_ind.size() == 0 ? true : (row_ind.back() < max_height - N - 1 ? true : false);
    if(check_row_border)
    {
        tmp_col_ind.clear();

        for (unsigned ind_j = N; ind_j < max_width - N; ind_j += step)
        {
            const unsigned p_idx = (max_height - N - 1) * width + ind_j;
            if(!is_patch_denoised(SAI_weights, p_idx, width, kHW))
                tmp_col_ind.push_back(ind_j);
        }

        const bool check_col_border = tmp_col_ind.size() == 0 ? true : (tmp_col_ind.back() < max_width - N - 1 ? true : false);
        if (check_col_border)
        {
            const unsigned p_idx = (max_height - N - 1) * width + max_width - N - 1;
            if(!is_patch_denoised(SAI_weights, p_idx, width, kHW))
                tmp_col_ind.push_back(max_width - N - 1);
        }
        if(tmp_col_ind.size() > 0)
        {
            row_ind.push_back(max_height - N - 1);
            col_ind.push_back(tmp_col_ind);
        }
    }
}


/**
 * @brief For convenience: Initialize parameters from command line for LFBM5D.
 *
 * @param LF_input_name, sub_img_name: input light field directory and sub-aperture image name;
 * @param gt_exists: flag indicating if ground truth light field exists for objective measures computation;
 * @param awidth, aheight, s_start, t_start, ang_major: light field angular size, angular index start, and ordering;
 * @param anHard, anWien: half size of the angular search window for hard thresholding and Wiener step respectively;
 * @param fSigma: assumed noise level;
 * @param lambdaHard5D: hard thresholding threshold;
 * @param LF_noisy_name, LF_basic_name, LF_denoised_name, LF_diff_name: name of the directories containing the results;
 * @param NHard, NWien: number of similar patches for hard thresholding and Wiener step respectively;
 * @param nSimHard, nDispHard, nSimWien, nDispWien: Half size of the search window for self-similarities and disparity respectively for HT and Wiener step respectively;
 * @param kHard, kWien: Patches size;
 * @param pHard, pWien: Processing step on row and columns;
 * @param tau_2D_hard, tau_4D_hard, tau_5D_hard, tau_2D_wien, tau_4D_wien, tau_5D_wien: names of the transforms;
 * @param useSD_1 (resp. useSD_2): if true, use weight based
 *        on the standard variation of the 5D group for the
 *        first (resp. second) step, otherwise use the number
 *        of non-zero coefficients after Hard Thresholding
 *        (resp. the norm of Wiener coefficients);
 * @param color_space: Transformation from RGB to YUV. Allowed values are RGB (do nothing), YUV, YCBCR and OPP;
 * @param nb_threads: number of threads for OpenMP parallel processing;
 * @param psnr_file_name: name of the file which will contain the psnr measures.
 *
 * @return none.
 **/
 int get_params(
    int argc
,   char **argv
,   char **LF_input_name
,   char **sub_img_name
,   char **sep
,   bool     &gt_exists
,   unsigned &awidth
,   unsigned &aheight
,   unsigned &s_start
,   unsigned &t_start
,   unsigned &anHard
,   unsigned &anWien
,   unsigned &ang_major
,   float &fSigma
,   float &lambdaHard5D
,   char **LF_noisy_name
,   char **LF_basic_name
,   char **LF_denoised_name
,   char **LF_diff_name
,   unsigned &NHard
,   unsigned &nSimHard
,   unsigned &nDispHard
,   unsigned &kHard
,   unsigned &pHard
,   unsigned &tau_2D_hard
,   unsigned &tau_4D_hard
,   unsigned &tau_5D_hard
,   bool     &useSD_1
,   unsigned &NWien
,   unsigned &nSimWien
,   unsigned &nDispWien
,   unsigned &kWien
,   unsigned &pWien
,   unsigned &tau_2D_wien
,   unsigned &tau_4D_wien
,   unsigned &tau_5D_wien
,   bool     &useSD_2
,   unsigned &color_space
,   unsigned &nb_threads
,   char **psnr_file_name
){
     //! Check if there is the right call for the algorithm
	if (argc < 38)
	{
		cout << "usage: LFBM5Ddenoising LF_dir SAI_name SAI_name_sep LF_awidth LF_aheight s_idx_start t_idx_start asw_size_ht asw_size_wien ang_major sigma lambda \
                 LF_dir_noisy LF_dir_basic LF_dir_denoised LF_dir_difference \
                 NHard nSimHard nDispHard khard pHard tau_2d_hard tau_4d_hard tau_5d_hard useSD_hard \
                 NWien nSimWien nDispWien kWien pWien tau_2d_wien tau_4d_wien tau_5d_wien useSD_wien \
                 color_space nb_threads resultsFile" << endl;
		return EXIT_FAILURE;
	}

    unsigned par_idx = 0;
    *LF_input_name = argv[++par_idx];
    *sub_img_name = argv[++par_idx];
    if(strcmp(*sub_img_name, "none" ) == 0) *sub_img_name = "";
    *sep = argv[++par_idx];
    if(strcmp(*sep, "none" ) == 0) *sep = "";
    gt_exists    = !(strcmp(*LF_input_name, "none" ) == 0);
    awidth       = atoi(argv[++par_idx]);
    aheight      = atoi(argv[++par_idx]);
    s_start      = atoi(argv[++par_idx]);
    t_start      = atoi(argv[++par_idx]);
    anHard       = atoi(argv[++par_idx]);
    anWien       = atoi(argv[++par_idx]);
    ang_major    = (strcmp(argv[++par_idx], "row") == 0 ? ROWMAJOR :
                   (strcmp(argv[  par_idx], "col") == 0 ? COLMAJOR : NONE));
    if (ang_major == NONE)
    {
        cout << "ang_major is not known. Choice is :" << endl;
        cout << " -row" << endl;
        cout << " -col" << endl;
        return EXIT_FAILURE;
    }
	fSigma       = atof(argv[++par_idx]);
    lambdaHard5D = atof(argv[++par_idx]);
    *LF_noisy_name     = argv[++par_idx];
    *LF_basic_name     = argv[++par_idx];
    *LF_denoised_name  = argv[++par_idx];
    *LF_diff_name      = argv[++par_idx];

    //! Hard thresholding step parameters
    NHard        = atof(argv[++par_idx]);
    nSimHard     = atof(argv[++par_idx]);
    nDispHard    = atof(argv[++par_idx]);
    kHard        = atof(argv[++par_idx]);
    pHard        = atof(argv[++par_idx]);
    tau_2D_hard  =  (strcmp(argv[++par_idx], "id"  ) == 0 ? ID :
                    (strcmp(argv[  par_idx], "dct" ) == 0 ? DCT :
                    (strcmp(argv[  par_idx], "bior") == 0 ? BIOR : NONE)));
    if (tau_2D_hard == NONE)
    {
        cout << "tau_2d_hard is not known. Choice is :" << endl;
        cout << " -id" << endl;
        cout << " -dct" << endl;
        cout << " -bior" << endl;
        return EXIT_FAILURE;
    }
    tau_4D_hard  = (strcmp(argv[++par_idx], "id"  ) == 0 ? ID :
                   (strcmp(argv[  par_idx], "dct" ) == 0 ? DCT :
                   (strcmp(argv[  par_idx], "sadct" ) == 0 ? SADCT : NONE)));
    if (tau_4D_hard == NONE)
    {
        cout << "tau_4d_hard is not known. Choice is :" << endl;
        cout << " -id" << endl;
        cout << " -dct" << endl;
        cout << " -sadct" << endl;
        return EXIT_FAILURE;
    }
    tau_5D_hard  = (strcmp(argv[++par_idx], "hw"  )  == 0 ? HADAMARD :
                   (strcmp(argv[  par_idx], "haar" ) == 0 ? HAAR :
                   (strcmp(argv[  par_idx], "dct" ) == 0 ? DCT : NONE)));
    if (tau_5D_hard == NONE)
    {
        cout << "tau_5d_hard is not known. Choice is :" << endl;
        cout << " -hw" << endl;
        cout << " -haar" << endl;
        cout << " -dct" << endl;
        return EXIT_FAILURE;
    }
    useSD_1      = (bool) atof(argv[++par_idx]);

    //! Wiener filter step parameters
    NWien        = atof(argv[++par_idx]);
    nSimWien     = atof(argv[++par_idx]);
    nDispWien    = atof(argv[++par_idx]);
    kWien        = atof(argv[++par_idx]);
    pWien        = atof(argv[++par_idx]);
    tau_2D_wien  =  (strcmp(argv[++par_idx], "id"  ) == 0 ? ID :
                    (strcmp(argv[  par_idx], "dct" ) == 0 ? DCT :
                    (strcmp(argv[  par_idx], "bior") == 0 ? BIOR : NONE)));
    if (tau_2D_wien == NONE)
    {
        cout << "tau_2d_wien is not known. Choice is :" << endl;
        cout << " -id" << endl;
        cout << " -dct" << endl;
        cout << " -bior" << endl;
        return EXIT_FAILURE;
    };
	tau_4D_wien  =  (strcmp(argv[++par_idx], "id"  ) == 0 ? ID :
                    (strcmp(argv[  par_idx], "dct" ) == 0 ? DCT :
                    (strcmp(argv[  par_idx], "sadct" ) == 0 ? SADCT : NONE)));
    if (tau_4D_wien == NONE)
    {
        cout << "tau_4d_wien is not known. Choice is :" << endl;
        cout << " -id" << endl;
        cout << " -dct" << endl;
        cout << " -sadct" << endl;
        return EXIT_FAILURE;
    };
	tau_5D_wien  =  (strcmp(argv[++par_idx], "hw"  )  == 0 ? HADAMARD :
                    (strcmp(argv[  par_idx], "haar" ) == 0 ? HAAR :
                    (strcmp(argv[  par_idx], "dct" ) == 0 ? DCT : NONE)));
    if (tau_5D_wien == NONE)
    {
        cout << "tau_5d_hard is not known. Choice is :" << endl;
        cout << " -hw" << endl;
        cout << " -haar" << endl;
        cout << " -dct" << endl;
        return EXIT_FAILURE;
    }
    useSD_2 = (bool) atof(argv[++par_idx]);

    //! Color space
	color_space  = (strcmp(argv[++par_idx], "rgb"  ) == 0 ? RGB   :
                                  (strcmp(argv[  par_idx], "yuv"  ) == 0 ? YUV   :
                                  (strcmp(argv[  par_idx], "ycbcr") == 0 ? YCBCR :
                                  (strcmp(argv[  par_idx], "opp"  ) == 0 ? OPP   : NONE))));
    if (color_space == NONE)
    {
        cout << "color_space is not known. Choice is :" << endl;
        cout << " -rgb" << endl;
        cout << " -yuv" << endl;
        cout << " -opp" << endl;
        cout << " -ycbcr" << endl;
        return EXIT_FAILURE;
    };

    nb_threads = atof(argv[++par_idx]);

    *psnr_file_name = argv[++par_idx];

    return EXIT_SUCCESS;
 }


 /**
 * @brief For convenience: Initialize parameters from command line for LFBM3D.
 *
 * @param LF_input_name, sub_img_name: input light field directory and sub-aperture image name;
 * @param gt_exists: flag indicating if ground truth light field exists for objective measures computation;
 * @param awidth, aheight, ang_major: light field angular size and ordering;
 * @param anHard, anWien: half size of the angular search window for hard thresholding and Wiener step respectively;
 * @param fSigma: assumed noise level;
 * @param lambdaHard3D: hard thresholding threshold;
 * @param LF_noisy_name, LF_basic_name, LF_denoised_name, LF_diff_name: name of the directories containing the results;
 * @param NHard, NWien: number of similar patches for hard thresholding and Wiener step respectively;
 * @param nHard, nWien: Half size of the search window for self-similarities;
 * @param kHard, kWien: Patches size;
 * @param pHard, pWien: Processing step on row and columns;
 * @param tau_2D_hard, tau_2D_wien: names of the transforms;
 * @param useSD_1 (resp. useSD_2): if true, use weight based
 *        on the standard variation of the 3D group for the
 *        first (resp. second) step, otherwise use the number
 *        of non-zero coefficients after Hard Thresholding
 *        (resp. the norm of Wiener coefficients);
 * @param color_space: Transformation from RGB to YUV. Allowed
 *        values are RGB (do nothing), YUV, YCBCR and OPP;
 * @param nb_threads: number of threads for OpenMP parallel processing;
 * @param psnr_file_name: name of the file which will contain the psnr measures.
 *
 * @return none.
 **/
 int get_params_BM3D(
    int argc
,   char **argv
,   char **LF_input_name
,   char **sub_img_name
,   char **sep
,   bool     &gt_exists
,   unsigned &awidth
,   unsigned &aheight
,   unsigned &s_start
,   unsigned &t_start
,   unsigned &anHard
,   unsigned &anWien
,   unsigned &ang_major
,   float &fSigma
,   float &lambdaHard3D
,   char **LF_noisy_name
,   char **LF_basic_name
,   char **LF_denoised_name
,   char **LF_diff_name
,   unsigned &NHard
,   unsigned &nHard
,   unsigned &kHard
,   unsigned &pHard
,   unsigned &tau_2D_hard
,   bool     &useSD_1
,   unsigned &NWien
,   unsigned &nWien
,   unsigned &kWien
,   unsigned &pWien
,   unsigned &tau_2D_wien
,   bool     &useSD_2
,   unsigned &color_space
,   unsigned &nb_threads
,   char **psnr_file_name
){
     //! Check if there is the right call for the algorithm
	if (argc < 27)
	{
		cout << "usage: LFBM3Ddenoising LF_dir SAI_name SAI_name_sep LF_awidth LF_aheight s_idx_start t_idx_start asw_size_ht asw_size_wien ang_major sigma lambda \
                 LF_dir_noisy LF_dir_basic LF_dir_denoised LF_dir_difference \
                 NHard nHard kHard pHard tau_2d_hard \
                 NWien nWien kWien pWien tau_2d_wien \
                 color_space nb_threads resultsFile" << endl;
		return EXIT_FAILURE;
	}

    unsigned par_idx = 0;
    *LF_input_name = argv[++par_idx];
    *sub_img_name = argv[++par_idx];
    if(strcmp(*sub_img_name, "none" ) == 0) *sub_img_name = "";
    *sep = argv[++par_idx];
    if(strcmp(*sep, "none" ) == 0) *sep = "";
    gt_exists    = !(strcmp(*LF_input_name, "none" ) == 0);
    awidth       = atoi(argv[++par_idx]);
    aheight      = atoi(argv[++par_idx]);
    s_start      = atoi(argv[++par_idx]);
    t_start      = atoi(argv[++par_idx]);
    anHard       = atoi(argv[++par_idx]);
    anWien       = atoi(argv[++par_idx]);
    ang_major    = (strcmp(argv[++par_idx], "row") == 0 ? ROWMAJOR :
                   (strcmp(argv[  par_idx], "col") == 0 ? COLMAJOR : NONE));
    if (ang_major == NONE)
    {
        cout << "ang_major is not known. Choice is :" << endl;
        cout << " -row" << endl;
        cout << " -col" << endl;
        return EXIT_FAILURE;
    }
	fSigma       = atof(argv[++par_idx]);
    lambdaHard3D = atof(argv[++par_idx]);
    *LF_noisy_name     = argv[++par_idx];
    *LF_basic_name     = argv[++par_idx];
    *LF_denoised_name  = argv[++par_idx];
    *LF_diff_name      = argv[++par_idx];

    //! Hard thresholding step parameters
    NHard        = atof(argv[++par_idx]);
    nHard     = atof(argv[++par_idx]);
    kHard        = atof(argv[++par_idx]);
    pHard        = atof(argv[++par_idx]);
    tau_2D_hard  =  (strcmp(argv[++par_idx], "dct" ) == 0 ? DCT :
                    (strcmp(argv[  par_idx], "bior") == 0 ? BIOR : NONE));
    if (tau_2D_hard == NONE)
    {
        cout << "tau_2d_hard is not known. Choice is :" << endl;
        cout << " -dct" << endl;
        cout << " -bior" << endl;
        return EXIT_FAILURE;
    }
    useSD_1      = (bool) atof(argv[++par_idx]);

    //! Wiener filter step parameters
    NWien        = atof(argv[++par_idx]);
    nWien     = atof(argv[++par_idx]);
    kWien        = atof(argv[++par_idx]);
    pWien        = atof(argv[++par_idx]);
    tau_2D_wien  =  (strcmp(argv[++par_idx], "dct" ) == 0 ? DCT :
                    (strcmp(argv[  par_idx], "bior") == 0 ? BIOR : NONE));
    if (tau_2D_wien == NONE)
    {
        cout << "tau_2d_wien is not known. Choice is :" << endl;
        cout << " -dct" << endl;
        cout << " -bior" << endl;
        return EXIT_FAILURE;
    };
    useSD_2 = (bool) atof(argv[++par_idx]);

    //! Color space
	color_space  = (strcmp(argv[++par_idx], "rgb"  ) == 0 ? RGB   :
                                  (strcmp(argv[  par_idx], "yuv"  ) == 0 ? YUV   :
                                  (strcmp(argv[  par_idx], "ycbcr") == 0 ? YCBCR :
                                  (strcmp(argv[  par_idx], "opp"  ) == 0 ? OPP   : NONE))));
    if (color_space == NONE)
    {
        cout << "color_space is not known. Choice is :" << endl;
        cout << " -rgb" << endl;
        cout << " -yuv" << endl;
        cout << " -opp" << endl;
        cout << " -ycbcr" << endl;
        return EXIT_FAILURE;
    };

    nb_threads = atof(argv[++par_idx]);

    *psnr_file_name = argv[++par_idx];

    return EXIT_SUCCESS;
 }



 /**
 * @brief Get current time
 **/
 timestamp_t get_timestamp()
 {
#ifndef WINDOWS
     struct timeval now;
     gettimeofday(&now, NULL);
     return now.tv_usec + (timestamp_t) now.tv_sec * 1000000;
#else
	 clock_t time = clock();
	 double duration = time / (double)CLOCKS_PER_SEC;
	 return (long)(duration * 1000000) % 1000000 + duration * 1000000;
#endif
 }

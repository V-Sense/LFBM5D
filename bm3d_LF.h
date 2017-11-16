#ifndef BM3D_LF_H_INCLUDED
#define BM3D_LF_H_INCLUDED

#include "bm3d.h"

/** ------------------ **/
/** - Main functions - **/
/** ------------------ **/
//! Main function
int run_bm3d_LF(
    const float sigma
,   std::vector<std::vector<float> > &LF_noisy
,   std::vector<unsigned> &LF_SAI_mask
,   std::vector<std::vector<float> > &LF_basic
,   std::vector<std::vector<float> > &LF_denoised
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
);

#endif // BM3D_LF_H_INCLUDED

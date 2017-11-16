#ifndef BM5D_H_INCLUDED
#define BM5D_H_INCLUDED

#include "bm3d.h"
#include "bm5d_core_processing.h"

/** ------------------ **/
/** - Main functions - **/
/** ------------------ **/
//! Hard thresholding
int run_bm5d_1st_step(
    const float sigma
,   const float lambdaHard5D
,   std::vector<std::vector<float> > &LF_noisy
,   std::vector<unsigned>            &LF_SAI_mask
,   std::vector<std::vector<float> > &LF_basic
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
);

//! Wiener filtering
int run_bm5d_2nd_step(
    const float sigma
,   std::vector<std::vector<float> > &LF_noisy
,   std::vector<unsigned>            &LF_SAI_mask
,   std::vector<std::vector<float> > &LF_basic
,   std::vector<std::vector<float> > &LF_denoised
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
);


#endif // BM5D_H_INCLUDED

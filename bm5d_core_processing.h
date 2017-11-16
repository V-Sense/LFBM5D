#ifndef BM5D_CORE_H_INCLUDED
#define BM5D_CORE_H_INCLUDED

#include "bm3d.h"

void bm5d_1st_step(
    const float sigma
,   float lambdaHard5D
,   std::vector<std::vector<float> > &LF_noisy
,   std::vector<std::vector<float> > &LF_basic_num
,   std::vector<std::vector<float> > &LF_basic_den
,   std::vector<unsigned> &LF_SAI_mask
,   const std::vector<unsigned> &procSAI
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
);

void bm5d_2nd_step(
    const float sigma
,   std::vector<std::vector<float> > &LF_noisy
,   std::vector<std::vector<float> > &LF_basic
,   std::vector<std::vector<float> > &LF_denoised_num
,   std::vector<std::vector<float> > &LF_denoised_den
,   std::vector<unsigned> &LF_SAI_mask
,   const std::vector<unsigned> &procSAI
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
);

void id_2d_process(
    std::vector<float> &table_2D
,   std::vector<float> const& img
,   const unsigned nHW
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned kHW
,   const unsigned i_r
);

void id_2d_process(
    std::vector<float> &table_2D
,   std::vector<float> const& img
,   const unsigned nHW
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned kHW
,   const unsigned i_r
,   const unsigned j_r
);

void dct_2d_process(
    std::vector<float> &DCT_table_2D
,   std::vector<float> const& img
,   fftwf_plan * plan_1
,   fftwf_plan * plan_2
,   const unsigned nHW
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned kHW
,   const unsigned i_r
,   const unsigned j_r
,   std::vector<float> const& coef_norm
);

void bior_2d_process(
    std::vector<float> &bior_table_2D
,   std::vector<float> const& img
,   const unsigned nHW
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned kHW
,   const unsigned i_r
,   const unsigned j_r
,   std::vector<float> &lpd
,   std::vector<float> &hpd
);

void dct_4d_process(
    std::vector<float> &group_4D
,   fftwf_plan * plan
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   const unsigned kHW
,   std::vector<float> const& coef_norm
);

void dct_4d_inverse(
    std::vector<float> &group_4D
,   fftwf_plan * plan
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   const unsigned kHW
,   std::vector<float> const& coef_norm_inv
);

void sadct_4d_process(
    std::vector<float> &group_4D
,   const std::vector<unsigned> &shape_mask
,   const std::vector<unsigned> &shape_idx
,   std::vector<unsigned> &shape_mask_col
,   std::vector<unsigned> &shape_idx_col
,   std::vector<unsigned> &shape_mask_dct
,   fftwf_plan * plan
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   const unsigned kHW
,   const std::vector<std::vector<float> > &coef_norm
);

void sadct_4d_inverse(
    std::vector<float> &group_4D
,   const std::vector<unsigned> &shape_mask
,   const std::vector<unsigned> &shape_idx
,   const std::vector<unsigned> &shape_mask_col
,   const std::vector<unsigned> &shape_idx_col
,   fftwf_plan * plan
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   const unsigned kHW
,   const std::vector<std::vector<float> > &coef_norm_inv
);

void ht_filtering_hadamard_5d(
    std::vector<float> &group_5D
,   std::vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   std::vector<float> const& sigma_table
,   const float lambdaHard5D
,   std::vector<float> &weight_table
);

void ht_filtering_hadamard_5d(
    std::vector<float> &group_5D
,   std::vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   std::vector<float> const& sigma_table
,   const float lambdaHard5D
,   std::vector<float> &weight_table
,   std::vector<unsigned> shape_mask_dct
);

void ht_filtering_haar_5d(
    std::vector<float> &group_5D
,   std::vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   std::vector<float> const& sigma_table
,   const float lambdaHard5D
,   std::vector<float> &weight_table
);

void ht_filtering_haar_5d(
    std::vector<float> &group_5D
,   std::vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   std::vector<float> const& sigma_table
,   const float lambdaHard5D
,   std::vector<float> &weight_table
,   std::vector<unsigned> shape_mask_dct
);

void ht_filtering_dct_5d(
    std::vector<float> &group_5D
,   fftwf_plan * plan
,   fftwf_plan * plan_inv
,   std::vector<float> &coef_norm
,   std::vector<float> &coef_norm_inv
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   std::vector<float> const& sigma_table
,   const float lambdaHard5D
,   std::vector<float> &weight_table
);

void ht_filtering_dct_5d(
    std::vector<float> &group_5D
,   fftwf_plan * plan
,   fftwf_plan * plan_inv
,   std::vector<float> &coef_norm
,   std::vector<float> &coef_norm_inv
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   std::vector<float> const& sigma_table
,   const float lambdaHard5D
,   std::vector<float> &weight_table
,   std::vector<unsigned> shape_mask_dct
);

void wiener_filtering_hadamard_5d(
    std::vector<float> &group_5D_org
,   std::vector<float> &group_5D_est
,   std::vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   std::vector<float> const& sigma_table
,   std::vector<float> &weight_table
);

void wiener_filtering_hadamard_5d(
    std::vector<float> &group_5D_org
,   std::vector<float> &group_5D_est
,   std::vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   std::vector<float> const& sigma_table
,   std::vector<float> &weight_table
,   std::vector<unsigned> shape_mask_dct
);

void wiener_filtering_haar_5d(
    std::vector<float> &group_5D_org
,   std::vector<float> &group_5D_est
,   std::vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   std::vector<float> const& sigma_table
,   std::vector<float> &weight_table
);

void wiener_filtering_haar_5d(
    std::vector<float> &group_5D_org
,   std::vector<float> &group_5D_est
,   std::vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   std::vector<float> const& sigma_table
,   std::vector<float> &weight_table
,   std::vector<unsigned> shape_mask_dct
);

void wiener_filtering_dct_5d(
    std::vector<float> &group_5D_org
,   std::vector<float> &group_5D_est
,   fftwf_plan * plan
,   fftwf_plan * plan_inv
,   std::vector<float> &coef_norm
,   std::vector<float> &coef_norm_inv
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   std::vector<float> const& sigma_table
,   std::vector<float> &weight_table
);

void wiener_filtering_dct_5d(
    std::vector<float> &group_5D_org
,   std::vector<float> &group_5D_est
,   fftwf_plan * plan
,   fftwf_plan * plan_inv
,   std::vector<float> &coef_norm
,   std::vector<float> &coef_norm_inv
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   std::vector<float> const& sigma_table
,   std::vector<float> &weight_table
,   std::vector<unsigned> shape_mask_dct
);

void sd_weighting_5d(
    std::vector<std::vector<float> > const& group_5D
,   const unsigned kHW_2
,   const unsigned nSx_r
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned chnls
,   std::vector<float> &weight_table
);

/** ---------------------------------- **/
/** - Preprocessing / Postprocessing - **/
/** ---------------------------------- **/
void preProcess_4d(
    std::vector<float> &coef_norm
,   std::vector<float> &coef_norm_inv
,   const unsigned awidth
,   const unsigned aheight
);

void preProcess_4d_sadct(
    std::vector<std::vector<float> > &coef_norm
,   std::vector<std::vector<float> > &coef_norm_inv
,   const unsigned max_dct_size
);

void preProcess_5d(
    std::vector<float> &coef_norm
,   std::vector<float> &coef_norm_inv
,   const unsigned NHW
);

void precompute_BM(
    std::vector<std::vector<unsigned> > &patch_table
,   const std::vector<float> &img
,   const unsigned width
,   const unsigned height
,   const unsigned kHW
,   const unsigned NHW
,   const unsigned nHW
,   const unsigned nHW_sim
,   const unsigned pHW
,   const float    tauMatch
);

int precompute_BM_stereo(
    std::vector<std::vector<unsigned> > &patch_table
,   std::vector<unsigned> &shape_table
,   const std::vector<float> &img1
,   const std::vector<float> &img2
,   const unsigned width
,   const unsigned height
,   const unsigned kHW
,   const unsigned nHW
,   const unsigned nHW_disp
,   const unsigned pHW
,   const float    tauMatch
);

void precompute_BM(
    std::vector<std::vector<unsigned> > &patch_table
,   const std::vector<float> &img
,   const unsigned width
,   const unsigned height
,   const unsigned kHW
,   const unsigned NHW
,   const unsigned nHW
,   const unsigned nHW_sim
,   const float    tauMatch
,   const std::vector<unsigned> row_ind
,   const std::vector<std::vector<unsigned> > column_ind_per_row
);

int precompute_BM_stereo(
    std::vector<std::vector<unsigned> > &patch_table
,   std::vector<unsigned> &shape_table
,   const std::vector<float> &img1
,   const std::vector<float> &img2
,   const unsigned width
,   const unsigned height
,   const unsigned kHW
,   const unsigned nHW
,   const unsigned nHW_disp
,   const float    tauMatch
,   const std::vector<unsigned> row_ind
,   const std::vector<std::vector<unsigned> > column_ind_per_row
);


#endif // BM5D_CORE_H_INCLUDED

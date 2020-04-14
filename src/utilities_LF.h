#ifndef UTILITIES_LF_H_INCLUDED
#define UTILITIES_LF_H_INCLUDED

#include <vector>
#include <fftw3.h>

#include "utilities.h"

typedef unsigned long long timestamp_t;

//! Read light field and check for empty sub-aperture images
int load_LF(
    char* name
,   char* sub_img_name
,   char* sep
,   std::vector<std::vector<float> > &LF
,   std::vector<unsigned> &LF_SAI_mask
,   const unsigned ang_major
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned s_start
,   const unsigned t_start
,   unsigned * width
,   unsigned * height
,   unsigned * chnls
);

//! Write light field
int save_LF(
    char* name
,   char* sub_img_name
,   char* sep
,   std::vector<std::vector<float> > &LF
,   std::vector<unsigned> &LF_SAI_mask
,   const unsigned ang_major
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned s_start
,   const unsigned t_start
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
);

//! Add noise to each sub-aperture image
void add_noise_LF(
    const std::vector<std::vector<float> > &LF
,   std::vector<unsigned> &LF_SAI_mask
,   std::vector<std::vector<float> > &LF_noisy
,   const float sigma
);

//! Create EPI from LF
void create_EPI_LF(
    const std::vector<std::vector<float> > &LF
,   std::vector<std::vector<float> > &LF_EPI
,   const unsigned EPI_scan
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
);

void EPI_2_SAI(
    std::vector<std::vector<float> > &LF
,   const std::vector<std::vector<float> > &LF_EPI
,   const unsigned EPI_scan
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
);

//! Add boundaries by symetry
int symetrize_LF(
    std::vector<std::vector<float> > &LF
,   std::vector<std::vector<float> > &LF_sym
,   std::vector<unsigned> &LF_SAI_mask
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
);

//! Subdivide light field into small sub-light-field
int sub_divide_LF(
    std::vector<std::vector<float> > &LF
,   std::vector<unsigned> &LF_SAI_mask
,   std::vector<std::vector<std::vector<float> > > &sub_LF
,   std::vector<unsigned> &w_table
,   std::vector<unsigned> &h_table
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned N
,   bool divide
);

int sub_divide_LF(
    std::vector<std::vector<float> > &LF
,   std::vector<unsigned> &LF_SAI_mask
,   std::vector<std::vector<std::vector<float> > > &sub_LF
,   std::vector<unsigned> &w_table
,   std::vector<unsigned> &h_table
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned N
);

int undivide_LF(
    const std::vector<std::vector<std::vector<float> > > &sub_LF_den
,   const std::vector<std::vector<std::vector<float> > > &sub_LF_num
,   std::vector<std::vector<float> > &LF_den
,   std::vector<std::vector<float> > &LF_num
,   std::vector<unsigned> &LF_SAI_mask
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned N
);

int update_div_borders_LF(
    std::vector<std::vector<std::vector<float> > > &sub_LF_den
,   std::vector<std::vector<std::vector<float> > > &sub_LF_num
,   std::vector<unsigned> &LF_SAI_mask
,   const unsigned awidth
,   const unsigned aheight
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned N
);

//! Compute all PSNRs and RMSEs (including average and standard deviation) between all sub-aperture images of LF_1 and LF_2
int compute_psnr_LF(
    const std::vector<std::vector<float> > &LF_1
,   const std::vector<std::vector<float> > &LF_2
,   std::vector<unsigned> &LF_SAI_mask
,   std::vector<float> &psnr
,   float *avg_psnr
,   float *std_psnr
,   std::vector<float> &rmse
,   float *avg_rmse
,   float *std_rmse
);

//! Compute the difference between LF_1 and LF_2
int compute_diff_LF(
    const std::vector<std::vector<float> > &LF_1
,   const std::vector<std::vector<float> > &LF_2
,   std::vector<unsigned> &LF_SAI_mask
,   std::vector<std::vector<float> > &LF_diff
,   const float sigma
);

//! Transform the color space of each sub-aperture image of the light field
int color_space_transform_LF(
    std::vector<std::vector<float> > &LF
,   std::vector<unsigned> &LF_SAI_mask
,   const unsigned color_space
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const bool rgb2yuv
);

int write_psnr_LF(
    const char *file_name
,   const char *LF_name
,   std::vector<unsigned> &LF_SAI_mask
,   const unsigned ang_major
,   const unsigned awidth
,   const unsigned aheight
,   const std::vector<float> &psnr
,   const float avg_psnr
,   float std_psnr
,   const std::vector<float> &rmse
,   const float avg_rmse
,   const float std_rmse
);

void compute_LF_angular_search_window(
    int &cs_asw
,   int &min_asw
,   int &max_asw
,   unsigned aidx
,   const unsigned asize
,   const unsigned asize_sw
);

int compute_LF_estimate(
    std::vector<unsigned> &LF_SAI_mask
,   std::vector<std::vector<float> > &LF_num
,   std::vector<std::vector<float> > &LF_den
,   std::vector<std::vector<float> > &LF_sub
,   std::vector<std::vector<float> > &LF_est
,   const unsigned asize
);

bool is_patch_denoised(
    std::vector<float> SAI_weights
,   const unsigned p_idx
,   const unsigned width
,   const unsigned kHW
);

float LF_denoised_percent(
    std::vector<std::vector<float> > LF_basic_den
,   std::vector<unsigned> &LF_SAI_mask
,   const unsigned width
,   const unsigned height
,	const unsigned chnls
,   const unsigned N
,   const unsigned kHW
);

void ind_initialize(
    std::vector<unsigned> &row_ind
,   std::vector<std::vector<unsigned> > &col_ind
,   const unsigned max_height
,   const unsigned max_width
,   const unsigned width
,   const unsigned N
,   const unsigned step
,   const unsigned kHW
,   std::vector<float> SAI_weights
);

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
,   bool &useSD_1
,   unsigned &NWien
,   unsigned &nSimWien
,   unsigned &nDispWien
,   unsigned &kWien
,   unsigned &pWien
,   unsigned &tau_2D_wien
,   unsigned &tau_4D_wien
,   unsigned &tau_5D_wien
,   bool &useSD_2
,   unsigned &color_space
,   unsigned &nb_threads
,   char **psnr_file_name
);

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
);


timestamp_t get_timestamp();

#endif // UTILITIES_LF_H_INCLUDED

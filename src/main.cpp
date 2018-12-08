#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string.h>
#include <numeric>

#include "bm3d.h"
#include "bm3d_LF.h"
#include "bm5d.h"
#include "utilities.h"
#include "utilities_LF.h"

#ifdef _OPENMP
    #include <omp.h>
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
 * @file   main.cpp
 * @brief  Main executable file for LFBM5D. 
 *
 * @author Martin Alain <alainm@scss.tcd.ie>
 *
 * This is an implementation of the LFBM5D filter for light field denoising. If you use or adapt this code in your work (either as a stand-alone tool or as a component of any algorithm), you need to cite the following paper:
 * Martin Alain, Aljosa Smolic, "Light Field Denoising by Sparse 5D Transform Domain Collaborative Filtering", IEEE International Workshop on Multimedia Signal Processing (MMSP 2017), 2017.
 * https://v-sense.scss.tcd.ie/wp-content/uploads/2017/08/LFBM5D_MMSP_camera_ready-1.pdf
 *
 * This code is built upon the implementation of the BM3D denoising filter released with the following IPOL paper:
 * Marc Lebrun, "An Analysis and Implementation of the BM3D Image Denoising Method", Image Processing On Line, 2 (2012), pp 175-213, https://doi.org/10.5201/ipol.2012.l-bm3d.
 *
 * Conventions and notations for light field:
 * - Angular size will be denoted awidth and aheight (and associated derivations) for angular width and angular height (noted n_a in the paper);
 * - In general, prefix 'a' before a variable stands for angular;
 * - Indexes for angular dimensions will be denoted s and t (and associated derivations);
 * - Image size will be denoted as usual width and heigth (and associated derivations);
 * - Indexes for image dimensions will be denoted i and j (legacy BM3D implementation);
 * - LF  = Light Field;
 * - SAI = Sub-Aperture Image.
 */


int main(int argc, char **argv)
{
    cout << "*********************************************************************************************************************" << endl;
    cout << "********************************************              START               ***************************************" << endl;
    cout << "*********************************************************************************************************************" << endl;

    //! Define and get params from command Line
    char *LF_input_name, *LF_noisy_name, *LF_basic_name, *LF_denoised_name, *LF_diff_name, *sub_img_name, *sep;
    bool gt_exists;
    unsigned awidth, aheight, s_start, t_start, anHard, anWien, ang_major;
    float fSigma, lambdaHard5D;
    unsigned NHard, NWien;
    unsigned nSimHard, nSimWien, nDispHard, nDispWien;
    unsigned kHard, kWien, pHard, pWien;
    unsigned tau_2D_hard, tau_4D_hard, tau_5D_hard;
    unsigned tau_2D_wien, tau_4D_wien, tau_5D_wien;
    bool useSD_1, useSD_2;
    unsigned color_space;
    unsigned nb_threads;
    char* psnr_file_name;

    if(get_params(argc, argv, &LF_input_name, &sub_img_name, &sep, gt_exists, awidth, aheight, s_start, t_start, anHard, anWien, ang_major, fSigma, lambdaHard5D, &LF_noisy_name, &LF_basic_name, &LF_denoised_name, &LF_diff_name,
                  NHard, nSimHard, nDispHard, kHard, pHard, tau_2D_hard, tau_4D_hard, tau_5D_hard, useSD_1,
                  NWien, nSimWien, nDispWien, kWien, pWien, tau_2D_wien, tau_4D_wien, tau_5D_wien, useSD_2,
                  color_space, nb_threads, &psnr_file_name) != EXIT_SUCCESS)
    {
        cout << "Problem while reading parameters from command line !" << endl;
        return EXIT_FAILURE;
    }

    //! Declarations
	vector<vector<float> > LF, LF_noisy, LF_basic, LF_denoised, LF_diff;
    vector<unsigned> LF_SAI_mask;
    unsigned  width, height, chnls;

    //! Check if OpenMP is used
#ifdef _OPENMP
    if(!nb_threads) //! If input nb_threads is 0, use maximum available
        nb_threads = omp_get_num_procs();

    //! In case where the number of processors isn't a power of 2
    if (!power_of_2(nb_threads))
        nb_threads = closest_power_of_2(nb_threads);
#else
    nb_threads = 1;
#endif

    //! Load light field
    unsigned awh = (unsigned) awidth * aheight;
    if(gt_exists)
    {
        LF.resize(awh);
        timestamp_t start_load = get_timestamp();
        if(load_LF(LF_input_name, sub_img_name, sep, LF, LF_SAI_mask, ang_major, awidth, aheight, s_start, t_start, &width, &height, &chnls) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        timestamp_t end_load = get_timestamp();
        float load_elapsed_secs = float(end_load-start_load) / 1000000.0f;
        cout << "Loading LF elapsed time = " << load_elapsed_secs << "s." << endl;
    }

     //! Add noise
    if(gt_exists)
    {
	    unsigned whc = (unsigned) width * height * chnls;
        LF_noisy.resize(awh);
        for(unsigned st = 0; st < awh; st++)
            LF_noisy[st].resize(whc, 0.0f);
            
        timestamp_t start_noise = get_timestamp();
        cout << endl << "Add noise [sigma = " << fSigma << "] ... " << flush;
        add_noise_LF(LF, LF_SAI_mask, LF_noisy, fSigma);
        timestamp_t end_noise = get_timestamp();
        float noise_elapsed_secs = float(end_noise-start_noise) / 1000000.0f;
        cout << "done in " << noise_elapsed_secs << "s." << endl;

        //! Save light field
        cout << endl << "Save noisy light field..." << endl;
        timestamp_t start_save = get_timestamp();
        if (save_LF(LF_noisy_name, sub_img_name, sep, LF_noisy, LF_SAI_mask, ang_major, awidth, aheight, s_start, t_start, width, height, chnls) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        timestamp_t end_save = get_timestamp();
        float save_elapsed_secs = float(end_save-start_save) / 1000000.0f;
        cout << "done in " << save_elapsed_secs << "s." << endl;
    }
    else //! Read noisy light field
    {
        LF_noisy.resize(awh);
        timestamp_t start_load_noisy = get_timestamp();
        if(load_LF(LF_noisy_name, sub_img_name, sep, LF_noisy, LF_SAI_mask, ang_major, awidth, aheight, s_start, t_start, &width, &height, &chnls) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        timestamp_t end_load_noisy = get_timestamp();
        float load_noisy_elapsed_secs = float(end_load_noisy-start_load_noisy) / 1000000.0f;
        cout << endl << "Loading noisy LF elapsed time = " << load_noisy_elapsed_secs << "s." << endl;
    }

    //! Allocate memory for LF results
	unsigned whc = (unsigned) width * height * chnls;
    LF_diff.resize(awh);
    LF_basic.resize(awh);
    LF_denoised.resize(awh);
    #pragma omp parallel for
    for(int st = 0; st < awh; st++)
    {
        LF_diff[st].resize(whc, 0.0f);
        LF_basic[st].resize(whc, 0.0f);
        LF_denoised[st].resize(whc, 0.0f);
    }

    //! Compute PSNR and RMSE
    vector<float> psnr_noisy, rmse_noisy;
    float avg_psnr_noisy, avg_rmse_noisy;
    float std_psnr_noisy, std_rmse_noisy;
    if(gt_exists)
    {
        if(compute_psnr_LF(LF, LF_noisy, LF_SAI_mask, psnr_noisy, &avg_psnr_noisy, &std_psnr_noisy, rmse_noisy, &avg_rmse_noisy, &std_rmse_noisy) != EXIT_SUCCESS)
            return EXIT_FAILURE;

        cout << endl << "Average PSNR:" << endl;
        cout << "- Noisy light field: " << avg_psnr_noisy << endl;
        
        //! writing measures
        write_psnr_LF(psnr_file_name, "noisy", LF_SAI_mask, ang_major, awidth, aheight, psnr_noisy, avg_psnr_noisy, std_psnr_noisy, rmse_noisy, avg_rmse_noisy, std_rmse_noisy);
    }

    
    //* -------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //* Denoising
    //* -------------------------------------------------------------------------------------------------------------------------------------------------------------------
    cout << endl << " ---> Running LFBM5D filter <--- " << endl << endl;
    timestamp_t start_bm5d = get_timestamp();
    
    //* -------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //* First step - Hard thresholding
    cout << "Step 1 running..." << endl;
    timestamp_t start_step1 = get_timestamp();
    if (run_bm5d_1st_step(fSigma, lambdaHard5D, LF_noisy, LF_SAI_mask, LF_basic, ang_major, awidth, aheight, anHard, width, height, chnls,
                          NHard, nSimHard, nDispHard, kHard, pHard, useSD_1, tau_2D_hard, tau_4D_hard, tau_5D_hard, color_space, nb_threads) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    timestamp_t end_step1 = get_timestamp();
    float step1_elapsed_secs = float(end_step1-start_step1) / 1000000.0f;
    cout << endl << "Step 1 done in " << step1_elapsed_secs << " secs." << endl << endl;

    //! Compute PSNR and RMSE
    vector<float> psnr_basic, rmse_basic;
    float avg_psnr_basic, avg_rmse_basic;
    float std_psnr_basic, std_rmse_basic;
    if(gt_exists)
    {
        if(compute_psnr_LF(LF, LF_basic, LF_SAI_mask, psnr_basic, &avg_psnr_basic, &std_psnr_basic, rmse_basic, &avg_rmse_basic, &std_rmse_basic) != EXIT_SUCCESS)
            return EXIT_FAILURE;
            
        cout << endl << "Average PSNR:" << endl;
        cout << "- Noisy light field: " << avg_psnr_noisy << endl;
        cout << "- Basic light field: " << avg_psnr_basic << endl;
        
        //! writing measures
        write_psnr_LF(psnr_file_name, "basic", LF_SAI_mask, ang_major, awidth, aheight, psnr_basic, avg_psnr_basic, std_psnr_basic, rmse_basic, avg_rmse_basic, std_rmse_basic);

        //! Compute Difference
        cout << endl << "Compute difference... ";
        timestamp_t start_diff = get_timestamp();
        if (compute_diff_LF(LF, LF_basic, LF_SAI_mask, LF_diff, fSigma) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        timestamp_t end_diff = get_timestamp();
        float diff_elapsed_secs = float(end_diff-start_diff) / 1000000.0f;
        cout << "done. Compute diff LF elapsed time = " << diff_elapsed_secs << "s." << endl;
    }

    //! Save light field
	cout << endl << "Save basic light field..." << endl;
    timestamp_t start_save = get_timestamp();
	if (save_LF(LF_basic_name, sub_img_name, sep, LF_basic, LF_SAI_mask, ang_major, awidth, aheight, s_start, t_start, width, height, chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    timestamp_t end_save = get_timestamp();
    float save_elapsed_secs = float(end_save-start_save) / 1000000.0f;
    cout << "done in " << save_elapsed_secs << "s." << endl;

    //* -------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //* Second  step - Wiener filtering
    cout << endl << endl;
    cout << "Step 2 running..." << endl;
    timestamp_t start_step2 = get_timestamp();
    if (run_bm5d_2nd_step(fSigma, LF_noisy, LF_SAI_mask, LF_basic, LF_denoised, ang_major, awidth, aheight, anWien, width, height, chnls,
                          NWien, nSimWien, nDispWien, kWien, pWien, useSD_2, tau_2D_wien, tau_4D_wien, tau_5D_wien, color_space, nb_threads) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    timestamp_t end_step2 = get_timestamp();
    float step2_elapsed_secs = float(end_step2-start_step2) / 1000000.0f;
    cout << endl << "Step 2 done in " << step2_elapsed_secs << " secs." << endl << endl;

    if(gt_exists)
    {
        //! Compute PSNR and RMSE
        vector<float> psnr, rmse;
        float avg_psnr, avg_rmse;
        float std_psnr, std_rmse;
        if(compute_psnr_LF(LF, LF_denoised, LF_SAI_mask, psnr, &avg_psnr, &std_psnr, rmse, &avg_rmse, &std_rmse) != EXIT_SUCCESS)
            return EXIT_FAILURE;
            
        cout << endl << "Average PSNR:"     << endl;
        cout << "- Noisy light field: "     << avg_psnr_noisy << endl;
        cout << "- Basic light field: "     << avg_psnr_basic << endl;
        cout << "- Denoised light field: "  << avg_psnr << endl;
        cout << endl;

        //! writing measures
        write_psnr_LF(psnr_file_name, "denoised", LF_SAI_mask, ang_major, awidth, aheight, psnr, avg_psnr, std_psnr, rmse, avg_rmse, std_rmse);

        //! Compute Difference
        cout << endl << "Compute difference... ";
        timestamp_t start_diff = get_timestamp();
        if (compute_diff_LF(LF, LF_denoised, LF_SAI_mask, LF_diff, fSigma) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        timestamp_t end_diff = get_timestamp();
        float diff_elapsed_secs = float(end_diff-start_diff) / 1000000.0f;
        cout << "done in " << diff_elapsed_secs << "s." << endl;
    }

    //! Save light field
    cout << endl << "Save denoised light field..." << endl;
    start_save = get_timestamp();
	if (save_LF(LF_denoised_name, sub_img_name, sep, LF_denoised, LF_SAI_mask, ang_major, awidth, aheight, s_start, t_start, width, height, chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    end_save = get_timestamp();
    save_elapsed_secs = float(end_save-start_save) / 1000000.0f;
    cout << "done in " << save_elapsed_secs << "s." << endl;

    if(gt_exists)
    {
        cout << endl << "Save diff light field..." << endl;
        start_save = get_timestamp();
        if (save_LF(LF_diff_name, sub_img_name, sep, LF_diff, LF_SAI_mask, ang_major, awidth, aheight, s_start, t_start, width, height, chnls) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        end_save = get_timestamp();
        save_elapsed_secs = float(end_save-start_save) / 1000000.0f;
        cout << "done in " << save_elapsed_secs << "s." << endl << endl;
    }


    timestamp_t end_bm5d = get_timestamp();
    float bm5d_elapsed_secs = float(end_bm5d-start_bm5d) / 1000000.0f;
    cout << "Total LFBM5D computing time = " << step1_elapsed_secs + step2_elapsed_secs << "s." << endl;
    cout << "Total elapsed time = " << bm5d_elapsed_secs << "s." << endl;

    cout << endl;
    cout << "*********************************************************************************************************************" << endl;
    cout << "********************************************         THIS IS THE END          ***************************************" << endl;
    cout << "*********************************************************************************************************************" << endl;

	return EXIT_SUCCESS;
}

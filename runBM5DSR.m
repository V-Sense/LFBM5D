% =========================================================================
% =========================================================================
%
% Author: Martin Alain <alainm@scss.tcd.ie>
% This is an implementation of the SR-LFBM5D filter for light field
% super-resolution. If you use or adapt this code in your work (either as a 
% stand-alone tool or as a component of any algorithm), you need to cite 
% the following paper:
% Martin Alain, Aljosa Smolic, 
% "Light Field Super-Resolution via LFBM5D Sparse Coding", 
% IEEE International Conference on Image Processing (ICIP 2018), 2018
% https://v-sense.scss.tcd.ie/?p=1551
%
% =========================================================================
% =========================================================================

%% Inputs & global SR parameters
[pathRef, pathIn, pathOut, aheight, awidth, s_start, t_start, sub_img_name, sep, factor, psf, bc] = getParams();

%% Run BM5DSR
BM5DSR(pathRef, pathIn, pathOut, aheight, awidth, s_start, t_start, sub_img_name, sep, factor, psf, bc)

%% No ground truth
% Use the function below to super-resolve light fields with unknown high
% resolution
% The functions getParams() and script generateDatasetSR may hve to be
% updated accordingly
% BM5DSR_no_GT(pathIn, pathOut, aheight, awidth, s_start, t_start, sub_img_name, sep, factor, psf, bc)

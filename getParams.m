function [pathRef, pathIn, pathOut, aheight, awidth, s_start, t_start, sub_img_name, sep, factor, psf, bc] = getParams()

%% SR downsampling factor
factor = 2; % 1, 2, 3, or 4
psf = getBlurringKernel(factor);

%% LF dimensions and naming
aheight = 3;
awidth = 3;
s_start = 1;
t_start = 1;
sub_img_name = 'SAI';
sep = '_';

%% Border to crop to compute PSNR
bc = 17;

%% Paths to LFs
% Input path
pathRef = './testing/sourceLF/';
pathIn = './testing/';

% Output path
pathOut = pathIn;

end


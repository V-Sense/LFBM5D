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


function BM5DSR(pathRef, pathIn, pathOut, aheight, awidth, s_start, t_start, sub_img_name, sep, factor, psf, bc)
% BM5DSR super-resolves an input low resolution light field (LF) using the
% SR-LFBM5D filter.
% 
% - pathRef points to the folder containing the reference LF (used to 
% compute PSNR)
% - pathIn points to the folder containing the input low resolution LF in 
% pathIn/inputLR and the corresponding first high resolution estimate 
% (usually obatined with bicubic upsampling) in pathIn/inputHR
% - pathOut points to the folder where the super-resolved LFs are written, 
% in pathOut/SR_xX/output_BM5D and pathOut/SR_xX/output_BP respectively,
% where X is the scaling factor
% - aheight, awidth correspond to the angular size of the LF
% - s_start, t_start correspond to the first angular indexes of the LF,
% used for the top left image
% - sub_img_name, sep correspond to the common name for all sub-aperture 
% images and the character to separate angular indexes respectively
% - factor is the scaling factor
% - psf is the blurring kernel used to get the input low resolution LF
% - bc correspond to the boreder (in pixels) to be cropped to compute the
% PSNR

%% Main loop parameters and init
switch factor
    case 1
        numIt = 10;
    case 2
        numIt = 10;
    case 3
        numIt = 30;
    case 4
        numIt = 50;
    otherwise
        numIt = 10;
end

ZHR     = cell(aheight, awidth);
ZLR     = cell(aheight, awidth);
ZBic    = cell(aheight, awidth);
ZBM5D   = cell(aheight, awidth);
ZIBP    = cell(aheight, awidth);

%  Quadatric decrease of Sigma between sigInit and sigEnd
sigInit = 12*factor;
sigEnd = 1.0;
a = 0.01;
% a = 1;
b = (sigEnd-sigInit)/(numIt-1) - (numIt-1)*a;
c = sigInit;
sigList = zeros(1, numIt);
for x = 0:(numIt-1)
    sigList(x+1) = a*x*x + b*x + c;
end

if nnz(sigList < 0)
    error('Negative sigma value, modify ''a'' parameter');
end


% Back projection parameter
betaBP = 1.75;

% Read inputs and compute bicubic LF PSNR
PBic  = zeros(aheight, awidth);
PBicCrop  = zeros(aheight, awidth);

str1 = [pathIn '/SR_x' num2str(factor) '/inputHR/' sub_img_name sep '%02d' sep '%02d.png'];
str2 = [pathRef sub_img_name sep '%02d' sep '%02d.png'];
str3 = [pathIn '/SR_x' num2str(factor) '/inputLR/' sub_img_name sep '%02d' sep '%02d.png'];
for t = 1:aheight
    for s = 1:awidth
        % Read bicubic
        nameIm = sprintf(str1, t-1+t_start, s-1+s_start);
        IBic = imread(nameIm);
        ZBic{t, s} = IBic;
        
        % Read original HR image
        nameIm = sprintf(str2, t-1+t_start, s-1+s_start);
        IRef = imread(nameIm);
        
        % Compute HR size
        yResLR = floor(size(IRef,1) / factor);
        xResLR = floor(size(IRef,2) / factor);
        yResHR = factor * yResLR;
        xResHR = factor * xResLR;
        
        IRef = IRef(1:yResHR,1:xResHR,:);
        ZHR{t, s} = IRef;
        
        % Read LR image
        nameIm = sprintf(str3, t-1+t_start, s-1+s_start);
        ILR = imread(nameIm);
        ZLR{t, s} = ILR;
        
        % Compute PSNR
        PBic(t, s) = psnr(IBic, IRef);
        PBicCrop(t, s) = psnr(IBic(bc:(end-bc+1),bc:(end-bc+1),:), IRef(bc:(end-bc+1),bc:(end-bc+1),:));
    end
end
PBicCropMean  = mean(PBicCrop(:));

PBM5Dit = zeros(numIt+1,1); PBM5Dit(1) = PBicCropMean;
PIBPit  = zeros(numIt+1,1); PIBPit(1)  = PBicCropMean;


%% Main loop
timeStart = tic;
for it = 1:numIt
    fprintf(['ITERATION ' num2str(it) ' ... '])
    loopStart = tic;
    % Create results dir
    BM5DDirHT = [pathOut '/SR_x' num2str(factor) '/output_BM5D/HT/'];
    if ~exist(BM5DDirHT, 'dir')
        mkdir(BM5DDirHT);
    end
    BM5DDirWien = [pathOut '/SR_x' num2str(factor) '/output_BM5D/Wiener/'];
    if ~exist(BM5DDirWien, 'dir')
        mkdir(BM5DDirWien);
    end
    IBPDDir = [pathOut '/SR_x' num2str(factor) '/output_BP/'];
    if ~exist(IBPDDir, 'dir')
        mkdir(IBPDDir);
    end
    
    if it == 1
        inputDir = [pathIn '/SR_x' num2str(factor) '/inputHR/'];
    else
        inputDir = IBPDDir;
    end
    
    %% First step: BM5D HT
    % Call BM5DSR exe
    if isunix
        exe = './LFBM5DSR';
    else
        exe = 'LFBM5DSR.exe';
    end
    sig = sigList(it);
    nb_threads = 8;
    cmd = [exe, ' none ' sub_img_name ' ' sep ' ' num2str(awidth) ' ' num2str(aheight) ' ' num2str(s_start) ' ' num2str(t_start) ' 1 1 row ' ...
        num2str(sig) ' 2.7 '...
        inputDir ' ' BM5DDirHT ' ' BM5DDirWien ' none ' ...
        '32 18 6 8 7 dct dct haar 0 32 18 6 4 2 id dct haar 0 ycbcr ' num2str(nb_threads) ' nooutputfile'];
    echo = ''; % use '-echo' to see the output of the BM5DSR exe
    [status, cmdout] = system(cmd, echo);
    
    if status ~= 0
        error(['Error when running LFBM5DSR at iteration ' num2str(it)]);
    end
    
    % Compute LF PSNR
    PSRBM5D  = zeros(aheight, awidth);
    PSRBM5DCrop  = zeros(aheight, awidth);
    str = [BM5DDirHT sub_img_name sep '%02d' sep '%02d.png'];
    % Uncomment the following line in case the LFBM5DSR exe is compiled with the Wiener step
%     str = [BM5DDirWien sub_img_name sep '%02d' sep '%02d.png']; 
    for t = 1:aheight
        for s = 1:awidth
            % Read BM5D ouput
            nameIm = sprintf(str, t-1+s_start, s-1+s_start);
            IBM5D = imread(nameIm);
            ZBM5D{t, s} = IBM5D;
            
            % Compute PSNR
            PSRBM5D(t, s) = psnr(IBM5D, ZHR{t, s});
            PSRBM5DCrop(t, s) = psnr(IBM5D(bc:(end-bc+1),bc:(end-bc+1),:), ZHR{t, s}(bc:(end-bc+1),bc:(end-bc+1),:));
        end
    end
    PSRBM5DCropMean  = mean(PSRBM5DCrop(:));
    PBM5Dit(it+1)  = PSRBM5DCropMean;
        
    
    %% Second step: Back projection
    % Compute BP for each image and PSNR
    PIBP  = zeros(aheight, awidth);
    PIBPCrop  = zeros(aheight, awidth);
    str = [IBPDDir sub_img_name sep '%02d' sep '%02d.png'];
    for t = 1:aheight
        for s = 1:awidth
            % Downsample output of BM5D
            % Blurring
            IBM5D = double(ZBM5D{t, s});
            IBM5DLR = IBM5D;
            for ch = 1:3
                % Blur the image using the NCSR code
                IBM5DLR(:,:,ch) = blur('fwd', IBM5D(:,:,ch), psf);
            end
            % Compute the down-sampling process
            IBM5DLR = IBM5DLR(1:factor:end,1:factor:end,:);
            
            diffLR = double(ZLR{t, s}) - IBM5DLR;
            diffHR = imresize(diffLR, factor, 'bicubic');
            
            if factor == 4
                % Use guided filtering to limit ringing artifacts of IBP
                % Scale diff HR to [0 1]
                minDiff = min(min(diffHR));
                diffHR = bsxfun(@minus, diffHR, minDiff);
                maxDiff = max(max(diffHR));
                diffHR = bsxfun(@rdivide, diffHR, maxDiff);
                
                Ig = squeeze(ZBic{t, s});
                Ig = bsxfun(@minus, Ig, min(min(Ig)));
                Ig = bsxfun(@rdivide, Ig, max(max(Ig)));
                diffHR = double(imguidedfilter(diffHR, Ig, ...
                    'DegreeOfSmoothing', 10e-4*diff(getrangefromclass(Ig)).^2, ...
                    'NeighborhoodSize', [3 3]));
                
                % Rescale to original range
                diffHR = bsxfun(@times, diffHR, maxDiff);
                diffHR = bsxfun(@plus, diffHR, minDiff);
            end
            
            IBP = IBM5D + betaBP .* diffHR;
            
            ZIBP{t, s} = uint8(IBP);
            
            % Write BP ouput
            nameIm = sprintf(str, t-1+s_start, s-1+s_start);
            imwrite(ZIBP{t, s}, nameIm);
            
            % Compute PSNR
            PIBP(t, s) = psnr(ZIBP{t, s}, ZHR{t, s});
            PIBPCrop(t, s) = psnr(ZIBP{t, s}(bc:(end-bc+1),bc:(end-bc+1),:), ZHR{t, s}(bc:(end-bc+1),bc:(end-bc+1),:));
        end
    end
    PIBPCropMean  = mean(PIBPCrop(:));
    PIBPit(it+1)  = PIBPCropMean;
    
    fprintf('done in %f s.\n', toc(loopStart))
    
    %% Stop loop if converging or decreasing
    epsCvg = 0.01;
    if it > 2
        if abs(PIBPit(it+1)-PIBPit(it)) < epsCvg && abs(PIBPit(it+1)-PIBPit(it-1)) < epsCvg
            disp('Convergence condition reached')
            break
        end
        if PIBPit(it+1) < PIBPit(it) && PIBPit(it+1) < PIBPit(it-1)
            disp('Decreasing condition reached')
            break
        end
    end
end

timeElapsed = toc(timeStart);
fprintf('Main loop done in %f s.\n', timeElapsed)

% Plot PSNR over iterations and center image of LF
PBM5Dit(PBM5Dit==0) = [];
PIBPit(PIBPit==0)   = [];
figure;
hold on; plot(0:(size(PBM5Dit,1)-1), PBM5Dit, 'x-b', 'LineWidth', 1.5)
hold on; plot(0:(size(PIBPit,1)-1),  PIBPit,  'o-r', 'LineWidth', 1.5)
xlabel('Iteration index')
ylabel('PSNR [dB]')
legend('SR-LFBM5D 1st step', 'SR-LFBM5D 2nd step', 'Location', 'southeast')
title('SR-LFBM5D Mean PSNR over iterations')

figure;
t = floor(aheight/2);
s = floor(awidth/2);
subplot(2, 2, 1); imshow(ZHR{t, s});    title('Original LF central view')
subplot(2, 2, 2); imshow(ZBic{t, s});   title(sprintf('Bicubic upsampling, PSNR = %0.2fdB', PBicCrop(t,s)))
subplot(2, 2, 3); imshow(ZBM5D{t, s});  title(sprintf('SR-LFBM5D 1st step, PSNR = %0.2fdB', PSRBM5DCrop(t,s)))
subplot(2, 2, 4); imshow(ZIBP{t, s});   title(sprintf('SR-LFBM5D 2nd step, PSNR = %0.2fdB', PIBPCrop(t,s)))

%% Save PSNR results
save( [pathOut '/SR_x' num2str(factor) '/PSNR_SR-LFBM5D.mat'], ...
      'PBic', 'PBicCrop', 'PSRBM5D', 'PSRBM5DCrop', 'PIBP', 'PIBPCrop');

fprintf(1, 'Mean PSNR Bicubic = %0.4fdB\nMean PSNR SR-LFBM5D 1st step = %0.4fdB\nMean PSNR SR-LFBM5D 2nd step = %0.4fdB\n',PBicCropMean, max(PBM5Dit), max(PIBPit));

end

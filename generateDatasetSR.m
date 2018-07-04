
%% Inputs & global SR parameters
[pathRef, pathIn, pathOut, aheight, awidth, s_start, t_start, sub_img_name, sep, factor, psf, bc] = getParams();

%% Read HR and generate LR
str = [pathRef sub_img_name sep '%02d' sep '%02d.png'];
Z   = cell(aheight, awidth);
ZLR = cell(aheight, awidth);
for t = 1:aheight
    for s = 1:awidth
        % Read HR
        nameIm = sprintf(str, t-1+t_start, s-1+s_start);
        IHR = double(imread(nameIm));
        
        % Compute HR size
        yResLR = floor(size(IHR,1) / factor);
        xResLR = floor(size(IHR,2) / factor);
        yResHR = factor * yResLR;
        xResHR = factor * xResLR;
        
        IHR = IHR(1:yResHR,1:xResHR,:);
        
        % Generate LR
        ILR = IHR;
        for ch = 1:3
            % Blur the image using the NCSR code
            ILR(:,:,ch) = blur('fwd', IHR(:,:,ch), psf);
        end
        % Compute the down-sampling process
        ILR = ILR(1:factor:end,1:factor:end,:);
        
        Z{t, s} = uint8(IHR);
        ZLR{t, s} = uint8(ILR);
    end
end

%% Write LR
dirOut = [pathIn '/SR_x' num2str(factor) '/inputLR/'];
if ~exist(dirOut, 'dir')
    mkdir(dirOut)
end

str = [ dirOut sub_img_name sep '%02d' sep '%02d.png'];
for t = 1:aheight
    for s = 1:awidth
        nameIm = sprintf(str, t-1+t_start, s-1+s_start);
        imwrite(ZLR{t, s}, nameIm);
    end
end

%% Write LR upsampled
dirOut = [pathIn '/SR_x' num2str(factor) '/inputHR/'];
if ~exist(dirOut, 'dir')
    mkdir(dirOut)
end

str = [dirOut sub_img_name sep '%02d' sep '%02d.png'];
ZHR = cell(aheight, awidth);
for t = 1:aheight
    for s = 1:awidth
        nameIm = sprintf(str, t-1+t_start, s-1+s_start);
        imwrite(imresize(ZLR{t, s}, factor, 'bicubic'), nameIm);
        ZHR{t, s} = uint8(imresize(ZLR{t, s}, factor, 'bicubic'));
    end
end

%% Compute PSNR
vPSNR     = zeros(awidth*aheight,1);
vPSNRCrop = zeros(awidth*aheight,1);
vidx = 1;
for t = 1:aheight
    for s = 1:awidth
        vPSNR(vidx) = psnr(ZHR{t, s}, Z{t, s});
        vPSNRCrop(vidx) = psnr(ZHR{t, s}(bc:(end-bc+1),bc:(end-bc+1),:), Z{t, s}(bc:(end-bc+1),bc:(end-bc+1),:));
        vidx = vidx+1;
    end
end
mean(vPSNR)
mean(vPSNRCrop)



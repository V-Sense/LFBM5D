function psf = getBlurringKernel(factor)

switch factor
    case 1
        gaussWinSize = 7;
        GaussSig = 1.6;
        
    case 2
        gaussWinSize = 7;
        GaussSig = 1.6;
    
    case 3
        gaussWinSize = 9;
        GaussSig = 1.6;
    
    case 4
        gaussWinSize = 11;
        GaussSig = 1.6;
        
    otherwise
        gaussWinSize = 7;
        GaussSig = 1.6;
end

psf = fspecial('gaussian', gaussWinSize, GaussSig);

end

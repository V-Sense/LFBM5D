This is a list of instructions / recommendations to help you compile the BM5D project on Windows.
This was tested on Windows 10 x64.

Note that if you are a Windows 10 user and only intend to use this code for the denoising and not for the super-resolution (branch SR), we highly recommend installing Ubuntu from the Microsoft Store, and follow the procedure for Linux, which is much easier than the Windows install.

## Microsoft visual studio
We recommend installing the latest version of the visual studio IDE at https://visualstudio.microsoft.com/ (we used Community 2017 for testing).
We recommend that for running any command line you use the Developer Command Prompt for VS (you may have to run it as administrator).

## Install dependencies

### FFTW
1) Download the dll files from http://fftw.org/install/windows.html and follow the instructions on that website. You basically have to generate the .lib files by running `lib /machine:x64 /def:libfftw3f-3.def` (you might have to change the /machine setting depending on your configuration). 
2) In src/CMakeLists.txt, add the path to the fftw dll directory lines 34 and 100, e.g. "C:/path/2/fftw", and add the path to the .lib lines 55 and 121, e.g. "C:/path/2/fftw/libfftw3f-3.lib".

### libpng and zlib
1) Download libpng and zlib source code from http://www.libpng.org/pub/png/libpng.html
2) Follow the section "VI. Building with project files" of the INSTALL file in libpng folder: basically go to projects/vstudio subfolder and read READDME.txt.
3) Modify the zlib.props file by indicating the zlib folder. Run vstudio.sln and compile it with the Release Library option.

Note: Make sure to choose the right configuration between Win32 and x64, default one is Win32. To create an x64 configuration, next to 'release library' option, in the 'Win32' menu, select 'Configuration manager'. Under Active solution platform' select '<New...>', then 'Type or select the new platform, select x64.). 
To compile the 'zlib' and 'libpng' projects, you might have to change compilation flag to NOT treat warnings as errors, in project properties -> C/C++ -> General -> Treat Warnings As Errors -> No (/WX-).
	
4) Finally, in src/CMakeLists.txt, add the path to the libpng directory lines 35 and 101, e.g. "C:/path/to/lpng1636". In addition, add the path to the .lib lines 56, 57, 1221 and 123, e.g. "C:/path/to/lpng1636/projects/vstudio/x64/Release Library/libpng16.lib" and "C:/path/to/lpng1636/projects/vstudio/x64/Release Library/zlib.lib"

## Usage
You will have to copy libfftw3f-3.dll in the same folder as your executable.
	
	


PROJECT(LFBM5Ddenoising)

set(${PROJECT_NAME}_PROJECT_SRCS 
	main.cpp
	
	bm3d.h
	bm3d.cpp
	bm5d.h
	bm5d.cpp
	bm5d_core_processing.h
	bm5d_core_processing.cpp
	io_png.h
	io_png.c
	lib_transforms.h
	lib_transforms.cpp
	
	mt19937ar.c
	mt19937ar.h
	
	utilities.h
	utilities.cpp
	utilities_LF.h
	utilities_LF.cpp
)

SOURCE_GROUP(Headers REGULAR_EXPRESSION "[.]h$")
SOURCE_GROUP(Config\\Headers REGULAR_EXPRESSION "_Config[.]h$")
SOURCE_GROUP(Config\\Sources REGULAR_EXPRESSION "_Config[.]cpp$")

IF(WIN32)
#If libfftw and libpng libraries are not in the environment path, then specify their location here
# Remember that even if you are under windows cmake does not like \ in path
set(LFBM5D_INCLUDE_DIR ${LFBM5D_INCLUDE_DIR}
"C:/path/to/fftw-3.3.5-dll64"
"C:/path/to/lpng1636"
)
ENDIF(WIN32)
IF(UNIX)
set(LFBM5D_INCLUDE_DIR ${LFBM5D_INCLUDE_DIR}
${PNG_INCLUDE_DIR}
${FFTW3_LIBRARIES_INCLUDE_DIR}
)
ENDIF(UNIX)

INCLUDE_DIRECTORIES(${LFBM5D_INCLUDE_DIR})

ADD_EXECUTABLE(${PROJECT_NAME} ${${PROJECT_NAME}_PROJECT_SRCS})
IF(MSVC)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
ENDIF(MSVC)

IF(WIN32)
#If libfftw and libpng libraries are not in the environment path, then specify their location here
set(LFBM5D_LINK_LIBRARIES ${LFBM5D_LINK_LIBRARIES}
	"C:/path/to/fftw-3.3.5-dll64/libfftw3f-3.lib"
	"C:/path/to/lpng1636/projects/vstudio/x64/Release Library/libpng16.lib"
	"C:/path/to/lpng1636/projects/vstudio/x64/Release Library/zlib.lib"
)
ENDIF(WIN32)
IF(UNIX)
set(LFBM5D_LINK_LIBRARIES ${LFBM5D_LINK_LIBRARIES}
fftw3f
png
${ZLIB_LIBRARIES}
)
ENDIF(UNIX)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${LFBM5D_LINK_LIBRARIES})


PROJECT(LFBM3Ddenoising)

set(${PROJECT_NAME}_PROJECT_SRCS 
	main_bm3d_LF.cpp
	
	bm3d.h
	bm3d.cpp
	bm3d_LF.cpp
	io_png.h
	io_png.c
	lib_transforms.h
	lib_transforms.cpp
	
	mt19937ar.c
	mt19937ar.h
	
	utilities.h
	utilities.cpp
	utilities_LF.h
	utilities_LF.cpp
)

SOURCE_GROUP(Headers REGULAR_EXPRESSION "[.]h$")
SOURCE_GROUP(Config\\Headers REGULAR_EXPRESSION "_Config[.]h$")
SOURCE_GROUP(Config\\Sources REGULAR_EXPRESSION "_Config[.]cpp$")

IF(WIN32)
#If libfftw and libpng libraries are not in the environment path, then specify their location here
set(LFBM5D_INCLUDE_DIR ${LFBM5D_INCLUDE_DIR}
"C:/path/to/fftw-3.3.5-dll64"
"C:/path/to/lpng1636"
)
ENDIF(WIN32)
IF(UNIX)
set(LFBM5D_INCLUDE_DIR ${LFBM5D_INCLUDE_DIR}
${PNG_INCLUDE_DIR}
${FFTW3_LIBRARIES_INCLUDE_DIR}
)
ENDIF(UNIX)

INCLUDE_DIRECTORIES(${LFBM5D_INCLUDE_DIR})

ADD_EXECUTABLE(${PROJECT_NAME} ${${PROJECT_NAME}_PROJECT_SRCS})
IF(MSVC)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
ENDIF(MSVC)

IF(WIN32)
#If libfftw and libpng libraries are not in the environment path, then specify their location here
set(LFBM5D_LINK_LIBRARIES ${LFBM5D_LINK_LIBRARIES}
	"C:/path/to/fftw-3.3.5-dll64/libfftw3f-3.lib"
	"C:/path/to/lpng1636/projects/vstudio/x64/Release Library/libpng16.lib"
	"C:/path/to/lpng1636/projects/vstudio/x64/Release Library/zlib.lib"
)
ENDIF(WIN32)
IF(UNIX)
set(LFBM5D_LINK_LIBRARIES ${LFBM5D_LINK_LIBRARIES}
fftw3f
png
${ZLIB_LIBRARIES}
)
ENDIF(UNIX)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${LFBM5D_LINK_LIBRARIES})
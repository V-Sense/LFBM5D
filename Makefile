# C source code
CSRC	= io_png.c \
		mt19937ar.c
# C++ source code
CXXSRC	= main.cpp \
		bm3d.cpp \
		bm5d.cpp \
		bm5d_core_processing.cpp \
		utilities.cpp \
		utilities_LF.cpp \
		lib_transforms.cpp

# all source code
SRC	= $(CSRC) $(CXXSRC)

# C objects
COBJ	= $(CSRC:.c=.o)
# C++ objects
CXXOBJ	= $(CXXSRC:.cpp=.o)
# all objects
OBJ	= $(COBJ) $(CXXOBJ)
# binary target
BIN	= LFBM5DSR

# use debug with `make DBG=1`
ifdef DBG
# C optimization flags
COPT	= -O0 -g -ftree-vectorize -funroll-loops -fno-inline

# C++ optimization flags
CXXOPT	= $(COPT)
else
# C optimization flags
COPT	= -O3 -ftree-vectorize -funroll-loops

# C++ optimization flags
CXXOPT	= $(COPT)
endif

# C compilation flags
CFLAGS	= $(COPT) -Wall -Wextra \
	-Wno-write-strings -ansi
# C++ compilation flags
CXXFLAGS	= $(CXXOPT) -Wall -Wextra \
	-Wno-write-strings -Wno-deprecated -ansi
# link flags
LDFLAGS	= -lpng -lfftw3f -lm

# use openMP with `make OMP=1`
ifdef OMP
CFLAGS	+= -fopenmp
CXXFLAGS	+= -fopenmp
LDFLAGS += -lgomp
else
CFLAGS	+= -Wno-unknown-pragmas
CXXFLAGS  += -Wno-unknown-pragmas
endif

# partial compilation of C source code
%.o: %.c %.h
	$(CC) -c -o $@  $< $(CFLAGS)
# partial compilation of C++ source code
%.o: %.cpp %.h
	$(CXX) -c -o $@  $< $(CXXFLAGS)

# link all the object code
$(BIN): $(OBJ) $(LIBDEPS)
	$(CXX) -o $@ $(OBJ) $(LDFLAGS)
	
.PHONY: clean
	
clean:
	rm -f *.o
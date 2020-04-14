#ifndef LIB_TRANSFORMS_INCLUDED
#define LIB_TRANSFORMS_INCLUDED

#include<vector>

//! Compute a Bior1.5 2D
void bior_2d_forward(
    std::vector<float> const& input
,   std::vector<float> &output
,   const unsigned N
,   const unsigned d_i
,   const unsigned r_i
,   const unsigned d_o
,   std::vector<float> const& lpd
,   std::vector<float> const& hpd
);

//! Compute a Bior1.5 2D inverse
void bior_2d_inverse(
    std::vector<float> &signal
,   const unsigned N
,   const unsigned d_s
,   std::vector<float> const& lpr
,   std::vector<float> const& hpr
);

//! Precompute the Bior1.5 coefficients
void bior15_coef(
    std::vector<float> &lp1
,   std::vector<float> &hp1
,   std::vector<float> &lp2
,   std::vector<float> &hp2
);

//! Apply Walsh-Hadamard transform (non normalized) on a vector of size N = 2^n
void hadamard_transform(
    std::vector<float> &vec
,   std::vector<float> &tmp
,   const unsigned N
,   const unsigned d
);

//! Process the log2 of N
unsigned log2(
    const unsigned N
);

//! Obtain index for periodic extension
void per_ext_ind(
    std::vector<unsigned> &ind_per
,   const unsigned N
,   const unsigned L
);

void haar_forward(
    std::vector<float> &vec
,   std::vector<float> &tmp
,   const unsigned N
,   const unsigned D
);

void haar_inverse(
    std::vector<float> &vec
,   std::vector<float> &tmp
,   unsigned n
,   const unsigned N
,   const unsigned D
);

#endif // LIB_TRANSFORMS_INCLUDED

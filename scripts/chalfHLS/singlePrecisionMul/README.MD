VivadoHLS version 2017.1

mul.cpp
Specify pragma to ulter the number of DSP used. Available pragmas are:

#pragma HLS RESOURCE variable=product core=FMul_nodsp
#pragma HLS RESOURCE variable=product core=FMul_meddsp
#pragma HLS RESOURCE variable=product core=FMul_maxdsp

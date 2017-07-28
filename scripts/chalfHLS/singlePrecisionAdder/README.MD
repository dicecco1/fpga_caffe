VivadoHLS version 2017.1

adders.cpp
Specify pragma to ulter the number of DSP used. Available pragmas are:

#pragma HLS RESOURCE variable=sum core=FAddSub_nodsp
#pragma HLS RESOURCE variable=sum core=FAddSub_fulldsps

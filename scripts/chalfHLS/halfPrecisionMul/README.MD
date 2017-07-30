VivadoHLS version 2017.1  

mul.cpp  
Specify pragma to ulter the number of DSP used. Available pragmas are:  
```
#pragma HLS RESOURCE variable=product core=HMul_nodsp  
#pragma HLS RESOURCE variable=product core=HMul_fulldsp  
#pragma HLS RESOURCE variable=product core=HMul_maxdsp
```

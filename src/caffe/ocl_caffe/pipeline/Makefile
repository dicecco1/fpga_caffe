EXECUTABLES = test

all: $(EXECUTABLES)

ifdef OPENCL_INC
  ACL_CFLAGS = -I$(OPENCL_INC)
endif

ifdef OPENCL_LIB
  ACL_LDFLAGS = -L$(OPENCL_LIB)
endif

CL_CFLAGS = $(XILINX_OPENCL)/runtime/include/1_2
CL_LDFLAGS = $(XILINX_OPENCL)/runtime/lib/x86_64 -lOpenCL

test: test.c cl-helper.c
	gcc -I $(CL_CFLAGS) -L $(CL_LDFLAGS) -std=gnu99 -o$@ $^ -lrt -lm -g

clean:
	rm -rf $(EXECUTABLES) *.o *xocc*.log *temp* *sdaccel* *shared* *tmp* *.Xil* kernels/.Xil binary/.Xil

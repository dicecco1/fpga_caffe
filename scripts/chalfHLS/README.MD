Vivado 2016.3
mulHLS: custom precision multiplication with rounding to nearest
mulHLS_roundOFF: custom precision multiplication with rounding to zero
adderHLS: custom precision addition with rounding to nearest
adderHLS_roundOFF: custom precision addition with rounding to zero

Description:
The programs in these folders (script.py) targets VivadoHLS 2016.3
Testing resource utilization of customized precision operations with place and route exponent size 4 - 7 and mentissa size 2 - 14
Reports are available in impl_reports/ of every folder

script.py
Run this script to test resource utilization of all exponent and mentissa range with place and route.
Graphs will be shown automatically after HLS finishes.
All results will also be stored in .csv file (adders_round.csv)
This might take up to 2 hours depends on your machine.

```python
>>> python script.py
```

After results shown in impl_reports/ run above command with -O flag to skip HLS and only generate graphs

```python
>>> python -O script.py
```

Graphs generation assumes all test results are available in impl_reports/ (52 results in total).
If need to generate graphs for partial results change report reading range as commented inside script.py 



Vivado 2017.1
singlePrecisionAdder: used to test single precision addition under different DSP utilization
halfPrecisionAdder: used to test half precision addition under different DSP utilization
singlePrecisionMul: used to test single precision multiplication under different DSP utilization
halfPrecisionMul: used to test half precision multilication under different DSP utilization

Description:
These programs targets VivadoHLS 2017.1
Testing resource utilization of Single Precision addition with the use of different number of DSPs
Run with:
```
vivado_hls run_hls.tcl
```
Reports are available in adders_prj/solution1/impl/report/verilog/adders_export.rpt

For difference core settings of DSPs, see README.MD inside each directory


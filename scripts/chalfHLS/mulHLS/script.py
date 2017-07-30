import subprocess
import os
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

class report:
	def __init__(self, CLB, LUT, FF, DSP, BRAM, SRL, EXP, MEN, CP_required, CP_achieved_post_synthesis, CP_achieved_post_implementation):
		self.CLB = CLB
		self.LUT = LUT
		self.FF = FF
		self.DSP = DSP
		self.BRAM = BRAM
		self.SRL = SRL
		self.EXP = EXP
		self.MEN = MEN
		self.CP_required = CP_required
		self.CP_achieved_post_synthesis = CP_achieved_post_synthesis
		self.CP_achieved_post_implementation = CP_achieved_post_implementation


report_list = []
exp_4 = []
exp_5 = []
exp_6 = []
exp_7 = []

# change range here to specify testing range for exponent and mentissa. eg. (4,8) stands for [4,8)
for exponent in range (4,8):
	for mentisa in range (2,15):

		# change this directory to the absolute path to impl_reports
		directory = '/home/lin/Desktop/chalfHLS/mulHLS/impl_reports'
		# run in python -O script.py or python script.py
		if __debug__:
			# run with -O flag to disable hls synthesis and graph generation
			# make a copy of the original_para.txt
			thisFile = "para.txt"
			copyfile('original_para.txt','para.txt')
			f_origin = open("original_para.txt")
			f_temp = open("para.txt", "w+")
			for line in f_origin:
				 if 'EXP_SIZE' in line:
				 	f_temp.write("#define EXP_SIZE " + str(exponent) + "\n")
				 elif 'MANT_SIZE' in line:
				 	f_temp.write("#define MANT_SIZE " + str(mentisa) + "\n")
				 else:
				 	f_temp.write(line)

			# change extension
			base = os.path.splitext(thisFile)[0]
			os.rename(thisFile, base + ".hpp")
			f_temp.close()

			# finish generating macro file, execute hls
			subprocess.call(["vivado_hls", "run_hls.tcl"])

			# finish hls, archieve report
			# change this directory to the absolute path of reports and impl_reports.
			copyfile('/home/lin/Desktop/chalfHLS/mulHLS/mul_prj/solution1/impl/report/verilog/mul_export.rpt', '/home/lin/Desktop/chalfHLS/mulHLS/impl_reports/temp_report.rtp')
			for file in os.listdir(directory):
				if file.startswith("temp_report"):
					os.rename(os.path.join(directory, file), os.path.join(directory, 'exp=' + str(exponent) +'men=' + str(mentisa) + '.txt'))
		
		# with -O flag only regenerate graph and report detailed usage in command line
		# extract information from report
		reportname = 'exp=' + str(exponent) + 'men=' + str(mentisa) + '.txt'
		with open(os.path.join(directory, reportname)) as f:
			for line in f:
				data = line.split()
				if 'CLB' in line:
					CLB = int(data[1].lstrip().rstrip())
				if 'LUT' in line:
					LUT = int(data[1].lstrip().rstrip())
					if exponent == 4:
						exp_4.append(LUT)
					if exponent == 5:
						exp_5.append(LUT)
					if exponent == 6:
						exp_6.append(LUT)
					if exponent == 7:
						exp_7.append(LUT)
				if 'FF' in line:
					FF = int(data[1].lstrip().rstrip())
				if 'DSP' in line:
					DSP = int(data[1].lstrip().rstrip())
				if 'BRAM' in line:
					BRAM = int(data[1].lstrip().rstrip())
				if 'SRL' in line:
					SRL = int(data[1].lstrip().rstrip())
				if 'CP required' in line:
					CP_required_str = data[2].lstrip().rstrip()
					CP_required = float(CP_required_str)
				if 'CP achieved post-synthesis' in line:
					CP_achieved_post_synthesis = float(data[3].lstrip().rstrip())
				if 'CP achieved post-implemetation' in line:
					CP_achieved_post_implementation = float(data[3].lstrip().rstrip())
		report_list.append(report(CLB, LUT, FF, DSP, BRAM, SRL, exponent, mentisa, CP_required, CP_achieved_post_synthesis, CP_achieved_post_implementation))


# save results in .cvs file
csv_f = open('mul_round.csv', 'wt')
writer = csv.writer(csv_f)
writer.writerow(('Multiplication', 'Round-to-Nearest'))
writer.writerow(('EXP', 'MAN', 'CLB', 'LUT', 'FF',  'DSP', 'BRAM', 'SRL', 'CP_req', 'CP_post_sysn', 'CP_post_impl'))
# total number of files is 52, change this accordingly to only save the partial results desired
for i in range(52):
	writer.writerow( ( str(report_list[i].EXP), str(report_list[i].MEN),  str(report_list[i].CLB),str(report_list[i].LUT),str(report_list[i].FF),str(report_list[i].DSP),str(report_list[i].BRAM),str(report_list[i].SRL),str(report_list[i].CP_required),str(report_list[i].CP_achieved_post_synthesis),str(report_list[i].CP_achieved_post_implementation),  ) )



# plot the data
man = range(2,15)
# the number of mentissa chosen in default is 13 (from 2 - 14)
single_no_DSP = [543] * 13
single_1_DSP = [224] * 13
single_2_DSP = [80] * 13
single_3_DSP = [86] * 13
no_DSP = [179] * 13
two_DSP = [35] * 13
one_DSP = [46] * 13
plt.ylabel('LUTs')
plt.xlabel('Mantissa')
plt.plot(man, exp_4, label="Exponent = 4")
plt.plot(man, exp_5, label="Exponent = 5")
plt.plot(man, exp_6, label="Exponent = 6")
plt.plot(man, exp_7, label="Exponent = 7")
plt.plot(man, single_no_DSP, label="SP 0 DSP")
plt.plot(man, single_1_DSP, label="SP 1 DSP")
plt.plot(man, single_2_DSP, label="SP 2 DSP")
plt.plot(man, single_3_DSP, label="SP 3 DSP")
plt.plot(man, no_DSP, label = "HP 0 DSP")
plt.plot(man, two_DSP, label = "HP 2 DSP")
plt.plot(man, one_DSP, label = "HP 1 DSP")
plt.legend(bbox_to_anchor=(0.001, 0.999), loc=2, borderaxespad=0., prop={'size':10})
plt.title('Custom-Precision Floating-Point Multiplier\nLUT Utilization With Round-to-Nearest')
plt.grid(linestyle='--')
plt.show()

# single and half precision comparision results. Done in 2017.1 Vivado_hls

# multiplicatino single no DSP
# #=== Post-Implementation Resource usage ===
# CLB:            104
# LUT:            543
# FF:             500
# DSP:              0
# BRAM:             0
# SRL:              0
# #=== Final timing ===
# CP required:    4.000
# CP achieved post-synthesis:    1.957
# CP achieved post-implementation:    2.714
# Timing met

# single mul 1 DSP
# #=== Post-Implementation Resource usage ===
# CLB:             47
# LUT:            224
# FF:             158
# DSP:              1
# BRAM:             0
# SRL:              0
# #=== Final timing ===
# CP required:    4.000
# CP achieved post-synthesis:    2.926
# CP achieved post-implementation:    3.371
# Timing met

# single mul 2(full) DSP
# #=== Post-Implementation Resource usage ===
# CLB:             21
# LUT:             80
# FF:             166
# DSP:              2
# BRAM:             0
# SRL:              0
# #=== Final timing ===
# CP required:    4.000
# CP achieved post-synthesis:    2.442
# CP achieved post-implementation:    2.487
# Timing met

# single mul 3(max) DSP
# #=== Post-Implementation Resource usage ===
# CLB:             18
# LUT:             86
# FF:             119
# DSP:              3
# BRAM:             0
# SRL:              0
# #=== Final timing ===
# CP required:    4.000
# CP achieved post-synthesis:    2.442
# CP achieved post-implementation:    2.507
# Timing met

# 2 DSP
# #=== Post-Implementation Resource usage ===
# CLB:             12
# LUT:             35
# FF:              75
# DSP:              2
# BRAM:             0
# SRL:              0
# #=== Final timing ===
# CP required:    4.000
# CP achieved post-synthesis:    3.425
# CP achieved post-implementation:    3.643
# Timing met

# no DSP
# #=== Post-Implementation Resource usage ===
# CLB:             37
# LUT:            179
# FF:              92
# DSP:              0
# BRAM:             0
# SRL:              0
# #=== Final timing ===
# CP required:    4.000
# CP achieved post-synthesis:    2.668
# CP achieved post-implementation:    2.949
# Timing met

# 1 DSP
# #=== Post-Implementation Resource usage ===
# CLB:             10
# LUT:             46
# FF:              60
# DSP:              1
# BRAM:             0
# SRL:              0
# #=== Final timing ===
# CP required:    4.000
# CP achieved post-synthesis:    2.915
# CP achieved post-implementation:    3.341
# Timing met

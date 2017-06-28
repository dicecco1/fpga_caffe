import subprocess
import os
from shutil import copyfile
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

class report:
	def __init__(self, SLICE, LUT, FF, DSP, BRAM, SRL, CP_required, CP_achieved_post_synthesis, CP_achieved_post_implemetation, EXP, MEN):
		self.SLICE = SLICE
		self.LUT = LUT
		self.FF = FF
		self.DSP = DSP
		self.BRAM = BRAM
		self.SRL = SRL
		self.CP_required = CP_required
		self.CP_achieved_post_synthesis = CP_achieved_post_synthesis
		self.CP_achieved_post_implemetation = CP_achieved_post_implemetation
		self.EXP = EXP
		self.MEN = MEN

# variable initialization
report_list = []
second_report_list = []
third_report_list = []
# exp_4 holds the LUT usage of adder when exponent size of 4 and mantisa from 4 to 20
exp_4 = []
exp_5 = []
exp_6 = []
exp_7 = []
# _1, _2 corresponds to the second and third experiment
exp_4_1 = []
exp_5_1 = []
exp_6_1 = []
exp_7_1 = []
exp_4_2 = []
exp_5_2 = []
exp_6_2 = []
exp_7_2 = []
# average of three experiments
exp_4_avg = []
exp_5_avg = []
exp_6_avg = []
exp_7_avg = []

for exponent in range (4,8):
	for mentisa in range (4,21):
		# all directory for results folder, change accordingly
		directory = '/home/linsun/Desktop/chalfHLS/mulHLS/impl_reports'
		second_directory = '/home/linsun/Desktop/chalfHLS/mulHLS/report1'
		third_directory = '/home/linsun/Desktop/chalfHLS/mulHLS/report2'

		#run in python -O script.py or python script.py
		if __debug__:
			#run with -O flag to disable hls synthesis and graph generation
			#make a copy of the original_para.txt
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

			#change extension
			base = os.path.splitext(thisFile)[0]
			os.rename(thisFile, base + ".hpp")
			f_temp.close()

			#finish generating macro file, execute hls
			subprocess.call(["vivado_hls", "run_hls.tcl"])

			#finish hls, archieve report
			copyfile('/home/linsun/Desktop/chalfHLS/mulHLS/mul_prj/solution1/impl/report/verilog/mul_export.rpt', '/home/linsun/Desktop/chalfHLS/mulHLS/impl_reports/temp_report.rtp')
			for file in os.listdir(directory):
				if file.startswith("temp_report"):
					os.rename(os.path.join(directory, file), os.path.join(directory, 'exp=' + str(exponent) +'men=' + str(mentisa) + '.txt'))
		
		#with -O flag only regenerate graph and report detailed usage in command line
		#extract information from report
		reportname = 'exp=' + str(exponent) + 'men=' + str(mentisa) + '.txt'
		with open(os.path.join(directory, reportname)) as f:
			for line in f:
				data = line.split()
				if 'SLICE' in line:
					SLICE = data[1]
				if 'LUT' in line:
					LUT = float(data[1])
					if exponent == 4:
						exp_4.append(LUT)
					if exponent == 5:
						exp_5.append(LUT)
					if exponent == 6:
						exp_6.append(LUT)
					if exponent == 7:
						exp_7.append(LUT)
				if 'FF' in line:
					FF = data[1]
				if 'DSP' in line:
					DSP = data[1]
				if 'BRAM' in line:
					BRAM = data[1]
				if 'SRL' in line:
					SRL = data[1]
				if 'CP required' in line:
					CP_required = data[2]
				if 'CP achieved post-synthesis' in line:
					CP_achieved_post_synthesis = data[3]
				if 'CP achieved post-implemetation' in line:
					CP_achieved_post_implemetation = data[3]
		report_list.append(report(SLICE, str(LUT), FF, DSP, BRAM, SRL, CP_required, CP_achieved_post_synthesis, CP_achieved_post_implemetation, exponent, mentisa))

		# read in results of the second experiment
		reportname = 'exp=' + str(exponent) + 'men=' + str(mentisa) + '.txt'
		with open(os.path.join(second_directory, reportname)) as f:
			for line in f:
				data = line.split()
				if 'SLICE' in line:
					SLICE = data[1]
				if 'LUT' in line:
					LUT = float(data[1])
					if exponent == 4:
						exp_4_1.append(LUT)
					if exponent == 5:
						exp_5_1.append(LUT)
					if exponent == 6:
						exp_6_1.append(LUT)
					if exponent == 7:
						exp_7_1.append(LUT)
				if 'FF' in line:
					FF = data[1]
				if 'DSP' in line:
					DSP = data[1]
				if 'BRAM' in line:
					BRAM = data[1]
				if 'SRL' in line:
					SRL = data[1]
				if 'CP required' in line:
					CP_required = data[2]
				if 'CP achieved post-synthesis' in line:
					CP_achieved_post_synthesis = data[3]
				if 'CP achieved post-implemetation' in line:
					CP_achieved_post_implemetation = data[3]
		second_report_list.append(report(SLICE, str(LUT), FF, DSP, BRAM, SRL, CP_required, CP_achieved_post_synthesis, CP_achieved_post_implemetation, exponent, mentisa))

		# read in results of the third experiment
		reportname = 'exp=' + str(exponent) + 'men=' + str(mentisa) + '.txt'
		with open(os.path.join(third_directory, reportname)) as f:
			for line in f:
				data = line.split()
				if 'SLICE' in line:
					SLICE = data[1]
				if 'LUT' in line:
					LUT = float(data[1])
					if exponent == 4:
						exp_4_2.append(LUT)
					if exponent == 5:
						exp_5_2.append(LUT)
					if exponent == 6:
						exp_6_2.append(LUT)
					if exponent == 7:
						exp_7_2.append(LUT)
				if 'FF' in line:
					FF = data[1]
				if 'DSP' in line:
					DSP = data[1]
				if 'BRAM' in line:
					BRAM = data[1]
				if 'SRL' in line:
					SRL = data[1]
				if 'CP required' in line:
					CP_required = data[2]
				if 'CP achieved post-synthesis' in line:
					CP_achieved_post_synthesis = data[3]
				if 'CP achieved post-implemetation' in line:
					CP_achieved_post_implemetation = data[3]
		third_report_list.append(report(SLICE, str(LUT), FF, DSP, BRAM, SRL, CP_required, CP_achieved_post_synthesis, CP_achieved_post_implemetation, exponent, mentisa))

# print out the summary of all usages of the first test
print '-----------------------First Experiment-----------------------'
for report_list_iter in range (0, len(report_list)):
	print 'Exp = '+ str(report_list[report_list_iter].EXP) + ' Men = ' + str(report_list[report_list_iter].MEN) + '\n' + 'SLICE LUT  FF  DSP BRAM SRL CP    CP_Synthesis CP_Implemetation\n' + report_list[report_list_iter].SLICE + '   ' + report_list[report_list_iter].LUT + '  ' + report_list[report_list_iter].FF + ' ' + report_list[report_list_iter].DSP + '   ' + report_list[report_list_iter].BRAM + '    ' + report_list[report_list_iter].SRL + '  ' + report_list[report_list_iter].CP_required + ' ' + report_list[report_list_iter].CP_achieved_post_synthesis + '        ' + report_list[report_list_iter].CP_achieved_post_implemetation
# print out the summary of all usages of the second test
print '-----------------------Second Experiment-----------------------'
for report_list_iter in range (0, len(second_report_list)):
	print 'Exp = '+ str(second_report_list[report_list_iter].EXP) + ' Men = ' + str(second_report_list[report_list_iter].MEN) + '\n' + 'SLICE LUT  FF  DSP BRAM SRL CP    CP_Synthesis CP_Implemetation\n' + second_report_list[report_list_iter].SLICE + '   ' + second_report_list[report_list_iter].LUT + '  ' + second_report_list[report_list_iter].FF + ' ' + second_report_list[report_list_iter].DSP + '   ' + second_report_list[report_list_iter].BRAM + '    ' + second_report_list[report_list_iter].SRL + '  ' + second_report_list[report_list_iter].CP_required + ' ' + second_report_list[report_list_iter].CP_achieved_post_synthesis + '        ' + second_report_list[report_list_iter].CP_achieved_post_implemetation
# print out the summary of all usages of the third test
print '-----------------------Third Experiment-----------------------'
for report_list_iter in range (0, len(third_report_list)):
	print 'Exp = '+ str(third_report_list[report_list_iter].EXP) + ' Men = ' + str(third_report_list[report_list_iter].MEN) + '\n' + 'SLICE LUT  FF  DSP BRAM SRL CP    CP_Synthesis CP_Implemetation\n' + third_report_list[report_list_iter].SLICE + '   ' + third_report_list[report_list_iter].LUT + '  ' + third_report_list[report_list_iter].FF + ' ' + third_report_list[report_list_iter].DSP + '   ' + third_report_list[report_list_iter].BRAM + '    ' + third_report_list[report_list_iter].SRL + '  ' + third_report_list[report_list_iter].CP_required + ' ' + third_report_list[report_list_iter].CP_achieved_post_synthesis + '        ' + third_report_list[report_list_iter].CP_achieved_post_implemetation

# compute average value of three experiments
i=0
while i < len(exp_4):
	exp_4_avg.append((exp_4[i]+exp_4_1[i]+exp_4_2[i])/3)
	exp_5_avg.append((exp_5[i]+exp_5_1[i]+exp_5_2[i])/3)
	exp_6_avg.append((exp_6[i]+exp_6_1[i]+exp_6_2[i])/3)
	exp_7_avg.append((exp_7[i]+exp_7_1[i]+exp_7_2[i])/3)
	i+=1

#plot the data
man = range(4,21)
# refer to the end of the code, single precision adder HLS result
single_LUTs = [78] * 17

# line graph position in 3D space
report_position = [7] * 17
second_report_position = [5] * 17
third_report_position = [3] * 17
avg_report_position = [1] * 17

#initialize figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_ylim([1,8])
ax.set_color_cycle(['blue','red','cyan','green','yellow'])
ax.plot(man, report_position, zs=exp_4, label='EXP4')
ax.plot(man, report_position, zs=exp_5, label='EXP5')
ax.plot(man, report_position, zs=exp_6, label='EXP6')
ax.plot(man, report_position, zs=exp_7, label='EXP7')
ax.plot(man, report_position, zs=single_LUTs, label='Single')
ax.set_color_cycle(['blue','red','cyan','green','yellow'])
ax.plot(man, second_report_position, zs=exp_4_1)
ax.plot(man, second_report_position, zs=exp_5_1)
ax.plot(man, second_report_position, zs=exp_6_1)
ax.plot(man, second_report_position, zs=exp_7_1)
ax.plot(man, second_report_position, zs=single_LUTs)
ax.set_color_cycle(['blue','red','cyan','green','yellow'])
ax.plot(man, third_report_position, zs=exp_4_2)
ax.plot(man, third_report_position, zs=exp_5_2)
ax.plot(man, third_report_position, zs=exp_6_2)
ax.plot(man, third_report_position, zs=exp_7_2)
ax.plot(man, third_report_position, zs=single_LUTs)
ax.set_color_cycle(['blue','red','cyan','green','yellow'])
ax.plot(man, avg_report_position, zs=exp_4_avg)
ax.plot(man, avg_report_position, zs=exp_5_avg)
ax.plot(man, avg_report_position, zs=exp_6_avg)
ax.plot(man, avg_report_position, zs=exp_7_avg)
ax.plot(man, avg_report_position, zs=single_LUTs)
ax.set_xlabel('Mantisa')
ax.set_ylabel('Report')
ax.set_zlabel('LUT')
ax.set_title('LUT Adder Round')
ax.text(12,6.6,85,'first')
ax.text(12,4.3,85,'Second')
ax.text(12,2.5,85,'Third')
ax.text(12,0.25,85,'Average')
ax.legend()
plt.show()

#=== Post-Implementation Resource usage ===
# SLICE:           47
# LUT:             78
# FF:             185
# DSP:              3
# BRAM:             0
# SRL:              0
# #=== Final timing ===
# CP required:    4.000
# CP achieved post-synthesis:    2.872
# CP achieved post-implemetation:    3.131
# Timing met
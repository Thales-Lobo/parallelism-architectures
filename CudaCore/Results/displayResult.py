import matplotlib.pyplot as plt
import json

with open('resultsData.json', 'r') as results:
    data = json.load(results)

question3 = data['question3']
question4 = data['question4']
question3_COALESCENT = data['question3_COALESCENT']
question4_COALESCENT = data['question4_COALESCENT']


#! General Data
x = [1024, 2048, 4096]
#Question3 Data
y3_matrix_time = [
    question3['1024']['completeMatrixProduct']['elapsedTime'],
    question3['2048']['completeMatrixProduct']['elapsedTime'],
    question3['4096']['completeMatrixProduct']['elapsedTime']
]
y3_matrix_gflops = [
    question3['1024']['completeMatrixProduct']['Gflops'],
    question3['2048']['completeMatrixProduct']['Gflops'],
    question3['4096']['completeMatrixProduct']['Gflops']
]
y3_kernel_time = [
    question3['1024']['kernelComputation']['elapsedTime'],
    question3['2048']['kernelComputation']['elapsedTime'],
    question3['4096']['kernelComputation']['elapsedTime']
]
y3_kernel_gflops = [
    question3['1024']['kernelComputation']['Gflops'],
    question3['2048']['kernelComputation']['Gflops'],
    question3['4096']['kernelComputation']['Gflops']
]

#Question4 Data
y4_matrix_time = [
    question4['1024']['completeMatrixProduct']['elapsedTime'],
    question4['2048']['completeMatrixProduct']['elapsedTime'],
    question4['4096']['completeMatrixProduct']['elapsedTime']
]
y4_matrix_gflops = [
    question4['1024']['completeMatrixProduct']['Gflops'],
    question4['2048']['completeMatrixProduct']['Gflops'],
    question4['4096']['completeMatrixProduct']['Gflops']
]
y4_kernel_time = [
    question4['1024']['kernelComputation']['elapsedTime'],
    question4['2048']['kernelComputation']['elapsedTime'],
    question4['4096']['kernelComputation']['elapsedTime']
]
y4_kernel_gflops = [
    question4['1024']['kernelComputation']['Gflops'],
    question4['2048']['kernelComputation']['Gflops'],
    question4['4096']['kernelComputation']['Gflops']
]

#Question3_COALESCENT Data
y3_matrix_time_COALESCENT = [
    question3_COALESCENT['1024']['completeMatrixProduct']['elapsedTime'],
    question3_COALESCENT['2048']['completeMatrixProduct']['elapsedTime'],
    question3_COALESCENT['4096']['completeMatrixProduct']['elapsedTime']
]
y3_matrix_gflops_COALESCENT = [
    question3_COALESCENT['1024']['completeMatrixProduct']['Gflops'],
    question3_COALESCENT['2048']['completeMatrixProduct']['Gflops'],
    question3_COALESCENT['4096']['completeMatrixProduct']['Gflops']
]
y3_kernel_time_COALESCENT = [
    question3_COALESCENT['1024']['kernelComputation']['elapsedTime'],
    question3_COALESCENT['2048']['kernelComputation']['elapsedTime'],
    question3_COALESCENT['4096']['kernelComputation']['elapsedTime']
]
y3_kernel_gflops_COALESCENT = [
    question3_COALESCENT['1024']['kernelComputation']['Gflops'],
    question3_COALESCENT['2048']['kernelComputation']['Gflops'],
    question3_COALESCENT['4096']['kernelComputation']['Gflops']
]

#Question4_COALESCENT Data
y4_matrix_time_COALESCENT = [
    question4_COALESCENT['1024']['completeMatrixProduct']['elapsedTime'],
    question4_COALESCENT['2048']['completeMatrixProduct']['elapsedTime'],
    question4_COALESCENT['4096']['completeMatrixProduct']['elapsedTime']
]
y4_matrix_gflops_COALESCENT = [
    question4_COALESCENT['1024']['completeMatrixProduct']['Gflops'],
    question4_COALESCENT['2048']['completeMatrixProduct']['Gflops'],
    question4_COALESCENT['4096']['completeMatrixProduct']['Gflops']
]
y4_kernel_time_COALESCENT = [
    question4_COALESCENT['1024']['kernelComputation']['elapsedTime'],
    question4_COALESCENT['2048']['kernelComputation']['elapsedTime'],
    question4_COALESCENT['4096']['kernelComputation']['elapsedTime']
]
y4_kernel_gflops_COALESCENT = [
    question4_COALESCENT['1024']['kernelComputation']['Gflops'],
    question4_COALESCENT['2048']['kernelComputation']['Gflops'],
    question4_COALESCENT['4096']['kernelComputation']['Gflops']
]

##Question4: SpeedUp (Acceleration)
y_speedUp = [
    y3_matrix_time[i]/y4_matrix_time[i] for i in range(len(y3_matrix_time))
]
y_speedUp_COALESCENT = [
    y3_matrix_time_COALESCENT[i]/y4_matrix_time_COALESCENT[i] for i in range(len(y3_matrix_time_COALESCENT))
]

# Graphics 
ALPHA = 0.25 #25% of transparency
fig_q3_time, ax_q3_time = plt.subplots()
fig_q3_gflops, ax_q3_gflops= plt.subplots()
fig_q4_time, ax_q4_time = plt.subplots()
fig_q4_gflops, ax_q4_gflops = plt.subplots()
fig_speedUp, ax_speedUp = plt.subplots()

#Axes configurations
ax_q3_time.plot(x, y3_matrix_time_COALESCENT, label='Complete Matrix Product', color='orange')
ax_q3_time.plot(x, y3_kernel_time_COALESCENT, label='Kernel Computation', color='blue')
ax_q3_time.plot(x, y3_matrix_time, label='(Non Coalescent)Complete Matrix Product', color='orange', linestyle='--', alpha=ALPHA)
ax_q3_time.plot(x, y3_kernel_time, label='(Non Coalescent)Kernel Computation', color='blue', linestyle='--', alpha=ALPHA)

ax_q3_gflops.plot(x, y3_matrix_gflops_COALESCENT, label='Complete Matrix Product', color='red')
ax_q3_gflops.plot(x, y3_kernel_gflops_COALESCENT, label='Kernel Computation', color='green')
ax_q3_gflops.plot(x, y3_matrix_gflops, label='(Non Coalescent)Complete Matrix Product', color='red', linestyle='--', alpha=ALPHA)
ax_q3_gflops.plot(x, y3_kernel_gflops, label='(Non Coalescent)Kernel Computation', color='green', linestyle='--', alpha=ALPHA)

ax_q4_time.plot(x, y4_matrix_time_COALESCENT, label='Complete Matrix Product', color='orange')
ax_q4_time.plot(x, y4_kernel_time_COALESCENT, label='Complete Kernel Computation', color='blue')
ax_q4_time.plot(x, y4_matrix_time, label='(Non Coalescent)Complete Matrix Product', color='orange', linestyle='--', alpha=ALPHA)
ax_q4_time.plot(x, y4_kernel_time, label='(Non Coalescent)Complete Kernel Computation', color='blue', linestyle='--', alpha=ALPHA)

ax_q4_gflops.plot(x, y4_matrix_gflops_COALESCENT, label='Complete Matrix Product', color='red')
ax_q4_gflops.plot(x, y4_kernel_gflops_COALESCENT, label='Kernel Computation', color='green')
ax_q4_gflops.plot(x, y4_matrix_gflops, label='(Non Coalescent)Complete Matrix Product', color='red', linestyle='--', alpha=ALPHA)
ax_q4_gflops.plot(x, y4_kernel_gflops, label='(Non Coalescent)Kernel Computation', color='green', linestyle='--', alpha=ALPHA)

ax_speedUp.plot(x, y_speedUp_COALESCENT, label='SpeedUp', color='purple')
ax_speedUp.plot(x, y_speedUp, label='(Non Coalescent)SpeedUp', color='purple', linestyle='--', alpha=ALPHA)

# Legends
ax_q3_time.legend()
ax_q3_gflops.legend()
ax_q4_time.legend()
ax_q4_gflops.legend()
ax_speedUp.legend()

# Titles 
ax_q3_time.set_title('Performance: Elapsed Time[s] x Size')
ax_q3_gflops.set_title('Performance: Gflops[Gb/s] x Size')
ax_q4_time.set_title('Performance: Elapsed Time[s] x Size')
ax_q4_gflops.set_title('Performance: Gflops[Gb/s] x Size')
ax_speedUp.set_title('Performance comparison: Kernel V2 over Kernel V1')

# X axes labels
for x in (ax_q3_time, ax_q3_gflops, ax_q4_time, ax_q4_gflops, ax_speedUp):
    x.set_xlabel('SIZE')

# Y axes labels  
ax_q3_time.set_ylabel('Execution Time[s]')
ax_q4_time.set_ylabel('Execution Time[s]')
ax_q3_gflops.set_ylabel('Gflops[Gb/s]')
ax_q4_gflops.set_ylabel('Gflops[Gb/s]')
ax_speedUp.set_ylabel('SpeedUp')

# Output directory
question3Directory = "./ImagesQuestion3/"
question4Directory = "./ImagesQuestion4/"

# Resolution
dpi = 1000

# Save images
fig_q3_time.savefig(question3Directory + "PerformanceKernelV1-Time.png", dpi=dpi)
fig_q3_gflops.savefig(question3Directory + "PerformanceKernelV1-Gflops.png", dpi=dpi)
fig_q4_time.savefig(question4Directory + "PerformanceKernelV2-Time.png", dpi=dpi)
fig_q4_gflops.savefig(question4Directory + "PerformanceKernelV2-Gflops.png", dpi=dpi)
fig_speedUp.savefig(question4Directory + "SpeedUp.png", dpi=dpi)

# Show images
plt.show()

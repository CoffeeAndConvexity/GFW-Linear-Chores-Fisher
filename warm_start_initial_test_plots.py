import numpy as np
import pandas as pd

data_without_ws = pd.read_csv('uniform.csv', index_col=0)
data_with_ws = pd.read_csv('uniform_warm_start.csv', index_col=0)

import matplotlib.pyplot as plt

print(data_without_ws)
print(data_with_ws)


# Plotting iteration numbers vs problem sizes
plt.figure(figsize=(10, 6))
plt.plot(data_without_ws['size'], data_without_ws['iteration_GFW_e'], marker='o', label='GFW w/o Warm Start', color='blue')
plt.plot(data_with_ws['size'], data_with_ws['iteration_GFW_e'], marker='o', label='GFW with Warm Start', color='black')
plt.plot(data_without_ws['size'], data_without_ws['iteration_GFW_a1'], marker='o', color='blue', linestyle='--')
plt.plot(data_with_ws['size'], data_with_ws['iteration_GFW_a1'], marker='o', color='black', linestyle='--')
plt.plot(data_without_ws['size'], data_without_ws['iteration_EPM_e'], marker='^', label='EPM w/o Warm Start', color='orange')
plt.plot(data_with_ws['size'], data_with_ws['iteration_EPM_e'], marker='^', label='EPM with Warm Start', color='brown')
plt.plot(data_without_ws['size'], data_without_ws['iteration_EPM_a1'], marker='^', color='orange', linestyle='--')
plt.plot(data_with_ws['size'], data_with_ws['iteration_EPM_a1'], marker='^', color='brown', linestyle='--')

plt.xlabel('Problem Size (N)')
plt.ylabel('Number of Iterations')
plt.title('Iteration Number vs Problem Size')
plt.legend()
plt.grid(True)
plt.savefig('uniform_iterations_warm_start_comparison.png')


# Plotting computation time vs problem sizes
plt.figure(figsize=(10, 6))
plt.plot(data_without_ws['size'], data_without_ws['runningtime_GFW_e'], marker='o', label='GFW w/o Warm Start', color='blue')
plt.plot(data_with_ws['size'], data_with_ws['runningtime_GFW_e'], marker='o', label='GFW with Warm Start', color='black')
plt.plot(data_without_ws['size'], data_without_ws['runningtime_GFW_a1'], marker='o', color='blue', linestyle='--')
plt.plot(data_with_ws['size'], data_with_ws['runningtime_GFW_a1'], marker='o', color='black', linestyle='--')
plt.plot(data_without_ws['size'], data_without_ws['runningtime_EPM_e'], marker='^', label='EPM w/o Warm Start', color='orange')
plt.plot(data_with_ws['size'], data_with_ws['runningtime_EPM_e'], marker='^', label='EPM with Warm Start', color='brown')
plt.plot(data_without_ws['size'], data_without_ws['runningtime_EPM_a1'], marker='^', color='orange', linestyle='--')
plt.plot(data_with_ws['size'], data_with_ws['runningtime_EPM_a1'], marker='^', color='brown', linestyle='--')

plt.xlabel('Problem Size (N)')
plt.ylabel('Computation Time (seconds)')
plt.title('Computation Time vs Problem Size')
plt.legend()
plt.grid(True)
plt.savefig('uniform_time_warm_start_comparison.png')


# Plotting solved instance ratio vs problem sizes
plt.figure(figsize=(10, 6))
plt.plot(data_without_ws['size'], data_without_ws['solved_GFW_e'], marker='o', label='GFW w/o Warm Start', color='blue')
plt.plot(data_with_ws['size'], data_with_ws['solved_GFW_e'], marker='o', label='GFW with Warm Start', color='black')
plt.plot(data_without_ws['size'], data_without_ws['solved_GFW_a1'], marker='o', color='blue', linestyle='--')
plt.plot(data_with_ws['size'], data_with_ws['solved_GFW_a1'], marker='o', color='black', linestyle='--')
plt.plot(data_without_ws['size'], data_without_ws['solved_EPM_e'], marker='^', label='EPM w/o Warm Start', color='orange')
plt.plot(data_with_ws['size'], data_with_ws['solved_EPM_e'], marker='^', label='EPM with Warm Start', color='brown')
plt.plot(data_without_ws['size'], data_without_ws['solved_EPM_a1'], marker='^', color='orange', linestyle='--')
plt.plot(data_with_ws['size'], data_with_ws['solved_EPM_a1'], marker='^', color='brown', linestyle='--')

plt.xlabel('Problem Size (N)')
plt.ylabel('Solved Instance Ratio')
plt.title('Solved Instance Ratio vs Problem Size')
plt.legend()
plt.grid(True)
plt.savefig('uniform_solved_ratio_warm_start_comparison.png')

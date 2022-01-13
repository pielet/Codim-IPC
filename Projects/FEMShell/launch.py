import subprocess

# Intel MKL number of threads
numThreads = '16'

# 1_swing_cloth, 2_rotate_cloth
script = ['1_swing_cloth.py']

opt_param = ["force"]
constrain_type = ["soft"]
opt_med = ["L-BFGS"]

for i in range(len(script)):
	for j in range(len(opt_param)):
		cmd = f"""export OMP_PROC_BIND=spreadexport OMP_PLACES=threads
				export MKL_NUM_THREADS={numThreads}
				export OMP_NUM_THREADS={numThreads}
				export VECLIB_MAXIMUM_THREADS={numThreads}
				python3 {script[i]} {opt_param[j]} {constrain_type[j]} {opt_med[j]}"""
		if subprocess.call([cmd], shell=True):
			continue

# script = ['1_swing_cloth.py']

# opt_param = ["trajectory"]
# constrain_type = ["hard"]
# opt_med = ["GN"]
# init_med = ["fix"]

# for i in range(len(script)):
# 	for j in range(len(opt_param)):
# 		cmd = f"""export OMP_PROC_BIND=spreadexport OMP_PLACES=threads
# 				export MKL_NUM_THREADS={numThreads}
# 				export OMP_NUM_THREADS={numThreads}
# 				export VECLIB_MAXIMUM_THREADS={numThreads}
# 				python3 {script[i]} {opt_param[j]} {constrain_type[j]} {opt_med[j]} {init_med[j]}
# 				python3 loss_vis.py {script[i]} {opt_param[j]} {constrain_type[j]} {opt_med[j]} {init_med[j]}"""
# 		if subprocess.call([cmd], shell=True):
# 			continue

# vis
# for i in range(len(script)):
# 	for j in range(len(opt_param)):
# 		cmd = f"python3 loss_vis.py {script[i]} {opt_param[j]} {constrain_type[j]} {opt_med[j]} {init_med[j]}"
# 		if subprocess.call([cmd], shell=True):
# 			continue
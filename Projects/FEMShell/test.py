import subprocess

# Intel MKL number of threads
numThreads = '16'

# 1_swing_cloth, 2_rotate_cloth
script = ['1_cloth.py']

# opt_param = ["force"]
# constrain_type = ["hard"]
# opt_med = ["FP"]

# opt_param = ["force"]
# constrain_type = ["soft"]
# opt_med = ["GD"]

# opt_param = ["force", "force", "force"]
# constrain_type = ["soft", "soft", "hard"]
# opt_med = ["GD", "L-BFGS", "FP"]

# for i in range(len(script)):
# 	for j in range(len(opt_param)):
# 		cmd = f"""export OMP_PROC_BIND=spread
# 				export OMP_PLACES=threads
# 				export MKL_NUM_THREADS={numThreads}
# 				export OMP_NUM_THREADS={numThreads}
# 				export VECLIB_MAXIMUM_THREADS={numThreads}
# 				python3 {script[i]} {opt_param[j]} {constrain_type[j]} {opt_med[j]}"""
# 		if subprocess.call([cmd], shell=True):
# 			continue

opt_param = ["trajectory", "trajectory"]
constrain_type = ["soft"]
opt_med = ["GN"]
init_med = ["solve"]

for i in range(len(init_med)):
		cmd = f"""export OMP_PROC_BIND=spread
				export OMP_PLACES=threads
				export MKL_NUM_THREADS={numThreads}
				export OMP_NUM_THREADS={numThreads}
				export VECLIB_MAXIMUM_THREADS={numThreads}
				export CHOLMOD_USE_GPU=1
				python3 {script[0]} trajectory {constrain_type[i]} {opt_med[i]} {init_med[i]}"""
		if subprocess.call([cmd], shell=True):
			continue

opt_param = ["trajectory"]
constrain_type = ["soft"]
opt_med = ["GN"]
init_med = ["solve"]
epsilon = [1, 1e-2, 1e-4]

for i in range(len(epsilon)):
		cmd = f"""export OMP_PROC_BIND=spread
				export OMP_PLACES=threads
				export MKL_NUM_THREADS={numThreads}
				export OMP_NUM_THREADS={numThreads}
				export VECLIB_MAXIMUM_THREADS={numThreads}
				export CHOLMOD_USE_GPU=1
				python3 {script[0]} trajectory {constrain_type[0]} {opt_med[0]} {init_med[0]} {epsilon[i]}"""
		if subprocess.call([cmd], shell=True):
			continue

# cmd = f"""export OMP_PROC_BIND=spreadexport OMP_PLACES=threads
# 	export MKL_NUM_THREADS={numThreads}
# 	export OMP_NUM_THREADS={numThreads}
# 	export VECLIB_MAXIMUM_THREADS={numThreads}
# 	python3 1_swing_cloth.py alternate"""
# subprocess.call([cmd], shell=True)

# vis
# for i in range(len(script)):
# 	for j in range(len(opt_param)):
# 		cmd = f"python3 loss_vis.py {script[i]} {opt_param[j]} {constrain_type[j]} {opt_med[j]} {init_med[j]}"
# 		if subprocess.call([cmd], shell=True):
# 			continue
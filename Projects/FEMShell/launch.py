import subprocess

# Intel MKL number of threads
numThreads = '16'

# 1_swing_cloth, 2_rotate_cloth
script = ['4_flag.py']

opt_param = ["force"]
constrain_type = ["hard"]
opt_med = ["FP"]

# opt_param = ["force"]
# constrain_type = ["soft"]
# opt_med = ["GD"]

# opt_param = ["force", "force"]
# constrain_type = ["soft", "soft"]
# opt_med = ["GD", "L-BFGS"]

# for i in range(len(script)):
# 	for j in range(len(opt_param)):
# 		cmd = f"""export OMP_PROC_BIND=spreadexport OMP_PLACES=threads
# 				export MKL_NUM_THREADS={numThreads}
# 				export OMP_NUM_THREADS={numThreads}
# 				export VECLIB_MAXIMUM_THREADS={numThreads}
# 				python3 {script[i]} {opt_param[j]} {constrain_type[j]} {opt_med[j]}"""
# 		if subprocess.call([cmd], shell=True):
# 			continue

opt_param = ["trajectory", "trajectory", "trajectory", "trajectory"]
constrain_type = ["hard", "soft", "hard", "soft"]
opt_med = ["GN", "GN", "GN", "GN"]
init_med = ["solve", "solve", "load", "load"]

for i in range(len(init_med)):
		cmd = f"""export OMP_PROC_BIND=spreadexport OMP_PLACES=threads
				export MKL_NUM_THREADS={numThreads}
				export OMP_NUM_THREADS={numThreads}
				export VECLIB_MAXIMUM_THREADS={numThreads}
				python3 {script[0]} trajectory {constrain_type[i]} {opt_med[i]} {init_med[i]}
				python3 loss_vis.py {script[0]} trajectory {constrain_type[i]} {opt_med[i]} {init_med[i]}"""
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
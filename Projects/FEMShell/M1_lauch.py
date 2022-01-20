import subprocess

numThreads = '16'

# 1_swing_cloth, 2_rotate_cloth
script = ['3_bunny.py']

opt_param = ["force", "force"]
constrain_type = ["soft", "hard"]
opt_med = ["L-BFGS", "FP"]

for i in range(len(script)):
	for j in range(len(opt_param)):
		cmd = f"""export OMP_PROC_BIND=spread
				export OMP_PLACES=threads
				export MKL_NUM_THREADS={numThreads}
				export OMP_NUM_THREADS={numThreads}
				export VECLIB_MAXIMUM_THREADS={numThreads}
				python3 {script[i]} {opt_param[j]} {constrain_type[j]} {opt_med[j]}"""
		if subprocess.call([cmd], shell=True):
			continue
import subprocess

# Intel MKL number of threads
numThreads = '16'

script = '1_swing_cloth.py'

cmd = f"""export OMP_PROC_BIND=spreadexport OMP_PLACES=threads
	export MKL_NUM_THREADS={numThreads}
	export OMP_NUM_THREADS={numThreads}
	export VECLIB_MAXIMUM_THREADS={numThreads}
	python3 {script}
"""

subprocess.call([cmd], shell=True)
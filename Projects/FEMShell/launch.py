import subprocess

# Intel MKL number of threads
numThreads = '16'

algI = '0'
script = '1_swing_cloth.py'
clothI = '0'
garment = '21'

command = f"export OMP_PROC_BIND=spread\nexport OMP_PLACES=threads\nexport MKL_NUM_THREADS={numThreads}\nexport OMP_NUM_THREADS={numThreads}\nexport VECLIB_MAXIMUM_THREADS={numThreads}\npython3 {script} {algI} {clothI} {garment}\n"
subprocess.call([command], shell=True)
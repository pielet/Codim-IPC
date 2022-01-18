import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

loss_path = "output/" + os.path.splitext(os.path.basename(sys.argv[1]))[0] + "/"
if len(sys.argv) > 2:
	loss_path += sys.argv[2]
	for i in range(3, len(sys.argv)):
		loss_path += "_" + sys.argv[i]
	loss_path += "/"


with open(loss_path + "loss.txt", 'r') as f:
	lines = list(f.readlines())

n_epoch = len(lines) // 3
loss_per_frame = []
for i in range(n_epoch):
	loss_per_frame.append([float(x) for x in lines[3 * i + 2].split()])
	n_frame = len(loss_per_frame[-1])

loss_per_frame = np.array(loss_per_frame)
loss = np.sum(loss_per_frame, axis=1)
assert(loss.shape[0] == n_epoch)

plt.plot(np.log(loss))
plt.savefig(loss_path + "loss.png")
plt.close()

fig, axis = plt.subplots(1, 1)
   
axis.set_xlim(-1, n_frame)
axis.set_ylim(0, loss_per_frame[0].max() / n_frame / 10)
data = axis.plot(np.arange(n_frame))[0]

def update(i):
	data.set_ydata(loss_per_frame[i])
	axis.set_title(f"epoch_{i}")
	# axis.set_ylim(loss_per_frame[i].min(), loss_per_frame[i].max())

total_time = 5
loss_ani = FuncAnimation(fig, update, frames=n_epoch, interval=(total_time * 1000 / n_epoch))
loss_ani.save(loss_path + "loss_per_frame.gif")
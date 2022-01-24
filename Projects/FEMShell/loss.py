import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

exp = sys.argv[1]
med = sys.argv[2]

loss_path = "output/" + exp + '/' + med + '/'


with open(loss_path + "loss.txt", 'r') as f:
	lines = list(f.readlines())

n_epoch = len(lines) // 2
loss = []
loss_per_frame = []
for i in range(n_epoch):
	loss.append([float(x) for x in lines[2 * i].split()])
	loss_per_frame.append([float(x) for x in lines[2 * i + 1].split()])
n_frame = len(loss_per_frame[-1])

loss_per_frame = np.array(loss_per_frame)
loss = np.array(loss)

if med.split('_')[0] == "force":
	min_con_i = np.argmin(loss[:, -2])
	min_loopy_i = np.argmin(loss[:, -1])

	print(f"best constrain epoch {min_con_i - 1}: loss {loss[min_con_i, 1]}, force {loss[min_con_i, 2]}, constrain {loss[min_con_i, 3]}, loopy {loss[min_con_i, 4]}")
	print(f"best loopy epoch {min_loopy_i - 1}: loss {loss[min_loopy_i, 1]}, force {loss[min_loopy_i, 2]}, constrain {loss[min_loopy_i, 3]}, loopy {loss[min_loopy_i, 4]}")

	plt.plot(loss[:, 1])
	plt.savefig(loss_path + "loss.png")
	plt.close()

	plt.plot(loss[:, 2])
	plt.savefig(loss_path + "force.png")
	plt.close()

	plt.plot(loss[:, 3])
	plt.savefig(loss_path + "constrain.png")
	plt.close()

	plt.plot(loss[:, 4])
	plt.savefig(loss_path + "loopy.png")
	plt.close()

else:
	plt.plot(np.log(loss[:, 2]))
	plt.savefig(loss_path + "loss.png")
	plt.close()

fig, axis = plt.subplots(1, 1)
   
axis.set_xlim(-1, n_frame)
axis.set_ylim(0, loss_per_frame[-1].max())
data = axis.plot(np.arange(n_frame))[0]

def update(i):
	data.set_ydata(loss_per_frame[i])
	axis.set_title(f"epoch_{i}")
	# axis.set_ylim(loss_per_frame[i].min(), loss_per_frame[i].max())

total_time = 5
loss_ani = FuncAnimation(fig, update, frames=n_epoch, interval=(total_time * 1000 / n_epoch))
loss_ani.save(loss_path + "loss_per_frame.gif")
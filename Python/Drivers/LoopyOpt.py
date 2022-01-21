import sys
sys.path.insert(0, "../../build")
import os
try:
	os.mkdir("output")
except OSError:
	pass

def make_directory(folder):
	try:
		os.mkdir(folder)
	except OSError:
		pass

import re
import math
import time
import numpy as np
from collections import deque
from JGSL import *

class LoopyOpt:
	def __init__(self, sim, opt_param="trajecoty", constrain_form="hard", opt_med="GD") -> None:
		"""
		sim: simulation base
		param_med: force, trajectory
		opt_med: GD (gradient descent_, L-BFGS, GN (Gauss-Newton), and FP (fast projection) (force-based only)
		"""
		self.b_debug = True
		self.sim = sim
		self.opt_param = opt_param
		self.constrain_form = constrain_form
		self.opt_med = opt_med

		# path
		self.output_path = self.sim.output_folder

		# simulation params
		self.dt = self.sim.dt
		self.n_frame = self.sim.frame_num
		self.n_vert = self.sim.n_vert
		self.n_DBC = self.sim.n_DBC
		self.M = self.sim.massMatrix
		
		# optimization params
		self.n_epoch = 100
		self.n_force_epoch = 0
		self.n_trajectory_epoch = 0
		self.n_alternate = 0
		self.epsilon = 1 # soft contrain
		self.L = 0.0 # hard constrain
		self.p = 2  # p-norm
		self.minmax = False
		self.init_med = "fix"  # for trajectory only: solve (0 loopy loss), fix (0 constrain loss)
		self.load_path = ""
		self.use_cg = False  # for trajectory only, direct or pcg
		self.cg_iter = 1000
		self.cg_tol = 0.1
		self.cg_regu = 1e-6

		# storage
		self.X0 = Storage.V3dStorage()
		self.X1 = Storage.V3dStorage()
		self.init_DBC = Storage.V4dStorage()
		self.one_frame = Storage.V3dStorage()
		self.trajectory = Storage.V3dStorage()
		self.loopy_trajectory = Storage.V3dStorage()
		self.control_force = Storage.V3dStorage()
		# line-search
		self.alpha = 1.0
		self.gradient = Storage.V3dStorage()
		self.descent_dir = Storage.V3dStorage()
		self.tentative = Storage.V3dStorage()
		# L-BFGS
		self.window_size = 5
		self.cur_window_size = 0
		self.last_x = Storage.V3dStorage()
		self.last_g = Storage.V3dStorage()
		self.delta_x_history = deque([Storage.V3dStorage() for _ in range(self.window_size)])
		self.delta_g_history = deque([Storage.V3dStorage() for _ in range(self.window_size)])
		self.pho = deque([0.0 for _ in range(self.window_size)])
		# debug
		self.loss_residual = Storage.V3dStorage()
		self.residual = Storage.V3dStorage()
		self.hess_residual = Storage.V3dStorage()
		self.A = CSR_MATRIX_D()

		# loss
		self.loss = 0.0
		self.force_loss = 0.0
		self.constrain_loss = 0.0
		self.valid_per_frame = StdVectorXi()
		self.loss_per_frame = StdVectorXd()
		self.valid = True
		self.loopy_loss = 0.0

		# output
		self.output_folder = "output/" + os.path.splitext(os.path.basename(sys.argv[0]))[0] + "/"
		make_directory(self.output_folder)
		if len(sys.argv) > 1:
			self.output_folder += sys.argv[1]
			for i in range(2, len(sys.argv)):
				self.output_folder += "_" + sys.argv[i]
			self.output_folder += "/"
		make_directory(self.output_folder)

		self.register_logger()

	def initialize(self):
		print(f"=============================== initialize ==============================")
		# init data space
		Control.Fill(self.X0, self.n_vert)
		Control.Fill(self.X1, self.n_vert)
		Control.Fill(self.one_frame, self.n_vert)
		Control.Fill(self.control_force, self.n_vert * self.n_frame)
		Control.Fill(self.trajectory, self.n_vert * self.n_frame)
		Control.Fill(self.loopy_trajectory, self.n_vert * self.n_frame)

		Control.Fill(self.gradient, self.n_vert * self.n_frame)
		Control.Fill(self.descent_dir, self.n_vert * self.n_frame)
		Control.Fill(self.tentative, self.n_vert * self.n_frame)

		Control.Fill(self.init_DBC, self.n_DBC)

		Control.Fill(self.residual, self.n_frame * self.n_vert)
		Control.Fill(self.hess_residual, self.n_frame * self.n_vert)
		Control.Fill(self.loss_residual, self.n_frame * self.n_vert)

		if self.opt_med == "L-BFGS":
			Control.Fill(self.last_x, self.n_frame * self.n_vert)
			Control.Fill(self.last_g, self.n_frame * self.n_vert)
			for i in range(self.window_size):
				Control.Fill(self.delta_x_history[i], self.n_frame * self.n_vert)
				Control.Fill(self.delta_g_history[i], self.n_frame * self.n_vert)

		init_folder = self.output_folder + "init/"
		make_directory(init_folder)
		self.sim.output_folder = init_folder

		# compute initial loss
		Control.Copy(self.sim.X, self.X0)
		Control.Copy(self.sim.DBC, self.init_DBC)
		self.sim.step(0)
		Control.Copy(self.sim.X, self.X1)

		if self.opt_param == "force":
			self.forward(self.control_force)
			Control.Copy(self.trajectory, self.loopy_trajectory)
			Control.SetFrame(self.n_frame - 2, self.loopy_trajectory, self.X0)
			Control.SetFrame(self.n_frame - 1, self.loopy_trajectory, self.X1)
			_, _ = self.compute_trajectory_loss(self.loopy_trajectory, False)
			self.loss = self.compute_force_loss(self.control_force, self.trajectory)
			print(f"[init] loss: {self.loss}, force: {self.force_loss}, constrain: {self.constrain_loss}, loopy: {self.loopy_loss}")
		elif self.opt_param == "trajectory":
			if self.init_med == "load":
				Control.Read(self.trajectory, self.load_path) # read anyway
			else:
				self.forward(self.control_force)
			Control.SetFrame(self.n_frame - 2, self.trajectory, self.X0)
			Control.SetFrame(self.n_frame - 1, self.trajectory, self.X1)

			self.valid, self.loss = self.compute_trajectory_loss(self.trajectory)
			Control.Write(self.loss_residual, init_folder + "/residual.txt")
			print(f"[init] valid: {self.valid}, loss: {self.loss}, loopy: {self.loopy_loss}, constrain: {self.constrain_loss}")
		
		self.output_trajectory(init_folder)
		self.output_loss()

		# TIMER_FLUSH(0, self.n_epoch, 0, self.n_epoch)

	def run(self):
		start = time.time()
		total_epoch = 0
		for i in range(self.n_epoch):
			total_epoch += 1
			epoch_start = time.time()
			if self.one_iter(i):
				print("optimization converged: small alpha ", self.alpha)
				break
			print("epoch time: ", time.time() - epoch_start)
		end = time.time()
		with open(self.output_folder + "loss.txt", 'a') as f:
			f.write(f"total time: {end - start}, avg. time: {(end - start) / total_epoch}")

		if not self.b_debug:
			final_folder = self.output_folder + f"epoch_{total_epoch}/"
			make_directory(final_folder)
			self.output_trajectory(final_folder)

	def alternate(self):
		cur_epoch = 0
		for i in range(self.n_alternate):
			# force
			self.opt_param = "force"
			self.opt_med = "GD"
			self.constrain_form = "soft"
			self.epsilon = 1e-4
			self.alpha = 1.0

			if i > 0:
				Control.GetFrame(self.n_frame - 2, self.trajectory, self.X0)
				Control.GetFrame(self.n_frame - 1, self.trajectory, self.X1)
				Control.Copy(self.loss_residual, self.control_force)
				Control.Scale(self.control_force, 1.0 / self.dt ** 2)
				self.loss = self.compute_force_loss(self.control_force, self.trajectory)
				self.output_loss()
				print("================== [switch to force optimization] ===================")
				print(f"loss: {self.loss}, force: {self.force_loss}, constrain: {self.constrain_loss}")

			for j in range(self.n_force_epoch):
				b_congerge = self.one_iter(cur_epoch)
				self.compute_trajectory_loss(self.trajectory)
				print(f"[loopy loss]: {self.loopy_loss} ({self.force_loss * self.dt ** 4 * 2})")
				cur_epoch += 1
				if b_congerge:
					print("[force optimization converged] alpha: ", self.alpha)
					break

			# trajectory
			self.opt_param = "trajectory"
			self.opt_med = "GN"
			self.constrain_form = "soft"
			self.epsilon = -1
			self.alpha = 1.0

			if i == 0:
				Control.SetFrame(self.n_frame - 2, self.trajectory, self.X0)
				Control.SetFrame(self.n_frame - 1, self.trajectory, self.X1)
			self.valid, self.loss = self.compute_trajectory_loss(self.trajectory)
			Control.Write(self.loss_residual, self.output_folder + f"epoch_{cur_epoch - 1}/residual.txt")
			self.output_loss()
			print("================== [switch to trajectory optimization] ===================")
			print(f"valid: {self.valid}, loss: {self.loss}, loopy: {self.loopy_loss}, constrain: {self.constrain_loss}")

			for j in range(self.n_trajectory_epoch):
				self.one_iter(cur_epoch)
				cur_epoch += 1
		
	def forward(self, control_force):
		""" update to self.trajectory """
		print("[start forward]", end=" ", flush=True)
		self.sim.t = 0.0
		self.sim.PNIterCount = 0
		Control.Copy(self.X0, self.sim.X)
		Control.SetVelocity(self.sim.nodeAttr, self.sim.init_velocity)
		Control.Copy(self.init_DBC, self.sim.DBC)

		self.sim.step(0) # skip X1

		for i in range(self.n_frame):
			Control.GetFrame(i, control_force, self.one_frame)
			self.sim.step(i + 1, self.one_frame)
			Control.SetFrame(i, self.trajectory, self.sim.X)

		print(f"avg. PN iter: {self.sim.PNIterCount / (self.n_frame + 2)}")


	def one_iter(self, cur_epoch):
		print(f"============================= epoch {cur_epoch} ==========================")
		if self.b_debug:
			epoch_folder = self.output_folder + f"epoch_{cur_epoch}/"
			make_directory(epoch_folder)
			self.sim.output_folder = epoch_folder

		if self.opt_param == "force":
			print("[compute adjoint vector]")
			Control.ComputeAdjointVector(self.n_vert, self.n_frame, self.dt, 
				self.sim.DBC, self.sim.Elem, self.sim.segs, self.sim.edge2tri, self.sim.edgeStencil, self.sim.edgeInfo,
				self.sim.thickness, self.sim.bendingStiffMult, self.sim.fiberStiffMult, self.sim.inextLimit, self.sim.s, self.sim.sHat, self.sim.kappa_s,
				self.sim.bodyForce, self.sim.k_wind, self.sim.wind_dir, self.sim.withCollision, self.sim.dHat2, self.sim.kappa, self.sim.mu, self.sim.epsv2, self.sim.fricIterAmt, self.sim.compNodeRange, self.sim.muComp, 
				self.sim.nodeAttr, self.sim.massMatrix, self.sim.elemAttr, self.sim.elasticity,
				self.sim.tet, self.sim.tetAttr, self.sim.tetElasticity, self.sim.rod, self.sim.rodInfo, self.sim.rodHinge, self.sim.rodHingeInfo, 
				self.sim.stitchInfo, self.sim.stitchRatio, self.sim.k_stitch, self.sim.discrete_particle,
				self.X0, self.X1, self.control_force, self.trajectory, self.gradient)
			
			if self.constrain_form == "hard" and self.opt_med == "FP":
				b_converge = self.fast_projection()
			elif self.constrain_form == "soft":
				# add 1/2 f^T * f
				Control.Scale(self.gradient, self.n_frame)
				Control.Axpy(self.epsilon * self.dt ** 4, self.control_force, self.gradient)
	
				Control.Copy(self.gradient, self.descent_dir)
				Control.Scale(self.descent_dir, -1.0)
				if self.opt_med == "L-BFGS":
					self.L_BFGS(cur_epoch, self.control_force, self.gradient)

				b_converge = self.force_line_search()

			Control.Copy(self.trajectory, self.loopy_trajectory)
			Control.SetFrame(self.n_frame - 2, self.loopy_trajectory, self.X0)
			Control.SetFrame(self.n_frame - 1, self.loopy_trajectory, self.X1)
			_, _ = self.compute_trajectory_loss(self.loopy_trajectory, False)
			print(f"loss: {self.loss}, force: {self.force_loss}, constrain: {self.constrain_loss}, loopy: {self.loopy_loss}")

			# TIMER_FLUSH(cur_epoch + 1, self.n_epoch, cur_epoch + 1, self.n_epoch)

		elif self.opt_param == "trajectory":
			if self.use_cg:
				self.cg_iter = min(self.n_vert * self.n_frame * 3, self.cg_iter + 500)
			if self.constrain_form == "hard":
				if self.opt_med == "GN":
					Control.ComputeTrajectoryGradientDescent(self.n_vert, self.n_frame, self.dt, self.p, self.epsilon, self.use_cg, self.cg_iter, self.cg_tol, self.cg_regu,
						self.sim.DBC, self.sim.Elem, self.sim.segs, self.sim.edge2tri, self.sim.edgeStencil, self.sim.edgeInfo,
						self.sim.thickness, self.sim.bendingStiffMult, self.sim.fiberStiffMult, self.sim.inextLimit, self.sim.s, self.sim.sHat, self.sim.kappa_s,
						self.sim.bodyForce, self.sim.k_wind, self.sim.wind_dir, self.sim.withCollision, self.sim.dHat2, self.sim.kappa, self.sim.mu, self.sim.epsv2, self.sim.fricIterAmt, self.sim.compNodeRange, self.sim.muComp, 
						self.sim.nodeAttr, self.sim.massMatrix, self.sim.elemAttr, self.sim.elasticity,
						self.sim.tet, self.sim.tetAttr, self.sim.tetElasticity, self.sim.rod, self.sim.rodInfo, self.sim.rodHinge, self.sim.rodHingeInfo, 
						self.sim.stitchInfo, self.sim.stitchRatio, self.sim.k_stitch, self.sim.discrete_particle,
						self.X0, self.X1, self.trajectory, self.gradient, self.descent_dir, self.residual, self.hess_residual, self.A)
				else:
					Control.ComputeTrajectoryGradient(self.n_vert, self.n_frame, self.dt, self.p, self.epsilon, self.use_cg, self.cg_iter, self.cg_tol, self.cg_regu,
						self.sim.DBC, self.sim.Elem, self.sim.segs, self.sim.edge2tri, self.sim.edgeStencil, self.sim.edgeInfo,
						self.sim.thickness, self.sim.bendingStiffMult, self.sim.fiberStiffMult, self.sim.inextLimit, self.sim.s, self.sim.sHat, self.sim.kappa_s,
						self.sim.bodyForce, self.sim.k_wind, self.sim.wind_dir, self.sim.withCollision, self.sim.dHat2, self.sim.kappa, self.sim.mu, self.sim.epsv2, self.sim.fricIterAmt, self.sim.compNodeRange, self.sim.muComp, 
						self.sim.nodeAttr, self.sim.massMatrix, self.sim.elemAttr, self.sim.elasticity,
						self.sim.tet, self.sim.tetAttr, self.sim.tetElasticity, self.sim.rod, self.sim.rodInfo, self.sim.rodHinge, self.sim.rodHingeInfo, 
						self.sim.stitchInfo, self.sim.stitchRatio, self.sim.k_stitch, self.sim.discrete_particle,
						self.X0, self.X1, self.trajectory, self.gradient, self.descent_dir, self.residual, self.hess_residual, self.A)
				if self.opt_med == "L-BFGS":
					self.L_BFGS(cur_epoch, self.trajectory, self.gradient)
			elif self.constrain_form == "soft":
				if self.opt_med == "GN":
					Control.ComputeTrajectoryGradientDescent_SoftCon(self.n_vert, self.n_frame, self.dt, self.p, self.epsilon, self.use_cg, self.cg_iter, self.cg_tol, self.cg_regu,
						self.sim.DBC, self.sim.Elem, self.sim.segs, self.sim.edge2tri, self.sim.edgeStencil, self.sim.edgeInfo,
						self.sim.thickness, self.sim.bendingStiffMult, self.sim.fiberStiffMult, self.sim.inextLimit, self.sim.s, self.sim.sHat, self.sim.kappa_s,
						self.sim.bodyForce, self.sim.k_wind, self.sim.wind_dir, self.sim.withCollision, self.sim.dHat2, self.sim.kappa, self.sim.mu, self.sim.epsv2, self.sim.fricIterAmt, self.sim.compNodeRange, self.sim.muComp, 
						self.sim.nodeAttr, self.sim.massMatrix, self.sim.elemAttr, self.sim.elasticity,
						self.sim.tet, self.sim.tetAttr, self.sim.tetElasticity, self.sim.rod, self.sim.rodInfo, self.sim.rodHinge, self.sim.rodHingeInfo, 
						self.sim.stitchInfo, self.sim.stitchRatio, self.sim.k_stitch, self.sim.discrete_particle,
						self.X0, self.X1, self.trajectory, self.gradient, self.descent_dir, self.residual, self.hess_residual, self.A)
				else:
					Control.ComputeTrajectoryGradient_SoftCon(self.n_vert, self.n_frame, self.dt, self.p, self.epsilon, self.use_cg, self.cg_iter, self.cg_tol, self.cg_regu,
						self.sim.DBC, self.sim.Elem, self.sim.segs, self.sim.edge2tri, self.sim.edgeStencil, self.sim.edgeInfo,
						self.sim.thickness, self.sim.bendingStiffMult, self.sim.fiberStiffMult, self.sim.inextLimit, self.sim.s, self.sim.sHat, self.sim.kappa_s,
						self.sim.bodyForce, self.sim.k_wind, self.sim.wind_dir, self.sim.withCollision, self.sim.dHat2, self.sim.kappa, self.sim.mu, self.sim.epsv2, self.sim.fricIterAmt, self.sim.compNodeRange, self.sim.muComp, 
						self.sim.nodeAttr, self.sim.massMatrix, self.sim.elemAttr, self.sim.elasticity,
						self.sim.tet, self.sim.tetAttr, self.sim.tetElasticity, self.sim.rod, self.sim.rodInfo, self.sim.rodHinge, self.sim.rodHingeInfo, 
						self.sim.stitchInfo, self.sim.stitchRatio, self.sim.k_stitch, self.sim.discrete_particle,
						self.X0, self.X1, self.trajectory, self.gradient, self.descent_dir, self.residual, self.hess_residual, self.A,)
				if self.opt_med == "L-BFGS":
					self.L_BFGS(cur_epoch, self.trajectory, self.gradient)

			b_converge = self.trajectory_line_search()

			print(f"valid: {self.valid}, loss: {self.loss}, loopy: {self.loopy_loss}, constrain: {self.constrain_loss}")

		self.output_loss()
		if self.b_debug:
			self.output_trajectory(epoch_folder)
			self.output_debug_info(epoch_folder)

		# TIMER_FLUSH(cur_epoch + 1, self.n_epoch, cur_epoch + 1, self.n_epoch)

		return b_converge

	def L_BFGS(self, cur_epoch, x, g):
		if cur_epoch > 0:
			# enque
			self.delta_x_history.rotate(1)
			Control.Copy(x, self.delta_x_history[0])
			Control.Axpy(-1.0, self.last_x, self.delta_x_history[0])

			self.delta_g_history.rotate(1)
			Control.Copy(g, self.delta_g_history[0])
			Control.Axpy(-1.0, self.last_g, self.delta_g_history[0])

			self.pho.rotate(1)
			self.pho[0] = 1.0 / Control.Reduce(self.delta_x_history[0], self.delta_g_history[0])

			self.cur_window_size = min(self.window_size, self.cur_window_size + 1)

			# compute descent dir
			alpha = []
			for i in range(self.cur_window_size):
				alpha.append(self.pho[i] * Control.Reduce(self.descent_dir, self.delta_x_history[i]))
				Control.Axpy(-alpha[-1], self.delta_g_history[i], self.descent_dir)

			gamma = Control.Reduce(self.delta_x_history[0], self.delta_g_history[0])
			gamma /= Control.Reduce(self.delta_g_history[0], self.delta_g_history[0])
			Control.Scale(self.descent_dir, gamma)

			for i in reversed(range(self.cur_window_size)):
				beta = self.pho[i] * Control.Reduce(self.descent_dir, self.delta_g_history[i])
				Control.Axpy(alpha[i] - beta, self.delta_x_history[i], self.descent_dir)

	def force_line_search(self):
		self.alpha *= 2
		start_alpha = self.alpha
		threshold = 0.03 * Control.Reduce(self.gradient, self.descent_dir)

		while True:
			if self.alpha < 1e-6 * start_alpha:
				break

			Control.Copy(self.control_force, self.tentative)
			Control.Axpy(self.alpha, self.descent_dir, self.tentative)
			self.forward(self.tentative)
			loss = self.compute_force_loss(self.tentative, self.trajectory)

			print(f"[line search] alpha {self.alpha}: {loss}, threshold: {self.loss + self.alpha * threshold}")

			if loss < self.loss + self.alpha * threshold:
				break

			self.alpha /= 2

		if self.opt_med == "L-BFGS":
			Control.Copy(self.control_force, self.last_x)
			Control.Copy(self.gradient, self.last_g)
		Control.Copy(self.tentative, self.control_force)
		self.loss = loss

		return self.alpha < 1e-6 * start_alpha

	def trajectory_line_search(self):
		self.alpha *= 2
		start_alpha = self.alpha
		threshold = 0.03 * Control.Reduce(self.gradient, self.descent_dir)

		while True:
			if self.alpha < 1e-6 * start_alpha:
				break

			Control.Copy(self.trajectory, self.tentative)
			Control.Axpy(self.alpha, self.descent_dir, self.tentative)
			valid, loss = self.compute_trajectory_loss(self.tentative)

			print(f"[line search] alpha {self.alpha}: {valid} {loss}, threshold: {self.valid} {self.loss + self.alpha * threshold}")

			if valid and not self.valid:
				break
			if valid and self.valid and loss < self.loss + self.alpha * threshold:
				break

			self.alpha /= 2

		if self.opt_med == "L-BFGS":
			Control.Copy(self.trajectory, self.last_x)
			Control.Copy(self.gradient, self.last_g)

		Control.Copy(self.tentative, self.trajectory)

		self.valid, self.loss = valid, loss

		return self.alpha < 1e-6 * start_alpha

	def fast_projection(self):
		print("[fast projection]")
		grad_C_norm2 = Control.Reduce(self.gradient, self.gradient)
		delta_L = self.constrain_loss / grad_C_norm2
		Control.Axpy(-delta_L, self.gradient, self.control_force)
		self.L += delta_L

		self.forward(self.control_force)
		self.loss = self.compute_force_loss(self.control_force, self.trajectory)

		return False

	def compute_force_loss(self, control_force, trajectory):
		self.constrain_loss = Control.ComputeConstrain(self.n_vert, self.n_frame, self.sim.massMatrix, self.X0, self.X1, trajectory)
		self.force_loss = 0.5 * Control.Reduce(control_force, control_force)
		if self.constrain_form == "soft":
			return self.force_loss + self.n_frame * self.epsilon / (self.dt ** 3) * self.constrain_loss
		elif self.constrain_form == "hard":
			return self.force_loss + self.L * self.constrain_loss

	def compute_trajectory_loss(self, trajectory, b_con=True):
		valid = Control.ComputeLoopyLoss(self.n_vert, self.n_frame, self.dt, self.p,
			self.sim.DBC, self.sim.Elem, self.sim.segs, self.sim.edge2tri, self.sim.edgeStencil, self.sim.edgeInfo,
			self.sim.thickness, self.sim.bendingStiffMult, self.sim.fiberStiffMult, self.sim.inextLimit, self.sim.s, self.sim.sHat, self.sim.kappa_s,
			self.sim.bodyForce, self.sim.k_wind, self.sim.wind_dir, self.sim.withCollision, self.sim.dHat2, self.sim.kappa, self.sim.mu, self.sim.epsv2, self.sim.fricIterAmt, self.sim.compNodeRange, self.sim.muComp, 
			self.sim.nodeAttr, self.sim.massMatrix, self.sim.elemAttr, self.sim.elasticity,
			self.sim.tet, self.sim.tetAttr, self.sim.tetElasticity, self.sim.rod, self.sim.rodInfo, self.sim.rodHinge, self.sim.rodHingeInfo, 
			self.sim.stitchInfo, self.sim.stitchRatio, self.sim.k_stitch, self.sim.discrete_particle,
			self.X0, self.X1, trajectory, self.valid_per_frame, self.loss_per_frame, self.loss_residual)
		
		self.loopy_loss = 0.0
		if valid:
			for i in range(len(self.loss_per_frame)):
				self.loss_per_frame[i] /= self.dt ** 4
				self.loopy_loss += self.loss_per_frame[i]
		
		if b_con:
			self.constrain_loss = Control.ComputeConstrain(self.n_vert, self.n_frame, self.sim.massMatrix, self.X0, self.X1, trajectory)
		
		loss = self.loopy_loss + self.n_frame * self.epsilon / (self.dt ** 3) * self.constrain_loss

		return valid, loss

	def output_trajectory(self, epoch_folder):
		self.sim.output_folder = epoch_folder
		Control.Copy(self.X0, self.sim.X)
		self.sim.write(0)
		Control.Copy(self.X1, self.sim.X)
		self.sim.write(1)

		# write .obj
		for i in range(self.n_frame):
			Control.GetFrame(i, self.trajectory, self.sim.X)
			self.sim.write(i + 2)

		# write .txt
		if self.opt_param == "force":
			Control.Write(self.trajectory, epoch_folder + "trajectory.txt")

	def output_loss(self):
		with open(self.output_folder + "loss.txt", 'a') as f:
			if self.opt_param == "force":
				f.write(f"{self.alpha * 2} {self.loss} {self.force_loss} {self.constrain_loss} {self.loopy_loss}\n")
				f.write(' '.join([str(x) for x in self.loss_per_frame]))
				f.write('\n')
			elif self.opt_param == "trajectory":
				f.write(f"{self.alpha * 2} {self.loss} {self.loopy_loss} {self.constrain_loss}\n")
				f.write(' '.join([str(x) for x in self.loss_per_frame]))
				f.write('\n')

	def output_debug_info(self, epoch_folder):
		if self.opt_param == "force":
			Control.Write(self.control_force, epoch_folder + "control_force.txt")
			Control.Write(self.gradient, epoch_folder + "adjoint_vector.txt")
			Control.Write(self.descent_dir, epoch_folder + "descent_dir.txt")
		elif self.opt_param == "trajectory":
			Control.Write(self.residual, epoch_folder + "residual.txt")
			Control.Write(self.loss_residual, epoch_folder + "loss_residual.txt")
			Control.Write(self.hess_residual, epoch_folder + "hess_residual.txt")
			Control.Write(self.gradient, epoch_folder + "gradient.txt")
			Control.Write(self.descent_dir, epoch_folder + "descent_dir.txt")
			self.A.Output(epoch_folder + "A.txt")

	def register_logger(self):
		class Logger(object):
			def __init__(self, output_folder):
				log_folder = output_folder + "log/"
				make_directory(log_folder)
				self.terminal = sys.stdout
				self.log = open(log_folder + "log.txt", "w")

			def write(self, decorated_message):
				raw_message = re.sub(r'\x1b\[[\d;]+m', '', decorated_message);
				self.terminal.write(decorated_message)
				self.log.write(raw_message)

			def flush(self):
				pass
		sys.stdout = Logger(self.output_folder)
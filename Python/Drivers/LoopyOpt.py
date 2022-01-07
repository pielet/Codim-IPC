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
import numpy as np
from JGSL import *

class LoopyOpt:
    def __init__(self, sim, opt_param="force", constrain_form="hard", opt_med="GN") -> None:
        """
        sim: simulation base
        param_med: force, trajectory
        opt_med: GD (gradient descent_, L-BFGS, GN (Gauss-Newton), and FP (fast projection) (force-based only)
        """
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
        self.M = self.sim.massMatrix
        
        # optimization params
        self.n_epoch = 100
        self.epsilon = 1e-4 # soft contrain
        self.L = 0.0 # hard constrain
        self.init_med = "fix"  # for trajectory only: static, solve (0 loopy loss), fix (0 constrain loss), load (mixed loss)
        self.linear_solver_type = "direct"  # for trajectory only, direct or pcg

        # storage
        self.X0 = Storage.V3dStorage()
        self.X1 = Storage.V3dStorage()
        self.one_frame = Storage.V3dStorage()
        self.trajectory = Storage.V3dStorage()
        self.control_force = Storage.V3dStorage()
        # line-search
        self.gradient = Storage.V3dStorage()
        self.descent_dir = Storage.V3dStorage()

        # loss
        self.loss = 0.0
        self.force_loss = 0.0
        self.loopy_loss = 0.0
        self.constrain_loss = 0.0

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
        Control.Fill(self.gradient, self.n_vert * self.n_frame)
        Control.Fill(self.descent_dir, self.n_vert * self.n_frame)

        # compute initial loss
        Control.Copy(self.sim.X, self.X0)
        self.forward(self.control_force)
        if self.opt_param == "force":
            if self.constrain_form == "soft":
                self.loss, self.force_loss, self.constrain_loss = self.compute_force_soft_loss(self.control_force, self.trajectory)
            elif self.constrain_form == "hard":
                self.loss, self.force_loss, self.constrain_loss = self.compute_force_hard_loss(self.control_force, self.trajectory)

    def run(self):
        for i in range(self.n_epoch):
            self.one_iter(i)
        
    def forward(self, control_force):
        """ update to self.trajectory """
        print("[start forward]")
        self.t = 0.0
        Control.Copy(self.X0, self.sim.X)
        Control.ZeroVelocity(self.sim.nodeAttr)

        self.sim.step() # skip X1
        Control.Copy(self.sim.X, self.X1)

        for i in range(self.n_frame):
            print(f"=================== frame {i} ===================")
            Control.GetFrame(i, control_force, self.one_frame)
            self.sim.step(self.one_frame)
            Control.SetFrame(i, self.trajectory, self.sim.X)

    def one_iter(self, cur_epoch):
        print(f"============================= epoch {cur_epoch} ==========================")
        epoch_folder = self.output_folder + f"epoch_{cur_epoch}/"
        make_directory(epoch_folder)
        self.sim.output_folder = epoch_folder

        if self.opt_param == "force":
            if self.opt_med == "FP":
                self.fast_projection()
            print(f"loss: {self.loss} force: {self.force_loss} constrain: {self.constrain_loss}")

        self.output_trajectory(epoch_folder)
        self.output_loss()

        TIMER_FLUSH(cur_epoch, self.n_epoch, cur_epoch, self.n_epoch)

    def fast_projection(self):
        print("[fast projection]")

        Control.ComputeAdjointVector(self.n_vert, self.n_frame, self.dt,
            self.sim.MDBC_tmin, self.sim.MDBC_tmax, self.sim.MDBC_period, self.sim.DBC, self.sim.DBCMotion,
            self.sim.Elem, self.sim.segs, self.sim.edge2tri, self.sim.edgeStencil, self.sim.edgeInfo,
            self.sim.thickness, self.sim.bendingStiffMult, self.sim.fiberStiffMult, self.sim.inextLimit, self.sim.s, self.sim.sHat, self.sim.kappa_s,
            self.sim.bodyForce, self.sim.withCollision, self.sim.dHat2, self.sim.kappa, self.sim.mu, self.sim.epsv2, self.sim.fricIterAmt, self.sim.compNodeRange, self.sim.muComp, 
            self.sim.nodeAttr, self.sim.massMatrix, self.sim.elemAttr, self.sim.elasticity,
            self.sim.tet, self.sim.tetAttr, self.sim.tetElasticity, self.sim.rod, self.sim.rodInfo, self.sim.rodHinge, self.sim.rodHingeInfo, 
            self.sim.stitchInfo, self.sim.stitchRatio, self.sim.k_stitch, self.sim.discrete_particle,
            self.X0, self.X1, self.control_force, self.trajectory, self.gradient)

        grad_C_norm2 = Control.Reduce(self.gradient, self.gradient)

        Control.Axpy(-self.constrain_loss / grad_C_norm2, self.gradient, self.control_force)

        self.forward(self.control_force)
        # TODO: check soft & hard constrain loss for SAP
        self.loss, self.force_loss, self.constrain_loss = self.compute_force_soft_loss(self.control_force, self.trajectory)

    def compute_force_soft_loss(self, control_force, trajectory):
        constrain_loss = Control.ComputeForceConstrain(self.n_vert, self.n_frame, self.sim.massMatrix, self.X0, self.X1, trajectory)
        force_loss = 0.5 * Control.Reduce(control_force, control_force)
        loss = self.epsilon * self.dt ** 4 * force_loss + self.n_frame * constrain_loss
        return loss, force_loss, constrain_loss

    def compute_force_hard_loss(self, control_force, trajectory):
        constrain_loss = Control.ComputeForceConstrain(self.n_vert, self.n_frame, self.sim.massMatrix, self.X0, self.X1, trajectory)
        force_loss = 0.5 * Control.Reduce(control_force, control_force)
        return force_loss + self.L * constrain_loss, force_loss, constrain_loss

    def output_trajectory(self, epoch_folder, b_output_params=True):
        self.sim.output_folder = epoch_folder
        Control.Copy(self.X0, self.sim.X)
        self.sim.write(0)
        Control.Copy(self.X1, self.sim.X)
        self.sim.write(1)

        for i in range(self.n_frame):
            Control.GetFrame(i, self.trajectory, self.sim.X)
            self.sim.write(i + 2)

        if b_output_params:
            if self.opt_param == "force":
                Control.Write(self.control_force, epoch_folder + "control_force.txt")
            Control.Write(self.trajectory, epoch_folder + "trajectory.txt")

    def output_loss(self):
        with open(self.output_folder + "loss.txt", 'a') as f:
            if self.opt_param == "force":
                f.write(f"{self.loss} {self.force_loss} {self.constrain_loss}\n")
            elif self.opt_param == "trajectory":
                f.write(f"{self.loss} {self.loopy_loss} {self.constrain_loss}\n")

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
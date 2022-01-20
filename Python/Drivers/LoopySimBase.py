import sys
import os
import re

import math
import numpy as np

from JGSL import *

sys.path.insert(0, "../../build")

try:
    os.mkdir("output")
except OSError:
    pass

def make_directory(folder):
    try:
        os.mkdir(folder)
    except OSError:
        pass

class LoopySimBase:
    def __init__(self, precision, dim, opt=False):
        Kokkos_Initialize()
        self.precision = precision
        self.dim = dim
        assert precision == "float" or precision == "double"
        assert dim == 2 or dim == 3
        self.set_type()

        self.dt = 0.01
        self.dx = 0.01
        self.gravity = self.Vec()
        self.frame_num = 240
        self.current_frame = 0
        self.symplectic = True

        self.output_folder = "output/" + os.path.splitext(os.path.basename(sys.argv[0]))[0] + "/"
        if not opt:
            make_directory(self.output_folder)
            if len(sys.argv) > 1:
                self.output_folder += sys.argv[1]
                for i in range(2, len(sys.argv)):
                    self.output_folder += "_" + sys.argv[i]
                self.output_folder += "/"
            make_directory(self.output_folder)

            self.register_logger()

        self.update_scale = None
        self.update_offset = None

        self.X0 = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        self.X = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        self.X_stage = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        self.Elem = Storage.V2iStorage() if self.dim == 2 else Storage.V3iStorage()
        self.segs = StdVectorVector2i()
        self.outputSeg = False
        self.nodeAttr = Storage.V2dV2dV2dSdStorage() if self.dim == 2 else Storage.V3dV3dV3dSdStorage()
        self.massMatrix = CSR_MATRIX_D()
        self.elemAttr = Storage.M2dM2dSdStorage()
        self.elasticity = FIXED_COROTATED_2.Create() #TODO: different material switch
        self.DBC = Storage.V3dStorage() if self.dim == 2 else Storage.V4dStorage()
        self.gravity = self.Vec(0, -9.81) if self.dim == 2 else self.Vec(0, -9.81, 0)
        self.bodyForce = StdVectorXd()
        self.edge2tri = StdMapPairiToi()
        self.edgeStencil = StdVectorVector4i()
        self.edgeInfo = StdVectorVector3d()
        self.thickness = 0 #TODO different thickness
        self.bendingStiffMult = 1
        self.fiberStiffMult = Vector4d(0, 0, 0, 0)
        self.inextLimit = Vector3d(0, 0, 0)
        self.s = Vector2d(1.01, 0)
        self.sHat = Vector2d(1, 1)
        self.kappa_s = Vector2d(0, 0)
        self.withCollision = False
        self.PNIterCount = 0
        self.PNTol = 1e-3
        self.dHat2 = 1e-6
        self.kappa = Vector3d(1e5, 0, 0)
        self.mu = 0
        self.epsv2 = 1e-6
        self.fricIterAmt = 1
        self.compNodeRange = StdVectorXi()
        self.muComp = StdVectorXd()
        self.staticSolve = False
        self.t = 0.0
        self.DBCMotion = Storage.DBCMotion()
        self.n_vert = 0
        self.n_DBC = 0
        self.init_velocity = Vector3d(0, 0, 0)

        self.k_wind = 0
        self.wind_dir = Vector3d(1, 0, 0)

        # 100% cotton, 100% wool, 95% wool 5% lycra, 100% polyester (PES), paper
        self.cloth_density_iso = [472.641509, 413.380282, 543.292683, 653.174603, 800]
        self.cloth_thickness_iso = [0.318e-3, 0.568e-3, 0.328e-3, 0.252e-3, 0.3e-3]
        self.cloth_Ebase_iso = [0.821e6, 0.170e6, 0.076e6, 0.478e6, 3e9] # for memb: 0.1x, 0.2, 1x, 0.1x, 2e4 ~ 1e5, for bending: 5e4 ~ 1e6
        self.cloth_membEMult_iso = [0.1, 0.2, 1, 0.1, 1]
        self.cloth_nubase_iso = [0.243, 0.277, 0.071, 0.381, 0.3]
        self.cloth_SL_iso = [1.0608, 1.085, 1.134, 1.0646, 1.005]

        # cotton, wool, canvas, silk, denim
        self.cloth_density = [103.6, 480.6, 294, 83, 400]
        self.cloth_thickness = [0.18e-3, 1.28e-3, 0.53e-3, 0.18e-3, 0.66e-3]
        self.cloth_Ebase = [1.076e6, 0.371e6, 2.009e6, 0.57e6, 2.448e6]
        self.cloth_weftWarpMult = [Vector4d(15.557e6, 7.779e6, 25.004e6, 1.076e6),\
            Vector4d(2.29e6, 1.145e6, 2.219e6, 0.371e6),\
            Vector4d(5.366e6, 2.683e6, 19.804e6, 2.009e6),\
            Vector4d(4.3e6, 4.971e6, 9.942e6, 0.57e6),\
            Vector4d(4.793e6, 4.515e6, 9.029e6, 2.448e6)]
        self.cloth_inextLimit = [Vector3d(0.14, 0.14, 0.063),\
            Vector3d(0.5, 0.62, 0.12),\
            Vector3d(0.11, 0.067, 0.059),\
            Vector3d(0.41, 0.34, 0.11),\
            Vector3d(0.28, 0.28, 0.05)]

        self.withVol = False
        self.tet = Storage.V4iStorage()
        self.tetAttr = Storage.M3dM3dSdStorage()
        self.tetElasticity = FIXED_COROTATED_3.Create() #TODO: different material switch
        self.TriVI2TetVI = Storage.SiStorage() #TODO: together?
        self.Tri = Storage.V3iStorage()
        self.outputRod = False
        self.rod = StdVectorVector2i()
        self.rodInfo = StdVectorVector3d()
        self.rodHinge = StdVectorVector3i()
        self.rodHingeInfo = StdVectorVector3d()
        self.discrete_particle = StdVectorXi()
        self.elasticIPC = False
        self.split = False

        self.stitchInfo = StdVectorVector3i()
        self.stitchRatio = StdVectorXd()
        self.k_stitch = 10

        self.shell_density = 1000
        self.shell_E = 1e5
        self.shell_nu = 0.4
        self.shell_thickness = 0.001

        self.scaleX = 1
        self.scaleY = 1
        self.scaleZ = 1
        self.scaleXTarget = 1
        self.scaleYTarget = 1
        self.scaleZTarget = 1
        self.scaleXMultStep = 1
        self.scaleYMultStep = 1
        self.scaleZMultStep = 1
        self.zeroVel = False

    def register_logger(self):
        class Logger(object):
            def __init__(self, output_folder):
                log_folder = output_folder + "log/"
                make_directory(log_folder)
                Set_Parameter("Basic.log_folder", log_folder)
                self.terminal = sys.stdout
                self.log = open(log_folder + "log.txt", "w")

            def write(self, decorated_message):
                raw_message = re.sub(r'\x1b\[[\d;]+m', '', decorated_message);
                self.terminal.write(decorated_message)
                self.log.write(raw_message)

            def flush(self):
                pass
        sys.stdout = Logger(self.output_folder)

    def set_type(self):
        if self.precision == "float":
            self.Scalar = Scalarf
            self.Vec = Vector2f if self.dim == 2 else Vector3f
            self.Mat = Matrix2f if self.dim == 2 else Matrix3f
        else:
            self.Scalar = Scalard
            self.Vec = Vector2d if self.dim == 2 else Vector3d
            self.Mat = Matrix2d if self.dim == 2 else Matrix3d

    def run(self):
        self.write(0)
        # do it twice to make sure the image is shown
        for f in range(self.frame_num):
            self.current_frame = f + 1
            self.step(f)
            self.write(f + 1)
            TIMER_FLUSH(f + 1, self.frame_num, f + 1, self.frame_num)
            if Get_Parameter("Terminate", False):
                break

    def add_shell_3D(self, filePath, translate, rotCenter, rotAxis, rotDeg): # 3D
        FEM.DiscreteShell.Add_Shell(filePath, translate, Vector3d(1, 1, 1), rotCenter, rotAxis, rotDeg, self.X, self.Elem, self.compNodeRange)
        self.n_vert = self.compNodeRange[-1]
        print(f"add component, #total_vertex_num={self.n_vert}")

    def add_object_3D(self, filePath, translate, rotCenter, rotAxis, rotDeg, scale):
        self.withVol = True
        ext = filePath.split('.')[-1]
        if ext == "vtk":
            meshCounter = MeshIO.Read_TetMesh_Vtk(filePath, self.X, self.tet)
        elif ext == "mesh":
            meshCounter = MeshIO.Read_TetMesh_Mesh(filePath, self.X, self.tet)
        self.n_vert = meshCounter[2]
        MeshIO.Transform_Points(translate, rotCenter, rotAxis, rotDeg, scale, meshCounter, self.X)
        print(f"add component, #total_vertex_num={self.n_vert}")
        return meshCounter

    def initialize(self, clothI, b_SL, membEMult=0.01, bendEMult=1, b_gravity=True):
        MeshIO.Append_Attribute(self.X, self.X0)
        self.shell_density = self.cloth_density_iso[clothI]
        self.shell_E = self.cloth_Ebase_iso[clothI] * membEMult
        self.shell_nu = self.cloth_nubase_iso[clothI]
        self.shell_thickness = self.cloth_thickness_iso[clothI]
        self.thickness = self.shell_thickness # later used as offset

        if not b_gravity:
            self.gravity = self.Vec(0, 0) if self.dim == 2 else self.Vec(0, 0, 0)
        
        self.dHat2 = FEM.DiscreteShell.Initialize_Shell_Hinge_EIPC(self.shell_density, self.shell_E, self.shell_nu, self.shell_thickness, self.dt, self.dHat2, self.X, self.Elem, self.segs, \
            self.edge2tri, self.edgeStencil, self.edgeInfo, self.nodeAttr, self.massMatrix, self.gravity, self.bodyForce, \
            self.elemAttr, self.elasticity, self.kappa)

        self.bendingStiffMult = bendEMult / membEMult

        if b_SL:
            self.kappa_s = Vector2d(1e3, 0)
            self.s = Vector2d(self.cloth_SL_iso[clothI], 0)

    def initialize_added_objects(self, velocity, p_density, E, nu):
        self.init_velocity = velocity
        MeshIO.Find_Surface_TriMesh(self.X, self.tet, self.TriVI2TetVI, self.Tri)
        vol = Storage.SdStorage()
        FEM.Compute_Vol_And_Inv_Basis(self.X, self.tet, vol, self.tetAttr)
        FIXED_COROTATED_3.All_Append_FEM(self.tetElasticity, vol, E, nu)
        FEM.Compute_Mass_And_Init_Velocity_NoAlloc(self.X, self.tet, vol, p_density, velocity, self.nodeAttr)
        FEM.Augment_Mass_Matrix_And_Body_Force(self.X, self.tet, vol, p_density, self.gravity, self.massMatrix, self.bodyForce)
    
    def initialize_OIPC(self, thickness, offset, stiffMult = 1):
        self.dHat2 = FEM.DiscreteShell.Initialize_OIPC(0.0, 0.0, thickness, 0.0, self.massMatrix, self.kappa, stiffMult)
        self.elasticIPC = False
        self.thickness = offset

    def load_frame(self, filePath):
        newX = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        newElem = Storage.V2iStorage() if self.dim == 2 else Storage.V3iStorage()
        MeshIO.Read_TriMesh_Obj(filePath, newX, newElem)
        self.X = newX
        # self.Elem = newElem
        FEM.Reset_Dirichlet(self.X, self.DBC)
    
    def load_velocity(self, folderPath, lastFrame, h):
        MeshIO.Load_Velocity(folderPath, lastFrame, h, self.nodeAttr)

    def set_DBC(self, DBC_bbox_min, DBC_bbox_max, idx_range=Vector4i(0, 0, 1000000000, -1)):
        DBC_range = Control.Set_Dirichlet(self.X, DBC_bbox_min, DBC_bbox_max, self.DBC, idx_range)
        self.n_DBC = DBC_range[1]
        return DBC_range

    def add_motion(self, begin, end, DBC_range, dist, rotCenter, rotAxis, angle, ease_ratio=0.2):
        Control.Add_DBC_Motion(begin, end, ease_ratio, DBC_range, dist, rotCenter, rotAxis, angle, self.DBCMotion)

    def step(self, cur_step, control_force=None):
        self.t += self.dt
        Control.Update_Dirichlet(self.t, self.DBCMotion, self.dt, self.DBC)

        if not control_force:
            control_force = Storage.V3dStorage()
            Control.Fill(control_force, self.n_vert)

        if self.elasticIPC:
            self.PNIterCount = self.PNIterCount + Control.Step_EIPC(cur_step, self.Elem, self.segs, self.DBC, \
                self.edge2tri, self.edgeStencil, self.edgeInfo, \
                self.thickness, self.bendingStiffMult, self.fiberStiffMult, self.inextLimit, self.s, self.sHat, self.kappa_s, \
                self.bodyForce, self.k_wind, self.wind_dir, control_force, self.dt, self.PNTol, self.withCollision, self.dHat2, self.kappa, self.mu, self.epsv2, self.fricIterAmt, \
                self.compNodeRange, self.muComp, self.staticSolve, \
                self.X, self.nodeAttr, self.massMatrix, self.elemAttr, self.elasticity, \
                self.tet, self.tetAttr, self.tetElasticity, self.rod, self.rodInfo, \
                self.rodHinge, self.rodHingeInfo, self.stitchInfo, self.stitchRatio, self.k_stitch,\
                self.discrete_particle, self.output_folder)
        else:
            self.PNIterCount = self.PNIterCount + Control.Step(cur_step, self.Elem, self.segs, self.DBC, \
                self.edge2tri, self.edgeStencil, self.edgeInfo, \
                self.thickness, self.bendingStiffMult, self.fiberStiffMult, self.inextLimit, self.s, self.sHat, self.kappa_s, \
                self.bodyForce, self.k_wind, self.wind_dir, control_force, self.dt, self.PNTol, self.withCollision, self.dHat2, self.kappa, self.mu, self.epsv2, self.fricIterAmt, \
                self.compNodeRange, self.muComp, self.staticSolve, \
                self.X, self.nodeAttr, self.massMatrix, self.elemAttr, self.elasticity, \
                self.tet, self.tetAttr, self.tetElasticity, self.rod, self.rodInfo, \
                self.rodHinge, self.rodHingeInfo, self.stitchInfo, self.stitchRatio, self.k_stitch,\
                self.discrete_particle, self.output_folder)
        # self.load_velocity('/Users/minchen/Desktop/JGSL/Projects/FEMShell/output/shrink/', 1, dt)
        if self.scaleXMultStep != 1 or self.scaleYMultStep != 1 or self.scaleZMultStep != 1:
            self.scaleX *= self.scaleXMultStep
            self.scaleY *= self.scaleYMultStep
            self.scaleZ *= self.scaleZMultStep
            if (self.scaleXMultStep > 1 and self.scaleX > self.scaleXTarget) or (self.scaleXMultStep < 1 and self.scaleX < self.scaleXTarget):
                self.scaleX = self.scaleXTarget
            if (self.scaleYMultStep > 1 and self.scaleY > self.scaleYTarget) or (self.scaleYMultStep < 1 and self.scaleY < self.scaleYTarget):
                self.scaleY = self.scaleYTarget
            if (self.scaleZMultStep > 1 and self.scaleZ > self.scaleZTarget) or (self.scaleZMultStep < 1 and self.scaleZ < self.scaleZTarget):
                self.scaleZ = self.scaleZTarget
            FEM.Update_Inv_Basis(self.X0, self.tet, self.tetAttr, self.scaleX, self.scaleY, self.scaleZ)
        if self.zeroVel:
            MeshIO.Zero_Velocity(self.nodeAttr)

    def write(self, frame_idx):
        if self.outputSeg:
            MeshIO.Write_SegMesh_Obj(self.X, self.segs, self.output_folder + "seg" + str(frame_idx) + ".obj")
        if self.outputRod:
            MeshIO.Write_SegMesh_Obj(self.X, self.rod, self.output_folder + "rod" + str(frame_idx) + ".obj")
        if self.withVol:
            MeshIO.Write_Surface_TriMesh_Obj(self.X, self.TriVI2TetVI, self.Tri, \
                    self.output_folder + "vol" + str(frame_idx) + ".obj")
            return
        MeshIO.Write_TriMesh_Obj(self.X, self.Elem, self.output_folder + "shell" + str(frame_idx) + ".obj")
import sys
sys.path.insert(0, "../../Python")
import Drivers
from JGSL import *

b_opt = False

if __name__ == "__main__":
    sim = Drivers.LoopySimBase("double", 3, b_opt)

    # sim params
    strain_limit = False
    cloth_material = 0
    size = '21'

    membEMult = 1
    bendEMult = 1

    sim.dt = 0.02
    sim.frame_num = 10
    sim.withCollision = False
    
    sim.add_object_3D("../FEM/input/bottle.vtk", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
    sim.add_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 0.1, 1.1), 
        Vector3d(1.0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0)

    if strain_limit:
        # iso
        sim.initialize(cloth_material, True, membEMult, bendEMult) # 0 means having gravity
    else:
        # iso, no strain limit
        sim.initialize(cloth_material, False, membEMult, bendEMult)

    sim.initialize_added_objects(Vector3d(0, 0, 0), 1000, 1e5, 0.4)
    sim.initialize_OIPC(1e-3, 0)

    if not b_opt:
        sim.run()
    else:
        # opt params (trajectory, force)
        opt_param = "force"
        constrain_type = "soft"
        opt_med = "GD"

        opt = Drivers.LoopyOpt(sim, opt_param, constrain_type, opt_med)
        opt.init_med = "solve"
        opt.n_epoch = 100
        opt.epsilon = 1e-4
        opt.p = 2
        opt.alpha = 1.0

        opt.initialize()
        opt.run()

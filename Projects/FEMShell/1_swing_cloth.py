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
    sim.frame_num = 50
    sim.withCollision = True
    
    sim.add_shell_3D("input/sphere1K_0.3.obj", Vector3d(0, -0.25, 0), \
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)
    # v, rotCenter, rotAxis, angVelDeg
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0)

    sim.add_shell_3D("input/square" + size + ".obj", Vector3d(0, sim.cloth_thickness[cloth_material] + 0.001, 0), \
        Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0)
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1e-3, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0)
    # sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1e-3, 1.1, 1.1), 
    #     Vector3d(-1, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0)
    # sim.MDBC_tmin = 1
    # sim.MDBC_tmax = 2

    if strain_limit:
        # iso
        sim.initialize(cloth_material, True, membEMult, bendEMult) # 0 means having gravity
    else:
        # iso, no strain limit
        sim.initialize(cloth_material, False, membEMult, bendEMult)

    sim.initialize_OIPC(1e-3, 0)

    if not b_opt:
        sim.run()
    else:
        # opt params
        opt_param = "force"
        constrain_type = "hard"
        opt_med = "FP"

        opt = Drivers.LoopyOpt(sim, opt_param, constrain_type, opt_med)

        opt.n_epoch = 10
        opt.epsilon = 1e-2

        opt.initialize()
        opt.run()

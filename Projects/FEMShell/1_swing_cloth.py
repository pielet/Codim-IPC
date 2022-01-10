import sys
sys.path.insert(0, "../../Python")
import Drivers
from JGSL import *

b_opt = True

if __name__ == "__main__":
    sim = Drivers.LoopySimBase("double", 3, b_opt)

    # sim params
    strain_limit = False
    cloth_material = 0
    size = '21'

    membEMult = 1
    bendEMult = 1

    sim.dt = 0.02
    sim.frame_num = 100
    sim.withCollision = False
    
    # sim.add_shell_3D("input/sphere1K_0.3.obj", Vector3d(0, -0.25, 0), \
    #     Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)
    # sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
    #     Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0)

    sim.add_shell_3D("input/square" + size + ".obj", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(0, 0, 1), 90)

    # DBC_bbox_min, DBC_bbox_max, idx_range
    DBC_range = sim.set_DBC(Vector3d(-0.1, 1.0 - 1e-3, -0.1), Vector3d(1.1, 1.1, 1.1))
    # begin, end, range, dist, rotCenter, rotAxis, angle, ease_ratio=0.2
    sim.add_motion(0, 1.0, DBC_range, Vector3d(0.5, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0)
    sim.add_motion(1.0, 2.0, DBC_range, Vector3d(-0.5, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0)

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
        # opt params (trajectory, force)
        opt_param = "trajectory"
        constrain_type = "hard"
        opt_med = "GN"

        opt = Drivers.LoopyOpt(sim, opt_param, constrain_type, opt_med)
        opt.init_med = "solve"
        opt.n_epoch = 50
        opt.epsilon = 1e-4
        opt.p = 2
        opt.alpha = 1.0

        opt.initialize()
        opt.run()

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

    membEMult = 1
    bendEMult = 1

    sim.dt = 0.02
    sim.frame_num = 100
    sim.withCollision = False

    sim.k_wind = 5e4
    sim.wind_dir = Vector3d(1.0, 0, 0)
    
    sim.add_shell_3D("input/square11x16.obj", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)

    # DBC_bbox_min, DBC_bbox_max, idx_range
    corner1 = sim.set_DBC(Vector3d(-0.1, -1.0, -0.1), Vector3d(1e-3, 1e-3, 1.1))
    corner2 = sim.set_DBC(Vector3d(-0.1, 1 - 1e-3, -0.1), Vector3d(1e-3, 1.1, 1.1))

    if strain_limit:
        # iso
        sim.initialize(cloth_material, True, membEMult, bendEMult) # 0 means having gravity
    else:
        # iso, no strain limit
        sim.initialize(cloth_material, False, membEMult, bendEMult)
    sim.initialize_OIPC(1e-3, 0)

    sim.load_initial_state("output/" + sys.argv[0].split('.')[0] + "/forward/", 60)

    if not b_opt:
        sim.run()
    else:
        # opt params (trajectory, force)
        opt_param = sys.argv[1]
        constrain_type = sys.argv[2]
        opt_med = sys.argv[3]

        opt = Drivers.LoopyOpt(sim, opt_param, constrain_type, opt_med)

        opt.b_debug = False

        if opt_param == "force":
            opt.n_epoch = 100
            opt.epsilon = 10
            opt.alpha = 1e-8
        elif opt_param == "trajectory":
            if len(sys.argv) > 4:
                opt.init_med = sys.argv[4]
                opt.load_path = "output/" + sys.argv[0].split('.')[0] + "/trajectory_soft_GN_load_300/"
                opt.load_epoch = 299
            opt.n_epoch = 500
            opt.epsilon = 0

        opt.initialize()
        opt.run()

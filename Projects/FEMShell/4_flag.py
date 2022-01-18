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

    if not b_opt:
        sim.run()
    else:
        # opt params (trajectory, force)
        opt_param = sys.argv[1]
        constrain_type = sys.argv[2]
        opt_med = sys.argv[3]

        opt = Drivers.LoopyOpt(sim, opt_param, constrain_type, opt_med)
        if len(sys.argv) > 4:
            opt.init_med = sys.argv[4]
        opt.load_path = "output/" + sys.argv[0].split('.')[0] + "/trajectory.txt"

        if opt_param == "force":
            opt.n_epoch = 50
            opt.epsilon = 1e-4
            opt.alpha = 1e-5
        elif opt_param == "trajectory":
            opt.n_epoch = 300
            opt.epsilon = -1

        opt.p = 2
        opt.minmax = False

        opt.use_cg = False
        opt.cg_iter_ratio = 0.05

        opt.initialize()
        opt.run()

        # alternate
        # opt = Drivers.LoopyOpt(sim, "force", "soft", "GD")
        # opt.n_alternate = 2
        # opt.n_force_epoch = 50
        # opt.n_trajectory_epoch = 100

        # opt.initialize()
        # opt.alternate()

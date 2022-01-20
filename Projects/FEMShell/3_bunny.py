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

    sim.dt = 0.01
    sim.frame_num = 90
    sim.withCollision = False
    
    sim.add_object_3D("../FEMShell/input/bunny_240.mesh", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(3, 3, 3))
    DBC_range = sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 0.02, 1.1))
    # DBC_range = sim.set_DBC(Vector3d(-0.1, 0.9, -0.1), Vector3d(1.1, 1.1, 1.1))

    # for i in range(10):
    #     sim.add_motion(2.0 * i, 2.0 * i + 1.0, DBC_range, Vector3d(-0.5, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0)
    #     sim.add_motion(2.0 * i + 1.0, 2.0 * (i + 1), DBC_range, Vector3d(0.5, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0)

    if strain_limit:
        # iso
        sim.initialize(cloth_material, True, membEMult, bendEMult) # 0 means having gravity
    else:
        # iso, no strain limit
        sim.initialize(cloth_material, False, membEMult, bendEMult)

    sim.initialize_added_objects(Vector3d(0, 0, 1.2), 1e3, 5e4, 0.4)
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
            opt.n_epoch = 100
            opt.epsilon = 1
            opt.alpha = 1
        elif opt_param == "trajectory":
            opt.n_epoch = 300
            opt.epsilon = 0

            opt.use_cg = True
            opt.cg_iter = 10000
            opt.cg_tol = 1e-3

        opt.initialize()
        opt.run()

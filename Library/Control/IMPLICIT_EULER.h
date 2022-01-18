#pragma once

#include <FEM/Shell/INC_POTENTIAL.h>
#include <Control/WIND.h>

#include <deque>


namespace JGSL {

//#define VERBOSE

template <class T, int dim, bool KL, bool elasticIPC, bool flow>
void LineSearch(
    MESH_ELEM<dim - 1>& Elem,
    const std::vector<VECTOR<int, 2>>& seg,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    const std::map<std::pair<int, int>, int>& edge2tri,
    const std::vector<VECTOR<int, 4>>& edgeStencil,
    const std::vector<VECTOR<T, 3>>& edgeInfo,
    const T thickness, T bendingStiffMult,
    const VECTOR<T, 4>& fiberStiffMult,
    const VECTOR<T, 3>& fiberLimit,
    VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s, 
    const std::vector<T>& b, 
    T h, T NewtonTol,
    bool withCollision,
    T dHat2, const VECTOR<T, 3>& kappaVec,
    T mu, T epsv2,
    const std::vector<int>& compNodeRange,
    const std::vector<T>& muComp,
    bool staticSolve,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    MESH_ELEM<dim>& tet,
    MESH_ELEM_ATTR<T, dim>& tetAttr,
    FIXED_COROTATED<T, dim>& tetElasticityAttr,
    const std::vector<VECTOR<int, 2>>& rod,
    const std::vector<VECTOR<T, 3>>& rodInfo,
    const std::vector<VECTOR<int, 3>>& rodHinge,
    const std::vector<VECTOR<T, 3>>& rodHingeInfo,
    const std::vector<VECTOR<int, 3>>& stitchInfo,
    const std::vector<T>& stitchRatio,
    T k_stitch,
    const std::vector<int>& particle,
    const std::string& outputFolder,
    std::vector<T>& sol,
    std::vector<bool>& DBCb,
    MESH_NODE<T, dim>& Xn,
    MESH_NODE<T, dim>& Xtilde,
    std::vector<VECTOR<int, dim + 1>>& constraintSet,
    std::vector<VECTOR<T, 2>>& stencilInfo,
    std::vector<VECTOR<int, 2>>& constraintSetPTEE,
    T kappa[],
    std::vector<int>& boundaryNode,
    std::vector<VECTOR<int, 2>>& boundaryEdge,
    std::vector<VECTOR<int, 3>>& boundaryTri,
    std::map<int, std::set<int>>& NNExclusion,
    std::vector<T>& BNArea,
    std::vector<T>& BEArea,
    std::vector<T>& BTArea,
    VECTOR<int, 2>& codimBNStartInd,
    std::vector<VECTOR<int, dim + 1>>& fricConstraintSet,
    std::vector<Eigen::Matrix<T, dim - 1, 1>>& closestPoint,
    std::vector<Eigen::Matrix<T, dim, dim - 1>>& tanBasis,
    std::vector<T>& normalForce,
    T& DBCStiff, T& Eprev, T& alpha, T& feasibleAlpha)
{
    // line search
    MESH_NODE<T, dim> Xprev;
    Append_Attribute(X, Xprev);
    alpha = 1.0;
    T E;
    bool valid = true; // whether the current iterate x is feasible
    if (kappa_s[0] > 0 || fiberStiffMult[0] > 0 || tet.size > 0) {
        do {
            X.Join(Xprev).Par_Each([&](int id, auto data) {
                auto &[x, xprev] = data;
                x[0] = xprev[0] + alpha * sol[id * dim];
                x[1] = xprev[1] + alpha * sol[id * dim + 1];
                if constexpr (dim == 3) {
                    x[2] = xprev[2] + alpha * sol[id * dim + 2];
                }
            });
            valid = Compute_IncPotential<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, 
                fiberLimit, s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, elasticityAttr, 
                false, constraintSet, stencilInfo, dHat2, kappa, staticSolve, b, 
                tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, 
                stitchInfo, stitchRatio, k_stitch, E);
            if (!valid) {
                alpha /= 2.0;
            }
        } while (!valid);
        Xprev.deep_copy_to(X);
#ifdef VERBOSE
        printf("inextensibility feasible alpha = %le\n", alpha);
#endif
    }
    if (withCollision) {
        Compute_Intersection_Free_StepSize<T, dim, false, elasticIPC>(X, boundaryNode, boundaryEdge, boundaryTri, 
            particle, rod, NNExclusion, codimBNStartInd, DBCb, sol, thickness, alpha); // CCD
#ifdef VERBOSE
        printf("intersection free step size = %le\n", alpha);
#endif
    }
    feasibleAlpha = alpha;
    do {
        X.Join(Xprev).Par_Each([&](int id, auto data) {
            auto &[x, xprev] = data;
            x[0] = xprev[0] + alpha * sol[id * dim];
            x[1] = xprev[1] + alpha * sol[id * dim + 1];
            if constexpr (dim == 3) {
                x[2] = xprev[2] + alpha * sol[id * dim + 2];
            }
        });
        valid = Compute_IncPotential<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
            s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, elasticityAttr, 
            false, constraintSet, stencilInfo, dHat2, kappa, staticSolve, b, 
            tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, 
            stitchInfo, stitchRatio, k_stitch, E);
        if (valid) {
            if (withCollision) {
                Compute_Constraint_Set<T, dim, false, elasticIPC>(X, nodeAttr, boundaryNode, boundaryEdge, boundaryTri, 
                    particle, rod, NNExclusion, BNArea, BEArea, BTArea, codimBNStartInd, DBCb, dHat2, thickness, false, constraintSet, constraintSetPTEE, stencilInfo);
                if (!constraintSet.empty()) {
                    T minDist2;
                    std::vector<T> dist2;
                    Compute_Min_Dist2<T, dim, elasticIPC>(X, constraintSet, thickness, dist2, minDist2);
                    if (minDist2 <= 0) {
                        std::cout << "safe guard backtrack!" << std::endl;
                        alpha /= 2;
                        E = Eprev + 1;
                        continue;
                    }
                }
            }
            valid = Compute_IncPotential<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, 
                fiberLimit, s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, elasticityAttr, 
                withCollision, constraintSet, stencilInfo, dHat2, kappa, staticSolve, b, 
                tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, 
                stitchInfo, stitchRatio, k_stitch, E);
            if (valid && withCollision && mu > 0) {
                Compute_Friction_Potential(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, E);
            }
            if (valid && DBCStiff) {
                Compute_DBC_Energy(X, nodeAttr, DBC, DBCStiff, E);
            }
        }
        alpha /= 2.0;
#ifdef VERBOSE
        printf("E %le, Eprev %le, alpha %le, valid %d\n", E, Eprev, alpha * 2, valid ? 1 : 0);
#endif
    } while (E > Eprev || !valid);
#ifdef VERBOSE
    printf("[line-search] alpha = %le\n", alpha * 2.0);
#endif
    Eprev = E;
}

template <class T, int dim, bool KL=false, bool elasticIPC=false, bool flow=false>
int Step(
    int cur_step,
    MESH_ELEM<dim - 1>& Elem,
    const std::vector<VECTOR<int, 2>>& seg,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    const std::map<std::pair<int, int>, int>& edge2tri,
    const std::vector<VECTOR<int, 4>>& edgeStencil,
    const std::vector<VECTOR<T, 3>>& edgeInfo,
    const T thickness, T bendingStiffMult,
    const VECTOR<T, 4>& fiberStiffMult,
    const VECTOR<T, 3>& fiberLimit,
    VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s, 
    const std::vector<T>& b, T k_wind, const VECTOR<T, dim>& wind_dir,
    MESH_NODE<T, dim>& control_force,
    T h, T NewtonTol,
    bool withCollision,
    T dHat2, VECTOR<T, 3>& kappaVec,
    T mu, T epsv2, int fricIterAmt,
    const std::vector<int>& compNodeRange,
    const std::vector<T>& muComp,
    bool staticSolve,
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    CSR_MATRIX<T>& M, // mass matrix
    MESH_ELEM_ATTR<T, dim - 1>& elemAttr,
    FIXED_COROTATED<T, dim - 1>& elasticityAttr,
    MESH_ELEM<dim>& tet,
    MESH_ELEM_ATTR<T, dim>& tetAttr,
    FIXED_COROTATED<T, dim>& tetElasticityAttr,
    const std::vector<VECTOR<int, 2>>& rod,
    const std::vector<VECTOR<T, 3>>& rodInfo,
    const std::vector<VECTOR<int, 3>>& rodHinge,
    const std::vector<VECTOR<T, 3>>& rodHingeInfo,
    const std::vector<VECTOR<int, 3>>& stitchInfo,
    const std::vector<T>& stitchRatio,
    T k_stitch,
    const std::vector<int>& particle,
    const std::string& outputFolder)
{
    Eigen::setNbThreads(1);
    TIMER_FLAG("implicitEuler");

    T kappa[] = {kappaVec[0], kappaVec[1], kappaVec[2]}; // dumb pybind does not support c array

    // record Xn and compute predictive pos Xtilde ==================================================================================================
    // input: nodeAttr[v], bodyForce, M
    // output: Xn, Xtilde
    std::vector<bool> DBCb(X.size, false);
    DBC.Each([&](int id, auto data) {
        auto &[dbcI] = data;
        int vI = dbcI(0);
        DBCb[dbcI(0)] = true; // bool array cannot be written in parallel by entries
    });

    MESH_NODE<T, dim> Xn, Xtilde;
    if (!staticSolve) {
        Append_Attribute(X, Xn);
        Append_Attribute(X, Xtilde);

        std::vector<T> a;
        std::vector<T> f_ext(b);
        control_force.Par_Each([&](int idx, auto data) {
            auto &[f] = data;
            for (int d = 0; d < dim; ++d) {
                f_ext[idx * dim + d] += f[d];
            }
        });
        if (k_wind > 0) {
            MESH_NODE<T, dim> wind_force;
            Append_Attribute(control_force, wind_force);
            Compute_Wind_Force<T, dim>(k_wind, wind_dir, DBCb, Elem, elasticityAttr, Xn, wind_force);
            wind_force.Par_Each([&](int idx, auto data) {
                auto &[f] = data;
                for (int d = 0; d < dim; ++d) {
                    f_ext[idx * dim + d] += f[d];
                }
            });
        }
        if (!Solve_Direct(M, f_ext, a)) {
            std::cout << "mass matrix factorization failed!" << std::endl;
            exit(-1);
        }

        Xtilde.Join(nodeAttr).Par_Each([&](int id, auto data) {
            auto &[x, x0, v, g, m] = data;
            if constexpr (flow) {
                v.setZero();
            }
            x[0] += h * v[0] + h * h * a[id * dim];
            x[1] += h * v[1] + h * h * a[id * dim + 1];
            if constexpr (dim == 3) {
                x[2] += h * v[2] + h * h * a[id * dim + 2];
            }
        });
#ifdef VERBOSE
        std::cout << "Xn and Xtilde prepared" << std::endl;
#endif
    }

    // compute contact primitives and areas ========================================================================================================
    // input: X, Elem, tet
    // output: boundaryNode/Edge/Tri, BNArea, BEArea, BTArea, codimBNStartInd, codimBNStartInd
    std::vector<int> boundaryNode;
    std::vector<VECTOR<int, 2>> boundaryEdge;
    std::vector<VECTOR<int, 3>> boundaryTri;
    std::vector<T> BNArea, BEArea, BTArea;
    VECTOR<int, 2> codimBNStartInd;
    std::map<int, std::set<int>> NNExclusion;
    if (withCollision) {
        if constexpr (dim == 2) {
            //TODO
        }
        else {
            BASE_STORAGE<int> TriVI2TetVI;
            BASE_STORAGE<VECTOR<int, 3>> Tri;
            Find_Surface_TriMesh<T, false>(X, tet, TriVI2TetVI, Tri);
            Append_Attribute(Elem, Tri);

            Find_Surface_Primitives_And_Compute_Area(X, Tri, boundaryNode, boundaryEdge, boundaryTri,
                BNArea, BEArea, BTArea);
            
            boundaryEdge.insert(boundaryEdge.end(), seg.begin(), seg.end());
            for (const auto& segI : seg) {
                boundaryNode.emplace_back(segI[0]);
                boundaryNode.emplace_back(segI[1]);
                //TODO: handle duplicates
            }

            boundaryEdge.insert(boundaryEdge.end(), rod.begin(), rod.end());
            BEArea.reserve(boundaryEdge.size());
            std::map<int, T> rodNodeArea;
            int segIInd = 0;
            for (const auto& segI : rod) {
                const VECTOR<T, dim>& v0 = std::get<0>(X.Get_Unchecked(segI[0]));
                const VECTOR<T, dim>& v1 = std::get<0>(X.Get_Unchecked(segI[1]));
                BEArea.emplace_back((v0 - v1).length() * M_PI * rodInfo[segIInd][2] / 6); // 1/6 of the cylinder surface participate in one contact
                
                rodNodeArea[segI[0]] += BEArea.back() / 2;
                rodNodeArea[segI[1]] += BEArea.back() / 2;

                BEArea.back() /= 2; // due to PE approx of \int_E PP and EE approx of \int_E PE
                
                ++segIInd;
            }
            codimBNStartInd[0] = boundaryNode.size();
            boundaryNode.reserve(boundaryNode.size() + rodNodeArea.size());
            BNArea.reserve(BNArea.size() + rodNodeArea.size());
            for (const auto& nodeI : rodNodeArea) {
                boundaryNode.emplace_back(nodeI.first);
                BNArea.emplace_back(nodeI.second);
            }
            codimBNStartInd[1] = boundaryNode.size();

            for (const auto& vI : particle) {
                boundaryNode.emplace_back(vI);
            }

            for (const auto& stitchI : stitchInfo) {
                NNExclusion[stitchI[0]].insert(stitchI[1]);
                NNExclusion[stitchI[0]].insert(stitchI[2]);
                NNExclusion[stitchI[1]].insert(stitchI[0]);
                NNExclusion[stitchI[2]].insert(stitchI[0]);
            }
        }
    }
#ifdef VERBOSE
    std::cout << "surface primitives found" << std::endl;
#endif

    // set Dirichlet boundary condition on X ====================================================================================================
    // input: X, DBC
    // output: DBCStiff, DBCb, DBCb_fixed
    T DBCAlpha = 1;
    std::vector<bool> DBCb_fixed(X.size, false); // this masks nodes that are fixed (DBC with 0 velocity)
    std::vector<T> DBCDisp(X.size * dim, T(0));
    DBC.Each([&](int id, auto data) {
        auto &[dbcI] = data;
        int vI = dbcI(0);
        const VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(vI));
        
        DBCDisp[vI * dim] = dbcI(1) - x(0);
        DBCDisp[vI * dim + 1] = dbcI(2) - x(1);
        if constexpr (dim == 3) {
            DBCDisp[vI * dim + 2] = dbcI(3) - x(2);
            if (!(DBCDisp[vI * dim] || DBCDisp[vI * dim + 1] || DBCDisp[vI * dim + 2])) {
                DBCb_fixed[vI] = true;
            }
        }
        else {
            if (!(DBCDisp[vI * dim] || DBCDisp[vI * dim + 1])) {
                DBCb_fixed[vI] = true;
            }
        }

    });
    // determine DBC alpha: CCD --------------------------------------------------------------------------------------------
    if (withCollision) {
        Compute_Intersection_Free_StepSize<T, dim, false, elasticIPC>(X, boundaryNode, boundaryEdge, boundaryTri, 
            particle, rod, NNExclusion, codimBNStartInd, DBCb, DBCDisp, thickness, DBCAlpha);
#ifdef VERBOSE
        printf("DBCAlpha under contact: %le\n", DBCAlpha);
#endif
    }
    // determine DBC alpha: strain limiting / fibber (aniso) / tet non degenerate ------------------------------------------
    if (kappa_s[0] > 0 || fiberStiffMult[0] > 0) {
        if (Elem.size) {
            T maxs, avgs, minc, avgc;
            Compute_Max_And_Avg_Stretch(Elem, staticSolve ? 1.0 : h, fiberStiffMult, DBCb, X, nodeAttr, elemAttr, elasticityAttr, maxs, avgs, minc, avgc);
#ifdef VERBOSE
            printf("maxs = %le, avgs = %le, minc = %le, avgc = %le\n", maxs, avgs, minc, avgc);
#endif
        }
        
        bool valid = false;
        T E;
        do {
            DBC.Each([&](int id, auto data) {
                auto &[dbcI] = data;
                VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(dbcI(0)));
                const VECTOR<T, dim> &xn = std::get<0>(Xn.Get_Unchecked(dbcI(0)));

                x(0) = xn[0] + DBCAlpha * (dbcI(1) - xn[0]);
                x(1) = xn[1] + DBCAlpha * (dbcI(2) - xn[1]);
                if constexpr (dim == 3) {
                    x(2) = xn[2] + DBCAlpha * (dbcI(3) - xn[2]);
                }
            });
            valid = true;
            if (kappa_s[0] > 0) {
                valid &= Compute_Inextensibility_Energy(Elem, staticSolve ? 1.0 : h, s, sHat, kappa_s, 
                    DBCb, X, nodeAttr, elemAttr, elasticityAttr, E);
            }
            if (fiberStiffMult[0] > 0) {
                valid &= Check_Fiber_Feasibility(Elem, staticSolve ? 1.0 : h, fiberLimit, DBCb, X, nodeAttr, elemAttr, elasticityAttr);
            }
            if (tet.size > 0) {
                std::vector<int> degenerate(tet.size);
                tetElasticityAttr.Par_Each([&](const int i, auto data) {
                    auto& [F, vol, lambda, mu] = data;

                    degenerate[i] = (F.determinant() <= 0);
                });
                int hasDegeneracy = std::accumulate(degenerate.begin(), degenerate.end(), 0);
                if (hasDegeneracy) {
                    valid = false;
                }
            }
            if (!valid) {
                DBCAlpha /= 2.0;
            }
#ifdef VERBOSE
            printf("backtracking DBCAlpha for inextensibility: %le\n", DBCAlpha);
#endif
        } while (!valid);
#ifdef VERBOSE
        printf("DBCAlpha under inextensibility: %le\n", DBCAlpha);
#endif
        Xn.deep_copy_to(X);
    }
    // determine DBC update type -----------------------------------------------------------------------
    // DBC step size == 1: hard constrain
    // DBC step size < 1: soft constrain
    T DBCStiff = 0, DBCPenaltyXn = 0;
    if (DBCAlpha == 1) {
        DBC.Each([&](int id, auto data) {
            auto &[dbcI] = data;
            VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(dbcI(0)));
            x(0) = dbcI(1);
            x(1) = dbcI(2);
            if constexpr (dim == 3) {
                x(2) = dbcI(3);
            }
        });
#ifdef VERBOSE
        printf("DBC handled\n");
#endif
    }
    else {
#ifdef VERBOSE
        printf("moved DBC by %le, turn on Augmented Lagrangian\n", DBCAlpha);
#endif
        DBCStiff = 1e6;
        Compute_DBC_Dist2(Xn, DBC, DBCPenaltyXn);
    }

    // Newton loop ============================================================================================================================
    int PNIter = 0;
    T L2Norm = 0;
    bool useGD = false;
    // linear solver variables
    CSR_MATRIX<T> sysMtr;
    std::vector<T> rhs(X.size * dim), sol(X.size * dim);
    // construct collision set and info ----------------------------------------------------------------------------------
    // input: X, nodeAttr[x0] (for EE), bounaryNode/Edge/Tri, area, params
    // output: constraintSet, constraintSetPTEE, stencilInfo
    std::vector<VECTOR<int, dim + 1>> constraintSet, constraintSet_prev;
    std::vector<T> dist2_prev;
    std::vector<VECTOR<int, 2>> constraintSetPTEE;
    std::vector<VECTOR<T, 2>> stencilInfo;
    // friction:
    std::vector<VECTOR<int, dim + 1>> fricConstraintSet;
    std::vector<Eigen::Matrix<T, dim - 1, 1>> closestPoint;
    std::vector<Eigen::Matrix<T, dim, dim - 1>> tanBasis;
    std::vector<T> normalForce;
#ifdef VERBOSE
    printf("computing initial energy\n");
#endif
    if (withCollision) {
        Compute_Constraint_Set<T, dim, false, elasticIPC>(X, nodeAttr, boundaryNode, boundaryEdge, boundaryTri, 
            particle, rod, NNExclusion, BNArea, BEArea, BTArea, codimBNStartInd, DBCb, dHat2, thickness, false, constraintSet, constraintSetPTEE, stencilInfo);
        if (mu > 0 || (muComp.size() && muComp.size() == compNodeRange.size() * compNodeRange.size())) {
            Compute_Friction_Basis<T, dim, elasticIPC>(X, constraintSet, stencilInfo, fricConstraintSet, closestPoint, tanBasis, normalForce, dHat2, kappa, thickness);
            if (muComp.size() && muComp.size() == compNodeRange.size() * compNodeRange.size()) {
                Compute_Friction_Coef<T, dim>(fricConstraintSet, compNodeRange, muComp, normalForce, mu); 
                // mu will be set to 1, normalForce will be multipled with different mu's in muComp
            }
        }
    }
    // start-up energy
    T Eprev;
    // it includes contact
    Compute_IncPotential<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
        s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, elasticityAttr, 
        withCollision, constraintSet, stencilInfo, dHat2, kappa, staticSolve, b, 
        tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, 
        stitchInfo, stitchRatio, k_stitch, Eprev);
    if (withCollision && mu > 0) {
        Compute_Friction_Potential(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, Eprev);
    }
    if (DBCStiff) {
        Compute_DBC_Energy(X, nodeAttr, DBC, DBCStiff, Eprev);
    }
#ifdef VERBOSE
    printf("entering Newton loop\n");
#endif
    std::deque<T> resRecord, MDBCProgressI;
    int fricIterI = 0;
    std::vector<VECTOR<T, 2>> strain_prev;

    do {
        // compute gradient
        Compute_IncPotential_Gradient<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
            s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, 
            withCollision, constraintSet, stencilInfo, dHat2, kappa, staticSolve, b, elasticityAttr, 
            tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo,
            stitchInfo, stitchRatio, k_stitch);
        if (withCollision && mu > 0) {
            Compute_Friction_Gradient(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, nodeAttr);
        }

        if (DBCStiff) {
            Compute_DBC_Gradient(X, nodeAttr, DBC, DBCStiff);
            for (int vI = 0; vI < DBCb_fixed.size(); ++vI) {
                if (DBCb_fixed[vI]) {
                    std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(vI)).setZero();
                }
            }
        }
        else {
            // project rhs for Dirichlet boundary condition
            DBC.Par_Each([&](int id, auto data) {
                auto &[dbcI] = data;
                std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(dbcI(0))).setZero();
            });
#ifdef VERBOSE
            std::cout << "project rhs for Dirichlet boundary condition " << DBC.size << std::endl;
#endif
        }

        nodeAttr.Par_Each([&](int id, auto data) {
            auto &[x0, v, g, m] = data;
            rhs[id * dim] = -g[0];
            rhs[id * dim + 1] = -g[1];
            if constexpr (dim == 3) {
                rhs[id * dim + 2] = -g[2];
            }
        });

        // compute Hessian
        if (!useGD) {
            // it includes contact, friction, and DBCStiff
            Compute_IncPotential_Hessian<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
                s, sHat, kappa_s, DBC, DBCb, DBCb_fixed, DBCStiff, X, Xn, Xtilde, nodeAttr, M, elemAttr, 
                withCollision, constraintSet, stencilInfo, fricConstraintSet, closestPoint, tanBasis, normalForce,
                dHat2, kappa, mu, epsv2, staticSolve, b, elasticityAttr, 
                tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, 
                stitchInfo, stitchRatio, k_stitch, true, sysMtr);
        }

        // compute search direction
        {
            TIMER_FLAG("linearSolve");

            if (useGD) {
#ifdef VERBOSE
                printf("use gradient descent\n");
#endif
                std::memcpy(sol.data(), rhs.data(), sizeof(T) * rhs.size());
            }
            else {
#ifdef AMGCL_LINEAR_SOLVER
                // AMGCL
                std::memset(sol.data(), 0, sizeof(T) * sol.size());
                Solve(sysMtr, rhs, sol, 1.0e-5, 1000, Default_FEM_Params<dim>(), true);
#else
                // direct factorization
                bool solverSucceed = Solve_Direct(sysMtr, rhs, sol);
                // for (int vI = 0; vI < X.size; ++vI) {
                //     for (int d = 0; d < dim; ++d) {
                //         if (std::isnan(rhs[vI * dim + d])) {
                //             printf("rhs nan %d\n", vI * dim + d);
                //             exit(-1);
                //         }
                //         if (std::isnan(sol[vI * dim + d])) {
                //             printf("sol nan %d\n", vI * dim + d);
                //             solverSucceed = false;
                //         }
                //     }
                // }
                if(!solverSucceed) {
                    FILE *out = fopen((outputFolder + "/Hessian_info.txt").c_str(), "a+");
                    fprintf(out, "Hessian not SPD in PNIter%d\n", PNIter);
                    fclose(out);
                    
                    // Compute_IncPotential_Hessian<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
                    //     s, sHat, kappa_s, DBC, DBCb, DBCb_fixed, DBCStiff, X, Xn, Xtilde, nodeAttr, M, elemAttr, 
                    //     withCollision, constraintSet, fricConstraintSet, closestPoint, tanBasis, normalForce,
                    //     dHat2, kappa, mu, epsv2, staticSolve, b, elasticityAttr, 
                    //     tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, true, sysMtr);
                    // if(!Solve_Direct(sysMtr, rhs, sol)) {
                        useGD = true;
                        printf("use gradient descent\n");
                        std::memcpy(sol.data(), rhs.data(), sizeof(T) * rhs.size());
                        // for (int vI = 0; vI < X.size; ++vI) {
                        //     for (int d = 0; d < dim; ++d) {
                        //         sol[vI * dim + d] /= sysMtr.Get_Matrix().coeff(vI * dim + d, vI * dim + d);
                        //     }
                        // }
                    // }
                }
#endif
            }
        }

        T alpha, feasibleAlpha;
        // output feasibleAlpha
        LineSearch<T, dim, KL, elasticIPC, flow>(Elem, seg, DBC, edge2tri, edgeStencil, edgeInfo, 
            thickness, bendingStiffMult, fiberStiffMult, fiberLimit, s, sHat, kappa_s, 
            b, h, NewtonTol, withCollision, dHat2, kappaVec, mu, epsv2, compNodeRange, muComp, 
            staticSolve, X, nodeAttr, M, elemAttr, elasticityAttr, tet, tetAttr, tetElasticityAttr, 
            rod, rodInfo, rodHinge, rodHingeInfo, stitchInfo, stitchRatio, k_stitch, particle, outputFolder,
            sol, DBCb, Xn, Xtilde, constraintSet, stencilInfo, constraintSetPTEE, kappa, 
            boundaryNode, boundaryEdge, boundaryTri, NNExclusion, BNArea, BEArea, BTArea, codimBNStartInd, 
            fricConstraintSet, closestPoint, tanBasis, normalForce, DBCStiff, Eprev, alpha, feasibleAlpha);

        if constexpr (!elasticIPC) {
            if (constraintSet_prev.size()) {
                T minDist2;
                std::vector<T> curdist2_prev;
                Compute_Min_Dist2<T, dim, elasticIPC>(X, constraintSet_prev, thickness, curdist2_prev, minDist2);
                bool updateKappa = false;
                for (int i = 0; i < curdist2_prev.size(); ++i) {
                    if (dist2_prev[i] < 1e-18 && curdist2_prev[i] < dist2_prev[i]) {
                        updateKappa = true;
                        break;
                    }
                }
                if (updateKappa && kappa[0] < kappa[1]) {
                    kappa[0] *= 2;
                    kappaVec[0] *= 2;

                    Compute_IncPotential<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
                        s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, elasticityAttr, 
                        withCollision, constraintSet, stencilInfo, dHat2, kappa, staticSolve, b, 
                        tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, 
                        stitchInfo, stitchRatio, k_stitch, Eprev);
                    if (withCollision && mu > 0) {
                        Compute_Friction_Potential(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, Eprev);
                    }
                    if (DBCStiff) {
                        Compute_DBC_Energy(X, nodeAttr, DBC, DBCStiff, Eprev);
                    }
                }
            }
        }

        std::vector<T> dist2;
        if (constraintSet.size()) {
            T minDist2;
            Compute_Min_Dist2<T, dim, elasticIPC>(X, constraintSet, thickness, dist2, minDist2);
#ifdef VERBOSE
            printf("minDist2 = %le, kappa = %le (max %le)\n", minDist2, kappa[0], kappa[1]);
#endif
        }
        constraintSet_prev = constraintSet;
        dist2_prev = dist2;

        if (kappa_s[0] > 0) { //TODO: for lower bound
            std::vector<VECTOR<T, 2>> strain;
            Compute_Inextensibility(Elem, staticSolve ? 1.0 : h, s, sHat, kappa_s, DBCb, X, nodeAttr, elemAttr, elasticityAttr, strain);
            T minSLSlack = 1;
            for (int i = 0; i < strain.size(); ++i) {
                T SLSlack = s[0] - strain[i][0];
                if (minSLSlack > SLSlack) {
                    minSLSlack = SLSlack;
                }
            }
#ifdef VERBOSE
            printf("minSLSlack = %le, kappa_s = %le\n", minSLSlack, kappa_s[0]);
#endif
            
            bool updateKappa_s = 0;
            if (strain_prev.size()) {
                for (int i = 0; i < strain.size(); ++i) {
                    if ((s[0] - strain_prev[i][0] < 1.0e-4 * (s[0] - sHat[0])) && 
                        (s[0] - strain[i][0] < 1.0e-4 * (s[0] - sHat[0])))
                    {
                        updateKappa_s = true;
                        break;
                    }
                }
            }
            if (updateKappa_s && kappa_s[0] < 1e5) {
                kappa_s[0] *= 2;

                Compute_IncPotential<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
                    s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, elasticityAttr, 
                    withCollision, constraintSet, stencilInfo, dHat2, kappa, staticSolve, b, 
                    tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, 
                    stitchInfo, stitchRatio, k_stitch, Eprev);
                if (withCollision && mu > 0) {
                    Compute_Friction_Potential(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, Eprev);
                }
                if (DBCStiff) {
                    Compute_DBC_Energy(X, nodeAttr, DBC, DBCStiff, Eprev);
                }
            }
            strain_prev = strain;
        }

        // stopping criteria
        T maxRes = 0.0, avgResMag = 0.0;
        L2Norm = 0.0;
        for (int i = 0; i < sol.size() / dim; ++i) {
            T curMag = 0;
            for (int j = 0; j < dim; ++j) {
                curMag += sol[i * dim + j] * sol[i * dim + j];
                if (maxRes < std::abs(sol[i * dim + j])) {
                    maxRes = std::abs(sol[i * dim + j]);
                }
                L2Norm += sol[i * dim + j] * sol[i * dim + j];
            }
            avgResMag += std::sqrt(curMag);
        }
        avgResMag /= sol.size() / dim - DBC.size;
        avgResMag /= (staticSolve ? 1 : h);
        maxRes /= (staticSolve ? 1 : h);
        L2Norm = std::sqrt(L2Norm / (sol.size() / dim - DBC.size));
        L2Norm /= (staticSolve ? 1 : h);

#ifdef VERBOSE
        printf("[PNIter%d]: Newton res = %le, tol = %le\n", PNIter, L2Norm, NewtonTol);
#endif
        PNIter += 1;

        FILE *out = fopen((outputFolder + "/residual.txt").c_str(), "a+");
        fprintf(out, "%d %le %le %le %le\n", PNIter, avgResMag, maxRes, Eprev, L2Norm);
        fclose(out);

        resRecord.push_back(L2Norm);
        if (resRecord.size() > 3) {
            resRecord.pop_front();
        }
        L2Norm = *std::max_element(resRecord.begin(), resRecord.end());

        if (useGD) {
            L2Norm = NewtonTol * 10; // ensures not exit Newton loop
        }

        if (alpha * 2 < 1e-8 && feasibleAlpha > 1e-8) {
            if (!useGD) {
                useGD = true;
                Eigen::VectorXd pe(sol.size()), mge(rhs.size());
                std::memcpy(pe.data(), sol.data(), sizeof(T) * sol.size());
                std::memcpy(mge.data(), rhs.data(), sizeof(T) * rhs.size());
                printf("\t-gdotp = %le, -gpcos = %le\n", mge.dot(pe), 
                    mge.dot(pe) / std::sqrt(mge.squaredNorm() * pe.squaredNorm()));
                printf("\tlinear solve relErr = %le\n", 
                    std::sqrt((sysMtr.Get_Matrix() * pe - mge).squaredNorm() / mge.squaredNorm()));
            }
            else {
#ifdef VERBOSE
                printf("GD tiny step size!\n");
#endif
            }
        }
        else {
            useGD = false;
        }

        if (DBCStiff) {
            T penaltyCur = 0;
            Compute_DBC_Dist2(X, DBC, penaltyCur);
            T progress = 1 - std::sqrt(penaltyCur / DBCPenaltyXn);
#ifdef VERBOSE
            printf("MDBC progress: %le, DBCStiff %le\n", progress, DBCStiff);
#endif

            MDBCProgressI.emplace_back(progress);
            if (MDBCProgressI.size() > 4) {
                MDBCProgressI.pop_front();
            }
            bool stopMDBC = false;
            if (MDBCProgressI.size() == 4) {
                T deltaProgress[3];
                for (int i = 0; i < 3; ++i) {
                    deltaProgress[i] = std::abs(MDBCProgressI[i] - MDBCProgressI[i + 1]);
                }
                if (PNIter > 30 && deltaProgress[0] < 0.001 && deltaProgress[1] < 0.001 && deltaProgress[2] < 0.001) {
                    stopMDBC = true;
                }
            }

            // if(progress < 0.99 && !stopMDBC) { // for character garment anim
            if(progress < 0.99) {
                //TODO: update Augmented Lagrangian parameters if necessary
                if (L2Norm < NewtonTol * 10) {
                    if (DBCStiff < 1e8) {
                        DBCStiff *= 2;
                        Compute_IncPotential<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
                            s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, elasticityAttr, 
                            withCollision, constraintSet, stencilInfo, dHat2, kappa, staticSolve, b, 
                            tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, 
                            stitchInfo, stitchRatio, k_stitch, Eprev);
                        if (withCollision && mu > 0) {
                            Compute_Friction_Potential(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, Eprev);
                        }
                        Compute_DBC_Energy(X, nodeAttr, DBC, DBCStiff, Eprev);
#ifdef VERBOSE
                        printf("updated DBCStiff to %le\n", DBCStiff);
#endif
                    }
                }

                L2Norm = NewtonTol * 10; // ensures not exit Newton loop
            }
            else {
                DBCStiff = 0;

                Compute_IncPotential<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, 
                    fiberLimit, s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, elasticityAttr, 
                    withCollision, constraintSet, stencilInfo, dHat2, kappa, staticSolve, b, 
                    tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, 
                    stitchInfo, stitchRatio, k_stitch, Eprev);
                if (withCollision && mu > 0) {
                    Compute_Friction_Potential(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, Eprev);
                }
#ifdef VERBOSE
                printf("DBC moved to target, turn off Augmented Lagrangian\n");
#endif
            }
        }

        if ((resRecord.size() == 3) && L2Norm <= NewtonTol) {
            ++fricIterI;
            if (fricIterI < fricIterAmt || fricIterAmt <= 0) {
                if (withCollision) {
                    if (mu > 0 || muComp.size() == compNodeRange.size() * compNodeRange.size()) {
                        Compute_Friction_Basis<T, dim, elasticIPC>(X, constraintSet, stencilInfo, fricConstraintSet, closestPoint, tanBasis, normalForce, dHat2, kappa, thickness);
                        if (muComp.size() == compNodeRange.size() * compNodeRange.size()) {
                            Compute_Friction_Coef<T, dim>(fricConstraintSet, compNodeRange, muComp, normalForce, mu); 
                            // mu will be set to 1, normalForce will be multipled with different mu's in muComp
                        }

                        // compute gradient
                        Compute_IncPotential_Gradient<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
                            s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, 
                            withCollision, constraintSet, stencilInfo, dHat2, kappa, staticSolve, b, elasticityAttr, 
                            tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo,
                            stitchInfo, stitchRatio, k_stitch);
                        Compute_Friction_Gradient(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, nodeAttr);
                        if (DBCStiff) {
                            Compute_DBC_Gradient(X, nodeAttr, DBC, DBCStiff);
                            for (int vI = 0; vI < DBCb_fixed.size(); ++vI) {
                                if (DBCb_fixed[vI]) {
                                    std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(vI)).setZero();
                                }
                            }
                        }
                        else {
                            // project rhs for Dirichlet boundary condition
                            DBC.Par_Each([&](int id, auto data) {
                                auto &[dbcI] = data;
                                std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(dbcI(0))).setZero();
                            });
                        }
                        nodeAttr.Par_Each([&](int id, auto data) {
                            auto &[x0, v, g, m] = data;
                            rhs[id * dim] = -g[0];
                            rhs[id * dim + 1] = -g[1];
                            if constexpr (dim == 3) {
                                rhs[id * dim + 2] = -g[2];
                            }
                        });
                        // compute Hessian
                        Compute_IncPotential_Hessian<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
                            s, sHat, kappa_s, DBC, DBCb, DBCb_fixed, DBCStiff, X, Xn, Xtilde, nodeAttr, M, elemAttr, 
                            withCollision, constraintSet, stencilInfo, fricConstraintSet, closestPoint, tanBasis, normalForce,
                            dHat2, kappa, mu, epsv2, staticSolve, b, elasticityAttr, 
                            tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, 
                            stitchInfo, stitchRatio, k_stitch, true, sysMtr);
                        // compute search direction
                        {
                            TIMER_FLAG("linearSolve");
#ifdef AMGCL_LINEAR_SOLVER
                            // AMGCL
                            std::memset(sol.data(), 0, sizeof(T) * sol.size());
                            Solve(sysMtr, rhs, sol, 1.0e-5, 1000, Default_FEM_Params<dim>(), true);
#else
                            // direct factorization
                            if(!Solve_Direct(sysMtr, rhs, sol)) {
                                FILE *out = fopen((outputFolder + "/Hessian_info.txt").c_str(), "a+");
                                fprintf(out, "Hessian not SPD in PNIter%d\n", PNIter);
                                fclose(out);
                                exit(-1);
                            }
#endif
                        }

                        L2Norm = 0.0;
                        for (int i = 0; i < sol.size() / dim; ++i) {
                            for (int j = 0; j < dim; ++j) {
                                L2Norm += sol[i * dim + j] * sol[i * dim + j];
                            }
                        }
                        L2Norm = std::sqrt(L2Norm / (sol.size() / dim - DBC.size));
                        L2Norm /= (staticSolve ? 1 : h);
#ifdef VERBOSE
                        printf("friction updated Newton res = %le, tol = %le\n", L2Norm, NewtonTol);
#endif
                        if (L2Norm > NewtonTol) {
                            Compute_IncPotential<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
                                s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, elasticityAttr, 
                                withCollision, constraintSet, stencilInfo, dHat2, kappa, staticSolve, b, 
                                tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, 
                                stitchInfo, stitchRatio, k_stitch, Eprev);
                            Compute_Friction_Potential(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, Eprev);
                            if (DBCStiff) {
                                Compute_DBC_Energy(X, nodeAttr, DBC, DBCStiff, Eprev);
                            }
                        }
                    }
                }
            }
        }

        if constexpr (flow) {
            if (!withCollision || constraintSet.empty()) {
                break;
            }
        }
    } while (((resRecord.size() < 3) || L2Norm > NewtonTol) && PNIter < 50); //TODO: newtonTol relative to bbox

    FILE *out = fopen((outputFolder + "/counter.txt").c_str(), "a+");
    fprintf(out, "%d", PNIter);
    if (withCollision) {
#ifdef VERBOSE
        printf("contact #: %lu\n", constraintSet.size());
#endif
        fprintf(out, " %lu", constraintSet.size());
    }
    fprintf(out, "\n");
    fclose(out);

    T maxs, avgs, minc, avgc;
    if (Elem.size) {
        Compute_Max_And_Avg_Stretch(Elem, staticSolve ? 1.0 : h, fiberStiffMult, DBCb, X, nodeAttr, elemAttr, elasticityAttr, maxs, avgs, minc, avgc);
#ifdef VERBOSE
        printf("maxs = %le, avgs = %le\n", maxs, avgs);
#endif
        out = fopen((outputFolder + "/stretch.txt").c_str(), "a+");
        fprintf(out, "%le %le %le %le\n", maxs, avgs, minc, avgc);
        fclose(out);
    }
    if (rod.size()) {
        Compute_Max_And_Avg_Stretch_Rod(X, rod, rodInfo, maxs, avgs);
#ifdef VERBOSE
        printf("rod: maxs = %le, avgs = %le\n", maxs, avgs);
#endif
        out = fopen((outputFolder + "/stretch_rod.txt").c_str(), "a+");
        fprintf(out, "%le %le\n", maxs, avgs);
        fclose(out);
    }

    if (!staticSolve) {
        // update velocity
        X.Join(nodeAttr).Par_Each([&](int id, auto data) {
            auto &[x, x0, v, g, m] = data;
            v = (x - std::get<0>(Xn.Get_Unchecked(id))) / h;
        });
    }

    return PNIter;
}

}
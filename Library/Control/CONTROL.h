#pragma once

#include <pybind11/pybind11.h>
#include <Utils/PROFILER.h>
#include <FEM/BOUNDARY_CONDITION.h>
#include <FEM/Shell/INC_POTENTIAL.h>
#include <Control/CONTROL_UTILS.h>
#include <Control/IMPLICIT_EULER.h>

namespace JGSL {

template <class T, int dim, bool KL=false, bool elasticIPC=false, bool flow=false>
void ComputeAdjointVector(
	int n_vert, int n_frame, T h,
	T MDBC_tmin, T MDBC_tmax, T MDBC_period,
	VECTOR_STORAGE<T, dim + 1>& DBC,
	DBC_MOTION<T, dim>& DBCMotion,
	MESH_ELEM<dim - 1>& Elem,
	const std::vector<VECTOR<int, 2>>& seg,
	const std::map<std::pair<int, int>, int>& edge2tri,
	const std::vector<VECTOR<int, 4>>& edgeStencil,
	const std::vector<VECTOR<T, 3>>& edgeInfo,
	const T thickness, T bendingStiffMult,
	const VECTOR<T, 4>& fiberStiffMult,
	const VECTOR<T, 3>& fiberLimit,
	VECTOR<T, 2>& s, VECTOR<T, 2>& sHat, VECTOR<T, 2>& kappa_s,
	const std::vector<T>& bodyForce,
	bool withCollision,
	T dHat2, VECTOR<T, 3>& kappaVec,
	T mu, T epsv2, int fricIterAmt,
	const std::vector<int>& compNodeRange,
	const std::vector<T>& muComp,
	MESH_NODE_ATTR<T, dim>& nodeAttr,
	CSR_MATRIX<T>& M,
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
	MESH_NODE<T, dim>& X0,
	MESH_NODE<T, dim>& X1,
	MESH_NODE<T, dim>& control_force,
	MESH_NODE<T, dim>& trajctory,
	MESH_NODE<T, dim>& adjoint_vector)
{
	TIMER_FLAG("Compute Adjoint Vector");

	T kappa[] = {kappaVec[0], kappaVec[1], kappaVec[2]};

	MESH_NODE<T, dim> X, Xn, Xtilde;

	Fill<T, dim>(X, n_vert);
	Fill<T, dim>(Xn, n_vert);
	Fill<T, dim>(Xtilde, n_vert);

	CSR_MATRIX<T> sysMtr;
	Eigen::Matrix<T, Eigen::Dynamic, 1> L(dim * n_vert), L1(dim * n_vert), L2(dim * n_vert);
	std::vector<T> sol(n_vert * dim);

	for (int i = n_frame - 1; i >= 0; --i) {
		std::cout << "============== adjoint vector " << i << " =============\n";
		
		{
			TIMER_FLAG("Prepare Hessian");
		// preapre hessian
		// prepare X, Xn, and Xtilde, set [v] in nodeAttr =======================================================
		GetFrame<T, dim>(i, trajctory, X);

		std::vector<T> f_ext(bodyForce), a;
		for (int j = 0; j < n_vert; ++j) {
			const VECTOR<T, dim>& x = std::get<0>(control_force.Get_Unchecked_Const(j));
			f_ext[dim * j] += x[0];
			f_ext[dim * j + 1] += x[1];
			if constexpr (dim == 3) {
				f_ext[dim * j + 2] += x[2];
			}
		}
		if (!Solve_Direct(M, f_ext, a)) {
			std::cout << "mass matrix factorization failed!" << std::endl;
			exit(-1);
		}

		if (i == 0) {
			X1.deep_copy_to(Xn);
			X0.deep_copy_to(Xtilde);
		}
		else if (i == 1) {
			GetFrame<T, dim>(0, trajctory, Xn);
			X1.deep_copy_to(Xtilde);
		}
		else {
			GetFrame<T, dim>(i - 1, trajctory, Xn);
			GetFrame<T, dim>(i - 2, trajctory, Xtilde);
		}
		Xtilde.Join(Xn).Par_Each([&](int idx, auto data) {
			auto &[x, xn] = data;
			for (int d = 0; d < dim; ++d) {
				x[d] = 2 * xn[d] - x[d] + h * h * a[idx * dim + d];
			}
		});
		nodeAttr.Join(X, Xn).Par_Each([&](int idx, auto data) {
			auto &[x0, v, g, m, x, xn] = data;
			v = (x - xn) / h;
		});

		// prepare contact promitives and areas =================================================================
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
		std::cout << "surface primitives found" << std::endl;

		// prepare DBC ===============================================================================================
		std::vector<bool> DBCb(X.size, false);
		std::vector<bool> DBCb_fixed(X.size, false);
		std::vector<T> DBCDisp(X.size * dim, T(0));
		DBC.Each([&](int id, auto data) {
			auto &[dbcI] = data;
			int vI = dbcI(0);
			const VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(vI));
			const VECTOR<T, dim> &xn = std::get<0>(Xn.Get_Unchecked(vI));

			bool fixed = true;
			for (int d = 0; d < dim; ++d) {
				dbcI[d + 1] = x[d];
				DBCDisp[dim * vI + d] = x[d] - xn[d];
				if (!DBCDisp[dim * vI + d]) fixed = false;
			}
			DBCb_fixed[vI] = fixed;
			DBCb[vI] = true;
		});
		T DBCStiff = 0; // we don't compute control force on boundary
		std::cout << "DBC handled" << std::endl;

		// prepare collision sets =====================================================================================
		std::vector<VECTOR<int, dim + 1>> constraintSet;
		std::vector<VECTOR<int, 2>> constraintSetPTEE;
		std::vector<VECTOR<T, 2>> stencilInfo;
		// friction
		std::vector<VECTOR<int, dim + 1>> fricConstraintSet;
		std::vector<Eigen::Matrix<T, dim - 1, 1>> closestPoint;
		std::vector<Eigen::Matrix<T, dim, dim - 1>> tanBasis;
		std::vector<T> normalForce;
		if (withCollision) {
			Compute_Constraint_Set<T, dim, false, elasticIPC>(X, nodeAttr, boundaryNode, boundaryEdge, boundaryTri, 
				particle, rod, NNExclusion, BNArea, BEArea, BTArea, codimBNStartInd, DBCb, dHat2, thickness, false, constraintSet, constraintSetPTEE, stencilInfo);
			if (mu > 0 || (muComp.size() && muComp.size() == compNodeRange.size() * compNodeRange.size())) {
				Compute_Friction_Basis<T, dim, elasticIPC>(X, constraintSet, stencilInfo, fricConstraintSet, closestPoint, tanBasis, normalForce, dHat2, kappa, thickness);
				if (muComp.size() && muComp.size() == compNodeRange.size() * compNodeRange.size()) {
					Compute_Friction_Coef<T, dim>(fricConstraintSet, compNodeRange, muComp, normalForce, mu); 
				}
			}
		}
		std::cout << "collision set found" << std::endl;

		// finally, compute hessian ================================================================================	
		Compute_IncPotential_Hessian<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
			s, sHat, kappa_s, DBC, DBCb, DBCb_fixed, DBCStiff, X, Xn, Xtilde, nodeAttr, M, elemAttr, 
			withCollision, constraintSet, stencilInfo, fricConstraintSet, closestPoint, tanBasis, normalForce,
			dHat2, kappa, mu, epsv2, false, bodyForce, elasticityAttr, 
			tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, 
			stitchInfo, stitchRatio, k_stitch, true, sysMtr);

		}
		std::cout << "hessian computed" << std::endl;

		// prepare rhs
		{
			TIMER_FLAG("Compute RHS");
		
		if (i == n_frame - 1) {
			GetFrame<T, dim>(i, trajctory, X);
			X.Join(X1).Par_Each([&](int idx, auto data) {
				auto &[x, x1] = data;
				for (int d = 0; d < dim; ++d) {
					L[dim * idx + d] = x[d] - x1[d];
				}
			});
			L = h * h * M.Get_Matrix() * L;
		}
		else if (i == n_frame - 2) {
			GetFrame<T, dim>(i, trajctory, X);
			X.Join(X0).Par_Each([&](int idx, auto data) {
				auto &[x, x0] = data;
				for (int d = 0; d < dim; ++d) {
					L[dim * idx + d] = x[d] - x0[d];
				}
			});
			L = M.Get_Matrix() * (h * h * L + 2 * L1);
		}
		else {
			L = M.Get_Matrix() * (2 * L1 - L2);
		}
		// project rhs
		DBC.Par_Each([&](int idx, auto data) {
			auto &[dbcI] = data;
			for (int d = 0; d < dim; ++d) {
				L[dbcI(0) * dim + d] = T(0);
			}
		});
		std::cout << "RHS prepared and projected" << std::endl;
		}

		// linear solve
		{
		TIMER_FLAG("Linear Solve");

		std::vector<T> rhs(L.data(), L.data() + L.size());
#ifdef AMGCL_LINEAR_SOLVER
		std::memset(sol.data(), 0, sizeof(T) * sol.size());
		Solve(sysMtr, rhs, sol, 1.0e-5, 1000, Default_FEM_Params<dim>(), true);
#else
		bool solverSucceed = Solve_Direct(sysMtr, rhs, sol);
		if (!solverSucceed) {
			printf("Hessian not SPD in frame %d\n", i);
			exit(-1);
		}
#endif
		std::cout << "linear solver finished" << std::endl;
		}

		L2 = L1;
 		for (int j = 0; j < n_vert; ++j) {
			VECTOR<T, dim>& vec = std::get<0>(adjoint_vector.Get_Unchecked(i * n_vert + j));
			for (int d = 0; d < dim; ++d) {
				L1[j * dim + d] = sol[j * dim + d];
				vec[d] = sol[j * dim + d];
			}
		}
	}
}

template <class T, int dim>
T ComputeForceConstrain(
	int n_vert, int n_frame,
	CSR_MATRIX<T>& M,
	MESH_NODE<T, dim>& X0,
	MESH_NODE<T, dim>& X1,
	MESH_NODE<T, dim>& trajctory)
{
	T constrain = 0.0;

	X0.Join(X1).Each([&](int idx, auto data) {
		auto &[x0, x1] = data;
		const VECTOR<T, dim>& xn = std::get<0>(trajctory.Get_Unchecked_Const((n_frame - 2) * n_vert + idx));
		const VECTOR<T, dim>& xnn = std::get<0>(trajctory.Get_Unchecked_Const((n_frame - 1) * n_vert + idx));

		constrain += 0.5 * M.Get_Item(idx * dim, idx * dim) * ((xn - x0).length2() + (xnn - x1).length2());
	});

	return constrain;
}

void Export_Control(py::module& m)
{
	m.def("Step", &Step<double, 3, false, false>);
    m.def("Step_EIPC", &Step<double, 3, false, true>);

	m.def("ComputeAdjointVector", &ComputeAdjointVector<double, 3>);
	m.def("ComputeForceConstrain", &ComputeForceConstrain<double, 3>);
}

}
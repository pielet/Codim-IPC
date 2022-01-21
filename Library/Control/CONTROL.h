#pragma once

#include <algorithm>
#include <pybind11/pybind11.h>
#include <Utils/PROFILER.h>
#include <FEM/Shell/INC_POTENTIAL.h>
#include <Control/BOUNDARY_CONDITION.h>
#include <Control/CONTROL_UTILS.h>
#include <Control/WIND.h>
#include <Control/IMPLICIT_EULER.h>

namespace JGSL {

template <class T, int dim, bool KL=false, bool elasticIPC=false, bool flow=false>
void ComputeAdjointVector(
	int n_vert, int n_frame, T h,
	VECTOR_STORAGE<T, dim + 1>& DBC,
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
	T k_wind, const VECTOR<T, dim>& wind_dir,
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

	MESH_NODE<T, dim> X, Xn, Xtilde, force;

	Fill<T, dim>(X, n_vert);
	Fill<T, dim>(Xn, n_vert);
	Fill<T, dim>(Xtilde, n_vert);
	Fill<T, dim>(force, n_vert);

	CSR_MATRIX<T> sysMtr, H_wind;
	Eigen::Matrix<T, Eigen::Dynamic, 1> L(dim * n_vert), L1(dim * n_vert), L2(dim * n_vert);
	std::vector<T> sol(n_vert * dim);

	for (int i = n_frame - 1; i >= 0; --i) {
#ifdef VERBOSE
		std::cout << "============== adjoint vector " << i << " =============\n";
#endif
		{
		TIMER_FLAG("Compute Hessian and RHS");
		// preapre hessian
		// prepare X, Xn, and Xtilde, set [v] in nodeAttr =======================================================
		GetFrame<T, dim>(i, trajctory, X);

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

		std::vector<T> f_ext(bodyForce), a;
		GetFrame<T, dim>(i, control_force, force);

		force.Par_Each([&](int idx, auto data) {
			auto &[f] = data;
			for (int d = 0; d < dim; ++d) {
				f_ext[idx * dim + d] += f[d];
			}
		});

		if (!Solve_Direct(M, f_ext, a)) {
			std::cout << "mass matrix factorization failed!" << std::endl;
			exit(-1);
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
#ifdef VERBOSE
		std::cout << "DBC handled" << std::endl;
#endif

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
#ifdef VERBOSE
		std::cout << "surface primitives found" << std::endl;
#endif

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
#ifdef VERBOSE
		std::cout << "collision set found" << std::endl;
#endif

		// finally, compute hessian ================================================================================	
		Compute_IncPotential_Hessian<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
			s, sHat, kappa_s, DBC, DBCb, DBCb_fixed, DBCStiff, X, Xn, Xtilde, nodeAttr, M, elemAttr, 
			withCollision, constraintSet, stencilInfo, fricConstraintSet, closestPoint, tanBasis, normalForce,
			dHat2, kappa, mu, epsv2, false, bodyForce, elasticityAttr, 
			tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, 
			stitchInfo, stitchRatio, k_stitch, true, sysMtr);

#ifdef VERBOSE
		std::cout << "hessian computed" << std::endl;
#endif

		// prepare rhs
		// if (k_wind > 0) {
		// 	std::vector<Eigen::Triplet<T>> wind_triplets;
		// 	Compute_Wind_Hessian<T, dim>(k_wind, wind_dir, DBCb, Elem, elasticityAttr, X, wind_triplets);
		// 	std::vector<Eigen::Triplet<T>> triplets;
		// 	Add_Block(sysMtr.Get_Matrix(), triplets, 0, 0);
		// 	Add_Block(wind_triplets, triplets, 0, 0, -h * h);
		// 	sysMtr.Construct_From_Triplet(n_vert * dim, n_vert * dim, triplets);
		// 	sysMtr.Get_Matrix() = sysMtr.Get_Matrix().transpose();
		// }
		
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
			// if (k_wind > 0) {
			// 	L += h * h * H_wind.Get_Matrix().transpose() * L1;
			// }
		}
		else {
			L = M.Get_Matrix() * (2 * L1 - L2);
			// if (k_wind > 0) {
			// 	L += h * h * H_wind.Get_Matrix().transpose() * L1;
			// }
		}
		// project rhs
		DBC.Par_Each([&](int idx, auto data) {
			auto &[dbcI] = data;
			for (int d = 0; d < dim; ++d) {
				L[dbcI(0) * dim + d] = T(0);
			}
		});
#ifdef VERBOSE
		std::cout << "RHS prepared and projected" << std::endl;
#endif
		}

		// linear solve
		{
		TIMER_FLAG("Linear Solve");

		std::vector<T> rhs(L.data(), L.data() + L.size());

		bool solverSucceed = Solve_Direct(sysMtr, rhs, sol);
		if (!solverSucceed) {
			printf("Hessian not SPD in frame %d\n", i);
			exit(-1);
		}
#ifdef VERBOSE
		std::cout << "linear solver finished" << std::endl;
#endif
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
T ComputeConstrain(
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

template <class T, int dim, bool KL=false, bool elasticIPC=false, bool flow=false>
bool ComputeLoopyLoss(
	int n_vert, int n_frame, T h, int p, // p-norm
	VECTOR_STORAGE<T, dim + 1>& DBC,
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
	T k_wind, const VECTOR<T, dim>& wind_dir,
	bool withCollision,
	T dHat2, VECTOR<T, 3>& kappaVec,
	T mu, T epsv2, int fricIterAmt,
	const std::vector<int>& compNodeRange,
	const std::vector<T>& muComp,
	MESH_NODE_ATTR<T, dim>& nodeAttrBase,
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
	MESH_NODE<T, dim>& trajctory,
	std::vector<int>& valid_per_frame,
	std::vector<T>& loss_per_frame,
	MESH_NODE<T, dim>& residual)
{
	loss_per_frame.resize(n_frame);
	valid_per_frame.resize(n_frame);

	T kappa[] = {kappaVec[0], kappaVec[1], kappaVec[2]};

	MESH_NODE<T, dim> X, Xn, Xtilde, force;
	MESH_NODE_ATTR<T, dim> nodeAttr;

	Fill<T, dim>(X, n_vert);
	Fill<T, dim>(Xn, n_vert);
	Fill<T, dim>(Xtilde, n_vert);
	Fill<T, dim>(force, n_vert);
	Append_NodeAttr(nodeAttrBase, nodeAttr);

	// prepare DBC ===============================================================================================
	std::vector<bool> DBCb(n_vert, false);
	DBC.Each([&](int id, auto data) {
		auto &[dbcI] = data;
		DBCb[dbcI(0)] = true;
	});

	for (int i = 0; i < n_frame; ++i) {
		// prepare X, Xn, and Xtilde, set [v] in nodeAttr =======================================================
		GetFrame<T, dim>(i, trajctory, X);
		GetFrame<T, dim>((i - 1 + n_frame) % n_frame, trajctory, Xn);
		GetFrame<T, dim>((i - 2 + n_frame) % n_frame, trajctory, Xtilde);

		std::vector<T> f_ext(bodyForce), a;
		if (k_wind > 0) {
			Compute_Wind_Force<T, dim>(k_wind, wind_dir, DBCb, Elem, elasticityAttr, Xn, force);
			force.Par_Each([&](int idx, auto data) {
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
		Xtilde.Join(Xn).Par_Each([&](int idx, auto data) {
			auto &[y, xn] = data;
			for (int d = 0; d < dim; ++d) {
				y[d] = 2 * xn[d] - y[d] + h * h * a[idx * dim + d];
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

		// check validation =========================================================================================
		bool valid = true;
		if (kappa_s[0] > 0 || fiberStiffMult[0] > 0 || tet.size > 0) {
			T E;
			valid = Compute_IncPotential<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, 
				fiberLimit, s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, elasticityAttr, 
				false, constraintSet, stencilInfo, dHat2, kappa, false, bodyForce, 
				tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, stitchInfo, stitchRatio, k_stitch, E);
		}
		T alpha = 1.0;
		if (withCollision) {
			std::vector<T> diffX(n_vert * dim);
			X.Join(Xn).Par_Each([&](int idx, auto data) {
				auto &[x, xn] = data;
				for (int d = 0; d < dim; ++d) {
					diffX[idx * dim + d] = x[d] - xn[d];
				}
			});
			Compute_Intersection_Free_StepSize<T, dim, false, elasticIPC>(Xn, boundaryNode, boundaryEdge, boundaryTri, 
				particle, rod, NNExclusion, codimBNStartInd, DBCb, diffX, thickness, alpha); // CCD
		}
		valid_per_frame[i] = valid && (alpha > 0.99);

		// compute gradient ==========================================================================================
		loss_per_frame[i] = 0.0;
		// if (valid_per_frame[i]) {
			Compute_IncPotential_Gradient<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, 
				thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
				s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, 
				withCollision, constraintSet, stencilInfo, dHat2, kappa, false, bodyForce, elasticityAttr, 
				tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, stitchInfo, stitchRatio, k_stitch);
			if (withCollision && mu > 0) {
				Compute_Friction_Gradient(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, nodeAttr);
			}

			// project gradient for DBC
			DBC.Par_Each([&](int id, auto data) {
				auto &[dbcI] = data;
				std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(dbcI(0))).setZero();
			});

			T loss = 0.0;
			nodeAttr.Each([&](int idx, auto data) {
				auto &[x0, v, g, m] = data;
				if (p == 2) {
					loss += 0.5 * g.length2();
				}
				else {
					for (int d = 0; d < dim; ++d) {
						loss += 1.0 / p * std::pow(std::abs(g[d]), p);
					}
				}
				std::get<0>(residual.Get_Unchecked(i * n_vert + idx)) = g;
			});

			loss_per_frame[i] = loss;
		}
	// }

	int valid = 1;
	for (int v : valid_per_frame) {
		valid &= v;
	}

	return valid;
}

template <class T, int dim, bool SC, bool GN, bool KL=false, bool elasticIPC=false, bool flow=false>
void ComputeTrajectoryGradient(
	int n_vert, int n_frame, T h, int p, T epsilon, bool use_cg, int cg_iter, T cg_tol, T regu,
	VECTOR_STORAGE<T, dim + 1>& DBC,
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
	T k_wind, const VECTOR<T, dim>& wind_dir,
	bool withCollision,
	T dHat2, VECTOR<T, 3>& kappaVec,
	T mu, T epsv2, int fricIterAmt,
	const std::vector<int>& compNodeRange,
	const std::vector<T>& muComp,
	MESH_NODE_ATTR<T, dim>& nodeAttrBase,
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
	MESH_NODE<T, dim>& trajctory,
	MESH_NODE<T, dim>& gradient,
	MESH_NODE<T, dim>& descent_dir,
	MESH_NODE<T, dim>& residual,
	MESH_NODE<T, dim>& hess_res,
	CSR_MATRIX<T>& A)
{
	T kappa[] = {kappaVec[0], kappaVec[1], kappaVec[2]};

	MESH_NODE<T, dim> X, Xn, Xtilde, force;
	MESH_NODE_ATTR<T, dim> nodeAttr;

	Fill<T, dim>(X, n_vert);
	Fill<T, dim>(Xn, n_vert);
	Fill<T, dim>(Xtilde, n_vert);
	Fill<T, dim>(force, n_vert);
	Append_NodeAttr(nodeAttrBase, nodeAttr);

	SetZero(gradient);
	std::vector<Eigen::Triplet<T>> triplets;
	std::vector<Eigen::Triplet<T>> dx_triplets(n_frame * n_vert * dim);
	std::vector<bool> global_DBCb(n_frame * n_vert, false);

	std::vector<CSR_MATRIX<T>> sysMtr_list(n_frame);

	// prepare DBC ===============================================================================================
	std::vector<bool> DBCb(X.size, false); // do not compute inner force between DBC
	std::vector<bool> DBCb_fixed(X.size, false); // DO NOT project hessian for DBC
	DBC.Each([&](int id, auto data) {
		auto &[dbcI] = data;
		int vI = dbcI(0);
		DBCb[vI] = true;
	});
	T DBCStiff = 0; // DBCStiff == 1 && DBCb_fixed all false => DO NOT project hessian
	for (int i = 0; i < n_frame; ++i) {
		for (int j = 0; j < n_vert; ++j) {
			global_DBCb[i * n_vert + j] = DBCb[j];
		}
	}

	int H_nonzero = 0;
	int HTH_nonzero = 0;

	for (int i = 0; i < n_frame; ++i) {
		// prepare X, Xn, and Xtilde, set [v] in nodeAttr =======================================================
		GetFrame<T, dim>(i, trajctory, X);
		GetFrame<T, dim>((i - 1 + n_frame) % n_frame, trajctory, Xn);
		GetFrame<T, dim>((i - 2 + n_frame) % n_frame, trajctory, Xtilde);

		std::vector<T> a;
		if (!Solve_Direct(M, bodyForce, a)) {
			std::cout << "mass matrix factorization failed!" << std::endl;
			exit(-1);
		}
		Xtilde.Join(Xn).Par_Each([&](int idx, auto data) {
			auto &[y, xn] = data;
			for (int d = 0; d < dim; ++d) {
				y[d] = 2 * xn[d] - y[d] + h * h * a[idx * dim + d];
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

		// project Xtilde
		Xtilde.Join(X).Par_Each([&](int idx, auto data) {
			auto &[y, x] = data;
			if (DBCb[idx]) {
				y = x;
			}
		});

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
		
		// compute deformation gradient
		if (kappa_s[0] > 0 || fiberStiffMult[0] > 0 || tet.size > 0) {
			T E;
			bool valid = Compute_IncPotential<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, 
				fiberLimit, s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, elasticityAttr, 
				false, constraintSet, stencilInfo, dHat2, kappa, false, bodyForce, 
				tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, stitchInfo, stitchRatio, k_stitch, E);
		}

		// compute gradient
		Compute_IncPotential_Gradient<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, 
			thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
			s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, 
			withCollision, constraintSet, stencilInfo, dHat2, kappa, false, bodyForce, elasticityAttr, 
			tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, stitchInfo, stitchRatio, k_stitch);
		if (withCollision && mu > 0) {
			Compute_Friction_Gradient(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, nodeAttr);
		}
		
		if (k_wind > 0) {
			Compute_Wind_Force<T, dim>(k_wind, wind_dir, DBCb, Elem, elasticityAttr, Xn, force);
		}

		// accumulate to residual
		nodeAttr.Join(force).Par_Each([&](int idx, auto data) {
			auto &[x0, v, g, m, f_wind] = data;

			VECTOR<T, dim>& res = std::get<0>(residual.Get_Unchecked(i * n_vert + idx));
			VECTOR<T, dim> total_g = g - h * h * f_wind;

			for (int d = 0; d < dim; ++d) {
				if (p == 2) {
					res[d] = total_g[d];
				}
				else {
					res[d] = total_g[d] * std::pow(std::abs(total_g[d]), p - 2);
				}
			}
		});

		// compute hessian
		CSR_MATRIX<T> sysMtr;
		Compute_IncPotential_Hessian<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
			s, sHat, kappa_s, DBC, DBCb, DBCb_fixed, DBCStiff, X, Xn, Xtilde, nodeAttr, M, elemAttr, 
			withCollision, constraintSet, stencilInfo, fricConstraintSet, closestPoint, tanBasis, normalForce,
			dHat2, kappa, mu, epsv2, false, bodyForce, elasticityAttr, 
			tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, 
			stitchInfo, stitchRatio, k_stitch, true, sysMtr, false);

		H_nonzero = sysMtr.Get_Matrix().nonZeros();
		HTH_nonzero = Eigen::SparseMatrix<T, Eigen::RowMajor>(sysMtr.Get_Matrix() * sysMtr.Get_Matrix().transpose()).nonZeros();

		// accumulate to triplets
		Add_Block(sysMtr.Get_Matrix(), triplets, i * n_vert * dim, i * n_vert * dim);
		Add_Block(M.Get_Matrix(), triplets, (i - 1 + n_frame) % n_frame * n_vert * dim, i * n_vert * dim, -2.0);
		Add_Block(M.Get_Matrix(), triplets, (i - 2 + n_frame) % n_frame * n_vert * dim, i * n_vert * dim);

		if (k_wind > 0) {
			std::vector<Eigen::Triplet<T>> wind_triplets;
			Compute_Wind_Hessian<T, dim>(k_wind, wind_dir, DBCb, Elem, elasticityAttr, Xn, wind_triplets);
			Add_Block(wind_triplets, triplets, (i - 1 + n_frame) % n_frame * n_vert * dim, i * n_vert * dim, -h * h, true);
		}

		if (p > 2) {
			nodeAttr.Par_Each([&](int idx, auto data) {
				auto &[x0, v, g, m] = data;
				int base = i * n_vert * dim + idx * dim;
				for (int d = 0; d < dim; ++d) {
					dx_triplets[base + d] = Eigen::Triplet<T>(base + d, base + d, (p - 1) * std::pow(std::abs(g[d]), p - 2));
				} 
			});
		}

// #define ADJOINT_SOLVER
#ifdef ADJOINT_SOLVER
		sysMtr.Project_DBC(DBCb, 3);
		sysMtr_list[i] = sysMtr;
#endif
	}

	printf("finish per-frame residual and triplets (%d), H (%d) HTH (%d)\n", triplets.size(), H_nonzero, HTH_nonzero);

	A.Construct_From_Triplet(n_frame * n_vert * dim, n_frame * n_vert * dim, triplets);
	printf("AT nonzeros: %d\n", A.Get_Matrix().nonZeros());

	Eigen::VectorXd eigen_residual(n_frame * n_vert * dim);
	node_to_eigen(residual, eigen_residual);
	Eigen::VectorXd eigen_gradient = A.Get_Matrix() * eigen_residual;
	eigen_to_node(eigen_gradient, gradient);

	// for visualization
	for (int i = 0; i < n_frame; ++i) {
		DBC.Par_Each([&](int id, auto data) {
			auto &[dbcI] = data;
			std::get<0>(residual.Get_Unchecked(i * n_vert + dbcI(0))).setZero();
		});
	}

	if constexpr (SC) {
		if (epsilon > 0) {
			X0.Join(X1).Par_Each([&](int idx, auto data) {
				auto &[x0, x1] = data;
				T a = M.Get_Item(idx * dim, idx * dim) * epsilon;
				std::get<0>(gradient.Get_Unchecked((n_frame - 2) * n_vert + idx)) += a * (std::get<0>(trajctory.Get_Unchecked((n_frame - 2) * n_vert + idx)) - x0);
				std::get<0>(gradient.Get_Unchecked((n_frame - 1) * n_vert + idx)) += a * (std::get<0>(trajctory.Get_Unchecked((n_frame - 1) * n_vert + idx)) - x1);
			});
		}
	}
	else {
		// project gradient for loopy constrain
		X.Par_Each([&](int idx, auto data) {
			std::get<0>(gradient.Get_Unchecked((n_frame - 2) * n_vert + idx)).setZero();
			std::get<0>(gradient.Get_Unchecked((n_frame - 1) * n_vert + idx)).setZero();
		});
	}

	// project gradient for DBC
	for (int i = 0; i < n_frame; ++i) {
		DBC.Par_Each([&](int id, auto data) {
			auto &[dbcI] = data;
			std::get<0>(gradient.Get_Unchecked(i * n_vert + dbcI(0))).setZero();
		});
	}

	printf("finish gradient computation\n");

	if constexpr (GN) {
#ifndef ADJOINT_SOLVER

		CSR_MATRIX<T> H;
		if (p == 2) {
			H.Get_Matrix() = (A.Get_Matrix() * A.Get_Matrix().transpose()).pruned(1e-6);
		}
		else {
			CSR_MATRIX<T> d2X;
			d2X.Construct_From_Triplet(n_frame * n_vert * dim, n_frame * n_vert * dim, dx_triplets);
			H.Get_Matrix() = (A.Get_Matrix() * d2X.Get_Matrix() * A.Get_Matrix().transpose()).pruned(1e-6);
		}
		printf("ATA nonzeros: %d\n", H.Get_Matrix().nonZeros());

		if constexpr (SC) {
			if (epsilon > 0) {
			// add to hessian
				for (int i = 0; i < n_vert * dim; ++i) {
					H.Get_Item((n_frame - 2) * n_vert * dim + i, (n_frame - 2) * n_vert * dim + i) += M.Get_Item(i, i) * epsilon;
					H.Get_Item((n_frame - 1) * n_vert * dim + i, (n_frame - 1) * n_vert * dim + i) += M.Get_Item(i, i) * epsilon;
				}
			}
		}
		else {
			// project A for loopy constrain
			for (int i = 0; i < n_vert; ++i) {
				global_DBCb[(n_frame - 2) * n_vert + i] = true;
				global_DBCb[(n_frame - 1) * n_vert + i] = true;
			}
		}

		H.Project_DBC(global_DBCb, 3);

		// b = - A^T * R = -g
		std::vector<T> sol(n_frame * n_vert * dim);
		std::vector<T> rhs(n_frame * n_vert * dim);

		gradient.Par_Each([&](int idx, auto data) {
			auto &[g] = data;
			for (int d = 0; d < dim; ++d) {
				rhs[idx * dim + d] = -g[d];
			}
		});

		if (use_cg) {
			// AMGCL
			H.Add_Identity(regu);
			std::memset(sol.data(), 0, sizeof(T) * sol.size());
			Solve<T, dim>(H, rhs, sol, cg_tol, cg_iter, Gauss_Newton_Params<dim>(), true);
		}
		else {
			// direct factorization

			T regular = 1e-6;
			while (!Solve_Direct(H, rhs, sol)) {
				std::cout << "add identity: " << regular << std::endl;
				H.Add_Identity(regular);
				regular *= 10;
			}
		}

		descent_dir.Par_Each([&](int idx, auto data) {
			auto &[desc] = data;
			for (int d = 0; d < dim; ++d) {
				desc[d] = sol[idx * dim + d];
			}
		});

		printf("finish Gauss-Newton linear solve\n");
#else
		Eigen::VectorXd yn(dim * n_vert), ynn(dim * n_vert);
		yn.setZero(); ynn.setZero();
		std::vector<T> Y(n_frame * n_vert * dim);
		std::vector<T> sol(n_vert * dim);
		std::vector<T> rhs(n_vert * dim);
		for (int i = n_frame - 1; i >= 0; --i) {
			GetFrame<T, dim>(i, gradient, X);
			node_to_vector(X, rhs, -1.0);

			Eigen::VectorXd Myn = M.Get_Matrix() * yn;
			for (int j = 0; j < rhs.size(); ++j) {
				rhs[j] += 2.0 * Myn[j];
			}
			Eigen::VectorXd Mynn = M.Get_Matrix() * ynn;
			for (int j = 0;j < rhs.size(); ++j) {
				rhs[j] -= Mynn[j];
			}

			Solve_Direct(sysMtr_list[i], rhs, sol);

			ynn = yn;
			for (int j = 0; j < sol.size(); ++j) {
				yn[j] = sol[j];
				Y[i * n_vert * dim + j] = sol[j];
			}
		}
		yn.setZero(); ynn.setZero();
		for (int i = 0; i < n_frame; ++i) {
			for (int j = 0; j < rhs.size(); ++j) {
				rhs[j] = Y[i * n_vert * dim + j];
			}
			Eigen::VectorXd Myn = M.Get_Matrix() * yn;
			for (int j = 0; j < rhs.size(); ++j) {
				rhs[j] += 2.0 * Myn[j];
			}
			Eigen::VectorXd Mynn = M.Get_Matrix() * ynn;
			for (int j = 0;j < rhs.size(); ++j) {
				rhs[j] -= Mynn[j];
			}

			Solve_Direct(sysMtr_list[i], rhs, sol);

			ynn = yn;
			for (int j = 0; j < sol.size(); ++j) {
				yn[j] = sol[j];
			}

			for (int j = 0; j < n_vert; ++j) {
				VECTOR<T, dim>& desc = std::get<0>(descent_dir.Get_Unchecked(i * n_vert + j));
				for (int d = 0; d < dim; ++d) {
					desc[d] = sol[j * dim + d];
				}
			}
		}
#endif
	}
	else {
		gradient.Join(descent_dir).Par_Each([&](int idx, auto data) {
			auto &[grad, desc] = data;
			desc = -grad;
		});
	}
}

template <class T, int dim, bool SC, bool GN, bool KL=false, bool elasticIPC=false, bool flow=false>
void ComputeTrajectoryGradientMinMax(
	int n_vert, int n_frame, T h, T epsilon, bool use_cg, int cg_iter, T cg_tol, T regu,
	const std::vector<T>& loss_per_frame,
	VECTOR_STORAGE<T, dim + 1>& DBC,
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
	T k_wind, const VECTOR<T, dim>& wind_dir,
	bool withCollision,
	T dHat2, VECTOR<T, 3>& kappaVec,
	T mu, T epsv2, int fricIterAmt,
	const std::vector<int>& compNodeRange,
	const std::vector<T>& muComp,
	MESH_NODE_ATTR<T, dim>& nodeAttrBase,
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
	MESH_NODE<T, dim>& trajctory,
	MESH_NODE<T, dim>& gradient,
	MESH_NODE<T, dim>& descent_dir,
	MESH_NODE<T, dim>& residual,
	MESH_NODE<T, dim>& hess_res,
	CSR_MATRIX<T>& local_H)
{
	int i = std::max_element(loss_per_frame.begin(), loss_per_frame.end()) - loss_per_frame.begin();
	std::cout << "max_i: " << i << std::endl;

	T kappa[] = {kappaVec[0], kappaVec[1], kappaVec[2]};
	
	MESH_NODE<T, dim> X, Xn, Xtilde;
	MESH_NODE_ATTR<T, dim> nodeAttr;

	Fill<T, dim>(X, n_vert);
	Fill<T, dim>(Xn, n_vert);
	Fill<T, dim>(Xtilde, n_vert);
	Append_NodeAttr(nodeAttrBase, nodeAttr);

	// prepare X, Xn, and Xtilde, set [v] in nodeAttr =======================================================
	GetFrame<T, dim>(i, trajctory, X);
	GetFrame<T, dim>((i - 1 + n_frame) % n_frame, trajctory, Xn);
	GetFrame<T, dim>((i - 2 + n_frame) % n_frame, trajctory, Xtilde);

	std::vector<T> a;
	if (!Solve_Direct(M, bodyForce, a)) {
		std::cout << "mass matrix factorization failed!" << std::endl;
		exit(-1);
	}
	Xtilde.Join(Xn).Par_Each([&](int idx, auto data) {
		auto &[y, xn] = data;
		for (int d = 0; d < dim; ++d) {
			y[d] = 2 * xn[d] - y[d] + h * h * a[idx * dim + d];
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

	// prepare DBC ===============================================================================================
	std::vector<bool> DBCb(X.size, false); // do not compute inner force between DBC
	std::vector<bool> DBCb_fixed(X.size, false); // DO NOT project hessian for DBC
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
		DBCb[vI] = true;
	});
	T DBCStiff = 0; // DBCStiff == 1 && DBCb_fixed all false => DO NOT project hessian

	// project Xtilde
	Xtilde.Join(X).Par_Each([&](int idx, auto data) {
		auto &[y, x] = data;
		if (DBCb[idx]) {
			y = x;
		}
	});

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
		
	// compute gradient
	Compute_IncPotential_Gradient<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, 
		thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
		s, sHat, kappa_s, DBCb, X, Xtilde, nodeAttr, M, elemAttr, 
		withCollision, constraintSet, stencilInfo, dHat2, kappa, false, bodyForce, elasticityAttr, 
		tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, stitchInfo, stitchRatio, k_stitch);
	if (withCollision && mu > 0) {
		Compute_Friction_Gradient(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, nodeAttr);
	}

	// compute hessian
	CSR_MATRIX<T> sysMtr;
	Compute_IncPotential_Hessian<T, dim, KL, elasticIPC, flow>(Elem, h, edge2tri, edgeStencil, edgeInfo, thickness, bendingStiffMult, fiberStiffMult, fiberLimit,
		s, sHat, kappa_s, DBC, DBCb, DBCb_fixed, DBCStiff, X, Xn, Xtilde, nodeAttr, M, elemAttr, 
		withCollision, constraintSet, stencilInfo, fricConstraintSet, closestPoint, tanBasis, normalForce,
		dHat2, kappa, mu, epsv2, false, bodyForce, elasticityAttr, 
		tet, tetAttr, tetElasticityAttr, rod, rodInfo, rodHinge, rodHingeInfo, 
		stitchInfo, stitchRatio, k_stitch, true, sysMtr, false);

	Eigen::VectorXd x_bar(dim * n_vert);
	SetZero(residual);
	nodeAttr.Par_Each([&](int idx, auto data) {
		auto &[x0, v, g, m] = data;
		for (int d = 0; d < dim; ++d) {
			x_bar[idx * dim + d] = g[d];
		}
		std::get<0>(residual.Get_Unchecked(i * n_vert + idx)) = g;
	});

	std::vector<Eigen::Triplet<T>> triplets;
	Add_Block(M.Get_Matrix(), triplets, 0, 0);
	Add_Block(M.Get_Matrix(), triplets, 0, n_vert * dim, -2.0);
	Add_Block(sysMtr.Get_Matrix(), triplets, 0, 2 * n_vert * dim);

	Eigen::SparseMatrix<T, Eigen::RowMajor> dx_bar_dx(n_vert * dim, 3 * n_vert * dim);
	dx_bar_dx.setFromTriplets(triplets.begin(), triplets.end());

	Eigen::VectorXd local_g = dx_bar_dx.transpose() * x_bar; // (3 * n_vert * dim, 1)

	// project DBC
	DBC.Par_Each([&](int idx, auto data) {
		auto &[dbc] = data;
		int vI = dbc[0];
		for (int d = 0; d < dim; ++d) {
			local_g[vI * dim + d] = 0.0;
			local_g[n_vert * dim + vI * dim + d] = 0.0;
			local_g[2 * n_vert * dim + vI * dim + d] = 0.0;
		}
	});

	// project DBC
	SetZero(gradient);
	for (int fi = 0; fi < 3; ++fi) {
		int cur_fi = (i - 2 + fi + n_frame) % n_frame;
		for (int vi = 0; vi < n_vert; ++vi) {
			VECTOR<T, dim>& g = std::get<0>(gradient.Get_Unchecked(cur_fi * n_vert + vi));
			for (int d = 0; d < dim; ++d) {
				g[d] = local_g[fi * n_vert * dim + vi * dim + d];
			}
		}
	}

	if constexpr (!SC) {
		for (int fi = 0; fi < 3; ++fi) {
			int cur_fi = (i - 2 + fi + n_frame) % n_frame;
			if (cur_fi == n_frame - 2 || cur_fi == n_frame - 1) {
				for (int vi = 0; vi < n_vert; ++vi) {
					std::get<0>(gradient.Get_Unchecked(cur_fi * n_vert + vi)).setZero();
				}
			}
		}
	}

	if constexpr (GN) {
		local_H.Get_Matrix() = dx_bar_dx.transpose() * dx_bar_dx;

		// project hessian
		std::vector<bool> global_DBCb(3 * n_vert * dim, false);
		DBC.Par_Each([&](int idx, auto data) {
			auto &[dbc] = data;
			int vI = dbc[0];
			for (int fi = 0; fi < 3; ++ fi) {
				global_DBCb[fi * n_vert + vI] = true;
			}
		});

		if constexpr (!SC) {
			// project A for loopy constrain
			for (int fi = 0; fi < 3; ++fi) {
				int cur_fi = (i - 2 + fi + n_frame) % n_frame;
				if (cur_fi == n_frame - 2 || cur_fi == n_frame - 1) {
					for (int vi = 0; vi < n_vert; ++vi) {
						global_DBCb[fi * n_vert + vi] = true;
					}
				}
			}
		}
		for (int fi = 0; fi < 3; ++fi) {
			int cur_fi = (i - 2 + fi + n_frame) & n_frame;
			for (int vi = 0; vi < n_vert; ++vi) {
				global_DBCb[fi * n_vert + vi] = true;
			}
		}

		local_H.Project_DBC(global_DBCb, dim);

		// linear solve
		std::vector<T> sol(3 * n_vert * dim);
		local_g = -local_g;
		std::vector<T> rhs(local_g.data(), local_g.data() + local_g.size());
		if (use_cg) {
			// AMGCL
			local_H.Add_Identity(regu);
			std::memset(sol.data(), 0, sizeof(T) * sol.size());
			Solve<T, dim>(local_H, rhs, sol, cg_tol, cg_iter, Default_FEM_Params<dim>(), true);
		}
		else {
			// direct factorization
			T regular = 1e-6;
			while (!Solve_Direct(local_H, rhs, sol)) {
				std::cout << "Add identity: " << regular << std::endl;
				local_H.Add_Identity(regular);
				regular *= 10;
			}
		}

		SetZero(descent_dir);
		for (int fi = 0; fi < 3; ++fi) {
			for (int vi = 0; vi < n_vert; ++vi) {
				VECTOR<T, dim>& decs = std::get<0>(descent_dir.Get_Unchecked((i - 2 + fi + n_frame) % n_frame * n_vert + vi));
				for (int d = 0; d < dim; ++d) {
					decs[d] = sol[fi * n_vert * dim + vi * dim + d];
				}
			}
		}
	}
	else {
		gradient.Join(descent_dir).Par_Each([&](int idx, auto data) {
			auto &[grad, desc] = data;
			desc = -grad;
		});
	}
}

void Export_Control(py::module& m)
{
	m.def("Step", &Step<double, 3, false, false>);
	m.def("Step_EIPC", &Step<double, 3, false, true>);

	m.def("ComputeAdjointVector", &ComputeAdjointVector<double, 3>);
	m.def("ComputeConstrain", &ComputeConstrain<double, 3>);

	m.def("ComputeLoopyLoss", &ComputeLoopyLoss<double, 3>);
	m.def("ComputeTrajectoryGradient", &ComputeTrajectoryGradient<double, 3, false, false>);
	m.def("ComputeTrajectoryGradient_SoftCon", &ComputeTrajectoryGradient<double, 3, true, false>);
	m.def("ComputeTrajectoryGradientDescent", &ComputeTrajectoryGradient<double, 3, false, true>);
	m.def("ComputeTrajectoryGradientDescent_SoftCon", &ComputeTrajectoryGradient<double, 3, true, true>);

	m.def("ComputeTrajectoryGradientMinMax", &ComputeTrajectoryGradientMinMax<double, 3, false, false>);
	m.def("ComputeTrajectoryGradientMinMax_SoftCon", &ComputeTrajectoryGradientMinMax<double, 3, true, false>);
	m.def("ComputeTrajectoryGradientDescentMinMax", &ComputeTrajectoryGradientMinMax<double, 3, false, true>);
	m.def("ComputeTrajectoryGradientDescentMinMax_SoftCon", &ComputeTrajectoryGradientMinMax<double, 3, true, true>);
}

}
#pragma once

#include <Math/VECTOR.h>
#include <FEM/DATA_TYPE.h>
#include <Control/CONTROL_UTILS.h>

namespace JGSL {
template <class T, int dim>
void Compute_Wind_Force(
	T k_wind, const VECTOR<T, dim>& wind_dir,
	const std::vector<bool>& DBCb,
	MESH_ELEM<dim - 1>& Elem,
	FIXED_COROTATED<T, dim - 1>& elasticityAttr,
	MESH_NODE<T, dim>& X, // mid-surface node coordinates
	MESH_NODE<T, dim>& force)
{
	SetZero(force);

	using VEC = VECTOR<T, dim>;
	Elem.Join(elasticityAttr).Each([&](int idx, auto data) {
		auto &[elemVInd, F, vol, lambda, mu] = data;
		if (!(DBCb[elemVInd[0]] && DBCb[elemVInd[1]] && DBCb[elemVInd[2]])) {
			const VEC& x0 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
			const VEC& x1 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
			const VEC& x2 = std::get<0>(X.Get_Unchecked(elemVInd[2]));

			VEC N = cross(x1 - x0, x2 - x1).Normalized();
			VEC local_f = k_wind * vol * N.dot(wind_dir.Normalized()) * N;

			for (int d = 0; d < dim; ++d) {
				std::get<0>(force.Get_Unchecked(elemVInd[d])) += local_f / dim;
			}
		}
	});
}

template <class T, int dim>
void Compute_Wind_Hessian(
	T k_wind, const VECTOR<T, dim>& wind_dir,
	const std::vector<bool>& DBCb,
	MESH_ELEM<dim - 1>& Elem,
	FIXED_COROTATED<T, dim - 1>& elasticityAttr,
	MESH_NODE<T, dim>& X,
	std::vector<Eigen::Triplet<T>>& triplets)
{
	using VEC = VECTOR<T, dim>;
	using MAT = MATRIX<T, dim>;

	std::vector<int> tripletStartInd(Elem.size);
	int nonDBCElemCount = 0;
	Elem.Each([&](int id, auto data) {
		auto &[elemVInd] = data;
		if (!(DBCb[elemVInd[0]] && DBCb[elemVInd[1]] && DBCb[elemVInd[2]])) {
				tripletStartInd[id] = triplets.size() + nonDBCElemCount * 81;
				++nonDBCElemCount;
		}
		else {
			tripletStartInd[id] = -1;
		}
	});
	triplets.resize(triplets.size() + nonDBCElemCount * 81);

	Elem.Join(elasticityAttr).Par_Each([&](int idx, auto data) {
		auto &[elemVInd, F, vol, lambda, mu] = data;
		if (!(DBCb[elemVInd[0]] && DBCb[elemVInd[1]] && DBCb[elemVInd[2]])) {
			const VEC& x0 = std::get<0>(X.Get_Unchecked(elemVInd[0]));
			const VEC& x1 = std::get<0>(X.Get_Unchecked(elemVInd[1]));
			const VEC& x2 = std::get<0>(X.Get_Unchecked(elemVInd[2]));

			VEC e1 = x1 - x0;
			VEC e2 = x2 - x1;
			VEC E = cross(e1, e2); 
			VEC N = E.Normalized();

			MAT dfdN = k_wind * vol / dim * (N.dot(wind_dir.Normalized()) * MAT(1.0) + outer_product(N, wind_dir.Normalized()));
			MAT dNdE = (MAT(1.0) - outer_product(N, N)) / E.length();
			MAT dEde1 = -axial(e2);
			MAT dEde2 = axial(e1);

			MAT dfde1 = dfdN * dNdE * dEde1;
			MAT dfde2 = dfdN * dNdE * dEde2;

			int offset = 0;
			for (int vi = 0; vi < dim; ++vi) {
				for (int i = 0; i < dim; ++i) for (int j = 0; j < dim; ++j) {
					triplets[tripletStartInd[idx] + offset++] = std::move(Eigen::Triplet<T>(
						elemVInd[vi] * dim + i, elemVInd[1] * dim + j, dfde1(i, j)));
					triplets[tripletStartInd[idx] + offset++] = std::move(Eigen::Triplet<T>(
						elemVInd[vi] * dim + i, elemVInd[2] * dim + j, dfde2(i, j)));
					triplets[tripletStartInd[idx] + offset++] = std::move(Eigen::Triplet<T>(
						elemVInd[vi] * dim + i, elemVInd[0] * dim + j, - (dfde1(i, j) + dfde2(i, j))));
				}
			}
		}
	});
}

}
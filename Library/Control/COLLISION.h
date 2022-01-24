#pragma once

#include <Math/VECTOR.h>
#include <FEM/DATA_TYPE.h>

namespace JGSL
{

template <class T, int dim>
void Compute_Plane_Collision_Potential(
	GROUND<T, dim>& plane,
	MESH_NODE<T, dim>& X,
	MESH_NODE<T, dim>& Xn,
	T h,
	T& E)
{
	using VEC = Eigen::Matrix<T, dim, 1>;

	plane.Each([&](int i, auto data) {
		auto &[p, mu, kn, kf, origin, dir] = data;
		X.Join(Xn).Each([&](int idx, auto data) {
			auto &[x, xn] = data;
			T depth = -(x - origin).dot(dir);
			if (depth > 0) {
				E += h * h * kn / p * std::pow(depth, p);

				if (mu > 0) {
					VEC v = (x - xn).to_eigen() / h;
					VEC us = v - v.dot(dir.to_eigen()) * dir.to_eigen();
					T fn = kn * std::pow(depth, p - 1);

					if (kf * us.norm() < mu * fn) {
						E += h * h * 0.5 * kf * us.dot(us);
					}
					else {
						E += h * h * (mu * fn * us.norm() - mu * mu * fn * fn / (2 * kf));
					}
				}
			}
		});
	});
}

template <class T, int dim>
void Compute_Plane_Collision_Gradient(
	GROUND<T, dim>& plane,
	MESH_NODE<T, dim>& X,
	MESH_NODE<T, dim>& Xn,
	T h,
	MESH_NODE_ATTR<T, dim>& nodeAttr)
{
	using VEC = Eigen::Matrix<T, dim, 1>;
	using MAT = Eigen::Matrix<T, dim, dim>;

	plane.Each([&](int i, auto data) {
		auto &[p, mu, kn, kf, origin, dir] = data;
		X.Join(Xn).Par_Each([&](int idx, auto data) {
			auto &[x, xn] = data;
			T depth = -(x - origin).dot(dir);
			if (depth > 0) {
				VEC local_g = - kn * std::pow(depth, p - 1) * dir.to_eigen();

				if (mu > 0) {
					VEC v = (x - xn).to_eigen() / h;
					VEC us = v - v.dot(dir.to_eigen()) * dir.to_eigen();
					T fn = kn * std::pow(depth, p - 1);
					MAT D = MAT::Identity() - dir.to_eigen() * dir.to_eigen().transpose();

					if (kf * us.norm() < mu * fn) {
						local_g += kf * D * us;
					}
					else {
						local_g += mu * fn * D * us.normalized();
					}
				}

				VECTOR<T, dim>& g = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(idx));
				for (int d = 0; d < dim; ++d) {
					g[d] += h * h * local_g[d];
				}
			}
		});
	});
}

template <class T, int dim>
void Compute_Plane_Collision_Hessian(
	GROUND<T, dim>& plane,
	MESH_NODE<T, dim>& X,
	MESH_NODE<T, dim>& Xn,
	T h,
	std::vector<Eigen::Triplet<T>>& triplets)
{
	using VEC = Eigen::Matrix<T, dim, 1>;
	using MAT = Eigen::Matrix<T, dim, dim>;

	plane.Each([&](int i, auto data) {
		auto &[p, mu, kn, kf, origin, dir] = data;
		X.Join(Xn).Each([&](int idx, auto data) {
			auto &[x, xn] = data;
			T depth = -(x - origin).dot(dir);
			if (depth > 0) {
				MAT H = kn * (p - 1) * std::pow(depth, p - 2) * dir.to_eigen() * dir.to_eigen().transpose();

				if (mu > 0) {
					VEC v = (x - xn).to_eigen() / h;
					VEC us = v - v.dot(dir.to_eigen()) * dir.to_eigen();
					T fn = kn * std::pow(depth, p - 1);
					MAT D = MAT::Identity() - dir.to_eigen() * dir.to_eigen().transpose();

					if (kf * us.norm() < mu * fn) {
						H += kf * D * D;
					}
					else {
						H += mu * fn / us.norm() * D * D;
					}
				}

				int base = triplets.size();
				triplets.resize(base + dim * dim);
				for (int bi = 0; bi < dim; ++bi) for (int bj = 0; bj < dim; ++bj) {
					triplets[base++] = Eigen::Triplet<T>(idx * dim + bi, idx * dim + bj, h * h * H(bi, bj));
				}
			}
		});
	});
}

template <class T, int dim>
void Add_Plane(int p, T mu, T kn, T kf, const VECTOR<T, dim>& origin, const VECTOR<T, dim>& dir, GROUND<T, dim>& planes)
{
	planes.Insert(planes.size, p, mu, kn, kf, origin, dir);
}

}
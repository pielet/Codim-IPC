#pragma once

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
	plane.Each([&](int i, auto data) {
		auto &[p, mu, kn, kf, origin, dir] = data;
		X.Join(Xn).Each([&](int j, auto data) {
			auto &[x, xn] = data;
			T depth = -(x - origin).dot(dir);
			if (depth > 0) {
				E += h * h * kn / p * std::pow(depth, p);

				if (mu > 0) {
					VECTOR<T, dim> v = (x - xn) / h;
					VECTOR<T, dim> us = v - v.dot(dir) * dir;
					T fn = kn * std::pow(depth, p - 1);

					if (kf * us.length() < mu * kn) {
						E += h * h * 0.5 * kf * us.length2();
					}
					else {
						E += h * h * mu * fn * us.length() - mu * mu * fn * fn / (2 * kf);
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
	plane.Each([&](int i, auto data) {
		auto &[p, mu, kn, kf, origin, dir] = data;
		X.Join(Xn, nodeAttr).Par_Each([&](int j, auto data) {
			auto &[x, xn, x0, v, g, m] = data;
			T depth = -(x - origin).dot(dir);
			if (depth > 0) {
				g += h * h * kn * std::pow(depth, p - 1) * dir;

				if (mu > 0) {
					VECTOR<T, dim> v = (x - xn) / h;
					VECTOR<T, dim> us = v - v.dot(dir) * dir;
					T fn = kn * std::pow(depth, p - 1);
					MATRIX<T, dim> D = MATRIX<T, dim>(1.0) - outer_product(dir, dir);

					if (kf * us.length() < mu * fn) {
						g += h * h * kf * D * us;
					}
					else {
						g += h * h * mu * fn * D * us.Normalized();
					}
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
	plane.Each([&](int i, auto data) {
		auto &[p, mu, kn, kf, origin, dir] = data;
		X.Join(Xn).Par_Each([&](int idx, auto data) {
			auto &[x, xn] = data;
			T depth = -(x - origin).dot(dir);
			if (depth > 0) {
				MATRIX<T, dim> H = kn * (p - 1) * std::pow(depth, p - 2) * outer_product(dir, dir);

				if (mu > 0) {
					VECTOR<T, dim> v = (x - xn) / h;
					VECTOR<T, dim> us = v - v.dot(dir) * dir;
					T fn = kn * std::pow(depth, p - 1);
					MATRIX<T, dim> D = MATRIX<T, dim>(1.0) - outer_product(dir, dir);

					if (kf * us.length() < mu * fn) {
						H += kf * D * D;
					}
					else {
						H += mu * fn / us.length() * D * D;
					}
				}

				int base = triplets.size();
				triplets.resize(base + dim * dim);
				for (int bi = 0; bi < dim; ++bi) for (int bj = 0; bj < dim; ++ bj) {
					triplets[base++] = Eigen::Triplet<T>(idx * dim + bi, idx * dim + bj, h * h * H(bi, bj));
				}
			}
		});
	});
}

template <class T, int dim>
void Add_Plane(int p, T mu, T kn, T kf, const VECTOR<T, dim>& origin, const VECTOR<T, dim>& dir, GROUND<T, dim>& planes)
{
	planes.Append(p, mu, kn, kf, origin, dir);
}

}
#pragma once

#include <pybind11/pybind11.h>
#include <Math/CSR_MATRIX.h>
#include <FEM/DATA_TYPE.h>
#include <Control/BOUNDARY_CONDITION.h>
#include <Control/COLLISION.h>

namespace JGSL {

template <class T, int dim>
void Fill(MESH_NODE<T, dim>& vec, int size)
{
	vec.Reserve(size);
	for (int i = 0; i < size; ++i) {
		vec.Append(VECTOR<T, dim>(T(0)));
	}
}

template <class T, int dim>
void Append_NodeAttr(MESH_NODE_ATTR<T, dim>& src, MESH_NODE_ATTR<T, dim>& dst)
{
	dst.Reserve(dst.size + src.size);
	src.Each([&](int id, auto data) {
		auto &[x0, v, g, m] = data;
		dst.Append(x0, v, g, m);
	});
}

template <class T, int dim>
void SetZero(MESH_NODE<T, dim>& X)
{
	X.Par_Each([](int idx, auto data) {
		auto &[x] = data;
		x.setZero();
	});
}

template <class T, int dim>
void SetVelocity(MESH_NODE_ATTR<T, dim>& nodeAttr, VECTOR<T, dim>& init_v)
{
	nodeAttr.Par_Each([&](int id, auto data) {
		auto &[x0, v, g, m] = data;
		v = init_v;
	});
}

template <class T, int dim>
void GetFrame(int fi, MESH_NODE<T, dim>& frames, MESH_NODE<T, dim>& one_frame)
{
	one_frame.Par_Each([&](int idx, auto data) {
		auto &[x] = data;
		x = std::get<0>(frames.Get_Unchecked_Const(fi * one_frame.size + idx));
	});
}

template <class T, int dim>
void SetFrame(int fi, MESH_NODE<T, dim>& frames, MESH_NODE<T, dim>& one_frame)
{
	one_frame.Par_Each([&](int idx, auto data) {
		auto &[x] = data;
		frames.update(fi * one_frame.size + idx, x);
	});
}

template <class T, int dim>
T Reduce(MESH_NODE<T, dim>& x, MESH_NODE<T, dim>& y)
{
	T sum = 0;
	x.Join(y).Each([&](int idx, auto data) {
		auto &[vx, vy] = data;
		sum += vx.dot(vy);
	});
	return sum;
}

template <class T, int dim>
void Axpy(T a, MESH_NODE<T, dim>& x, MESH_NODE<T, dim>& y)
{
	y.Join(x).Par_Each([&](int idx, auto data) {
		auto &[vy, vx] = data;
		vy += a * vx;
	});
}

template <class T, int dim>
void Copy(MESH_NODE<T, dim>& x, MESH_NODE<T, dim>& y)
{
	x.deep_copy_to(y);
}

template <class T, int dim>
void Scale(MESH_NODE<T, dim>& X, T a)
{
	X.Par_Each([&](int idx, auto data) {
		auto &[x] = data;
		x *= a;
	});
}

template <class T, int dim>
void node_to_eigen(MESH_NODE<T, dim>& node, Eigen::VectorXd& eigen)
{
	node.Par_Each([&](int idx, auto data) {
		auto &[x] = data;
		for (int d = 0; d < dim; ++d) {
			eigen[idx * dim + d] = x[d];
		}
	});
}

template <class T, int dim>
void eigen_to_node(const Eigen::VectorXd& eigen, MESH_NODE<T, dim>& node)
{
	node.Par_Each([&](int idx, auto data) {
		auto &[x] = data;
		for (int d = 0; d < dim; ++d) {
			x[d] = eigen[idx * dim + d];
		}
	});
}

template <class T, int dim>
void node_to_vector(MESH_NODE<T, dim>& node, std::vector<T>& vec, T a = 1.0)
{
	node.Par_Each([&](int idx, auto data) {
		auto &[x] = data;
		for (int d = 0; d < dim; ++d) {
			vec[idx * dim + d] = a * x[d];
		}
	});
}

template <class T, int dim>
void vector_to_node(const Eigen::VectorXd& vec, MESH_NODE<T, dim>& node)
{
	node.Par_Each([&](int idx, auto data) {
		auto &[x] = data;
		for (int d = 0; d < dim; ++d) {
			x[d] = vec[idx * dim + d];
		}
	});
}

template <class T, int dim>
void Read(MESH_NODE<T, dim>& X, const std::string& filename)
{
	std::ifstream is(filename);
	if (!is.is_open()) {
		std::cerr << filename << " not found!";
		exit(-1);
	}

	for (int i = 0; i < X.size; ++i) {
		VECTOR<T, dim>& x = std::get<0>(X.Get_Unchecked(i));
		if constexpr (dim == 2) {
			is >> x[0] >> x[1];
		}
		else if constexpr (dim == 3) {
			is >> x[0] >> x[1] >> x[2];
		}
	}
}

template <class T, int dim>
void Write(MESH_NODE<T, dim>& X, const std::string& filename)
{
	TIMER_FLAG("Write_Optimization_Params");

	FILE* file = fopen(filename.c_str(), "w");
	if (!file) {
		puts("failed to create file");
		exit(-1);
	}
	X.Each([&](int idx, auto data) {
		auto &[x] = data;
		if constexpr (dim == 2) {
			fprintf(file, "%le %le\n", x[0], x[1]);
		}
		else if constexpr (dim == 3) {
			fprintf(file, "%le %le %le\n", x[0], x[1], x[2]);
		}
	});
	fclose(file);
}

template <class T, int dim>
void Print(MESH_NODE<T, dim>& x)
{
	x.Each([](int idx, auto data) {
		auto &[vec] = data;
		if constexpr (dim == 2) {
			printf("(%le, %le)\n", vec[0], vec[1]);
		}
		else {
			printf("(%le, %le, %le)\n", vec[0], vec[1], vec[2]);
		}
	});
}

template <class T>
void Add_Block(Eigen::SparseMatrix<T, Eigen::RowMajor>& A, std::vector<Eigen::Triplet<T>>& triplets, int base_i, int base_j, T a = 1.0, bool trans = false)
{
	int idx = triplets.size();
	triplets.resize(idx + A.nonZeros());

	for (int i = 0; i < A.outerSize(); ++i) {
		typename Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(A, i);
		for (; it; ++it) {
			if (trans) {
				triplets[idx] = Eigen::Triplet<T>(base_i + it.col(), base_j + it.row(), a * it.value());
			}
			else {
				triplets[idx] = Eigen::Triplet<T>(base_i + it.row(), base_j + it.col(), a * it.value());
			}
			++idx;
		}
	}
}

template <class T>
void Add_Block(const std::vector<Eigen::Triplet<T>>& A, std::vector<Eigen::Triplet<T>>& triplets, int base_i, int base_j, T a = 1.0, bool trans = false)
{
	int idx = triplets.size();
	triplets.resize(idx + A.size());

	for (const auto t : A) {
		if (trans) {
			triplets[idx++] = Eigen::Triplet<T>(base_i + t.col(), base_j + t.row(), a * t.value());
		}
		else {
			triplets[idx++] = Eigen::Triplet<T>(base_i + t.row(), base_j + t.col(), a * t.value());
		}
	}
}

template <class T>
void Add_Identity(std::vector<Eigen::Triplet<T>>& triplets, int base_i, int base_j, int size, T a)
{
	int idx = triplets.size();
	triplets.resize(idx + size);

	for (int i = 0; i < size; ++i) {
		triplets[idx] = Eigen::Triplet<T>(base_i + i, base_j + i, a);
		++idx;
	}
}

void Export_Control_Utils(py::module& m)
{
	m.def("Fill", &Fill<double, 3>);
	m.def("Fill", &Fill<double, 4>);
	m.def("SetVelocity", &SetVelocity<double, 3>);
	m.def("GetFrame", &GetFrame<double, 3>);
	m.def("SetFrame", &SetFrame<double, 3>);
	m.def("Reduce", &Reduce<double, 3>);
	m.def("Axpy", &Axpy<double, 3>);
	m.def("Copy", &Copy<double, 3>);
	m.def("Copy", &Copy<double, 4>);
	m.def("Scale", &Scale<double, 3>);
	m.def("Read", &Read<double, 3>);
	m.def("Write", &Write<double, 3>);
	m.def("Print", &Print<double, 3>);

	m.def("Set_Dirichlet", &Set_Dirichlet<double, 3>);
    m.def("Add_DBC_Motion", &Add_DBC_Motion<double, 3>);
    m.def("Update_Dirichlet", &Update_Dirichlet<double, 3>);

    m.def("Add_Plane", &Add_Plane<double, 3>);
}

}
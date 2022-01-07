#pragma once

#include <pybind11/pybind11.h>
#include <Math/CSR_MATRIX.h>
#include <FEM/DATA_TYPE.h>

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
void ZeroVelocity(
    MESH_NODE_ATTR<T, dim>& nodeAttr)
{
    nodeAttr.Par_Each([&](int id, auto data) {
        auto &[x0, v, g, m] = data;
        v.setZero();
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
void Copy(MESH_NODE<T, dim>& x, MESH_NODE<T, dim> y)
{
    x.deep_copy_to(y);
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
            printf("(%f, %f)\n", vec[0], vec[1]);
        }
        else {
            printf("(%f, %f, %f)\n", vec[0], vec[1], vec[2]);
        }
    });
}

void Export_Control_Utils(py::module& m)
{
    m.def("Fill", &Fill<double, 3>);
    m.def("ZeroVelocity", &ZeroVelocity<double, 3>);
    m.def("GetFrame", &GetFrame<double, 3>);
    m.def("SetFrame", &SetFrame<double, 3>);
    m.def("Reduce", &Reduce<double, 3>);
    m.def("Axpy", &Axpy<double, 3>);
    m.def("Copy", &Copy<double, 3>);
    m.def("Read", &Read<double, 3>);
    m.def("Write", &Write<double, 3>);
    m.def("Print", &Print<double, 3>);
}

}
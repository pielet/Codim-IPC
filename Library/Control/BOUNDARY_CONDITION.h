#pragma once

#include <Utils/MESHIO.h>
#include <iostream>

namespace py = pybind11;
namespace JGSL {

template<class T, int dim>
using DBC_MOTION_STORAGE = BASE_STORAGE<T, T, T, VECTOR<int, 2>, VECTOR<T, dim>, VECTOR<T, dim>, VECTOR<T, dim>, T>; // range, v, rotCenter, rotAxis, angVelDeg

template <class T, int dim>
VECTOR<int, 2> Set_Dirichlet(MESH_NODE<T, dim>& X,
    const VECTOR<T, dim>& relBoxMin,
    const VECTOR<T, dim>& relBoxMax,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    const VECTOR<int, 4>& vIndRange = VECTOR<int, 4>(0, 0, __INT_MAX__, -1))
{
    if (!X.size) {
        puts("no nodes in the model!");
        exit(-1);
    }

    VECTOR<T, dim> bboxMin;
    VECTOR<T, dim> bboxMax;
    X.Each([&](int id, auto data) {
        if (id >= vIndRange[0] && id < vIndRange[2]) {
            auto &[x] = data;
            if (id == vIndRange[0]) {
                bboxMin = x;
                bboxMax = x;
            }
            else {
                for (int dimI = 0; dimI < dim; ++dimI) {
                    if (bboxMax(dimI) < x(dimI)) {
                        bboxMax(dimI) = x(dimI);
                    }
                    if (bboxMin(dimI) > x(dimI)) {
                        bboxMin(dimI) = x(dimI);
                    }
                }
            }
        }
    });

    VECTOR<T, dim> rangeMin = relBoxMin;
    VECTOR<T, dim> rangeMax = relBoxMax;
    for (int dimI = 0; dimI < dim; ++dimI) {
        rangeMin(dimI) *= bboxMax(dimI) - bboxMin(dimI);
        rangeMin(dimI) += bboxMin(dimI);
        rangeMax(dimI) *= bboxMax(dimI) - bboxMin(dimI);
        rangeMax(dimI) += bboxMin(dimI);
    }

    std::cout << "DBC node inds: ";
    VECTOR<int, 2> range;
    range[0] = DBC.size;
    int DBCCount = DBC.size;
    X.Each([&](int id, auto data) {
        if (id >= vIndRange[0] && id < vIndRange[2]) {
            auto &[x] = data;
            if constexpr (dim == 3) {
                if (x(0) >= rangeMin(0) && x(0) <= rangeMax(0) &&
                    x(1) >= rangeMin(1) && x(1) <= rangeMax(1) &&
                    x(2) >= rangeMin(2) && x(2) <= rangeMax(2))
                {
                    T r = x.length();
                    if (r >= PARAMETER::Get("Dirichlet_ring", (T)0.))
                        DBC.Insert(DBCCount++, VECTOR<T, dim + 1>(id, x(0), x(1), x(2)));
                    std::cout << " " << id;
                }
            }
            else {
                if (x(0) >= rangeMin(0) && x(0) <= rangeMax(0) &&
                    x(1) >= rangeMin(1) && x(1) <= rangeMax(1))
                {
                    DBC.Insert(DBCCount++, VECTOR<T, dim + 1>(id, x(0), x(1)));
                    std::cout << " " << id;
                }
            }
        }
    });
    range[1] = DBCCount;

    return range;
}

template <class T, int dim>
void Add_DBC_Motion(
    T begin, T end, T ease_ratio,
    const VECTOR<int, 2>& range,
    const VECTOR<T, dim>& dist,
    const VECTOR<T, dim>& rotCenter,
    const VECTOR<T, dim>& rotAxis,
    T angle,
    DBC_MOTION_STORAGE<T, dim>& DBCMotion)
{
    DBCMotion.Insert(DBCMotion.size, begin, end, ease_ratio, range, dist, rotCenter, rotAxis, angle);
}

template <class T>
T cubic_ease_func(T t, T t0, T t1, T ease_range, T L)
{
    T ta = t0 + ease_range * (t1 - t0);
    T tb = t1 - ease_range * (t1 - t0);

    T yh = (L * 2.0) / (t1 - t0 + tb - ta);
    if (t < t0 || t > t1)
        return 0.0;
    else {
        if (t < ta)
            return (yh * (t0 - t) * (t0 - t) * (t0 - 3.0 * ta + 2.0 * t)) /
                ((t0 - ta) * (t0 - ta) * (t0 - ta));
        else if (t > tb)
            return (yh * (t1 - t) * (t1 - t) * (t1 - 3.0 * tb + 2.0 * t)) /
                ((t1 - tb) * (t1 - tb) * (t1 - tb));
        else
            return yh;
    }
}

template <class T, int dim>
void Update_Dirichlet(
    T cur_t,
    DBC_MOTION_STORAGE<T, dim>& DBCMotion,
    T h, VECTOR_STORAGE<T, dim + 1>& DBC)
{
    DBCMotion.Each([&](int id, auto data) {
        auto &[begin, end, ease_ratio, range, L, rotCenter, rotAxis, angle] = data;

        VECTOR<T, dim> v;
        for (int d = 0; d < dim ;++d) {
            v[d] = cubic_ease_func(cur_t, begin, end, ease_ratio, L[d]);
        }
        T angVelDeg = cubic_ease_func(cur_t, begin, end, ease_ratio, angle);

#ifdef VERBOSE
        if (cur_t < end && cur_t > begin) {
            if constexpr (dim == 3) {
                printf("step DBC: velocity (%f, %f, %f) angle_velocity: %f\n", v[0], v[1], v[2], angVelDeg);
            }
        }
#endif

        //TODO: parallel the following loop
        for (int i = range[0]; i < range[1]; ++i) {
            VECTOR<T, dim + 1>& dbcI = std::get<0>(DBC.Get_Unchecked(i));
            
            if (angVelDeg) {
                if constexpr (dim == 2) {
                    T rotAngRad = angVelDeg / 180 * M_PI * h;
                    MATRIX<T, dim> rotMtr;
                    rotMtr(0, 0) = std::cos(rotAngRad);
                    rotMtr(0, 1) = -std::sin(rotAngRad);
                    rotMtr(1, 0) = -rotMtr(0, 1);
                    rotMtr(1, 1) = rotMtr(0, 0);

                    VECTOR<T, dim> x(dbcI[1] - rotCenter[0], dbcI[2] - rotCenter[1]);
                    VECTOR<T, dim> rotx = rotMtr * x;
                    dbcI[1] = rotx[0] + rotCenter[0];
                    dbcI[2] = rotx[1] + rotCenter[1];
                }
                else {
                    T rotAngRad = angVelDeg / 180 * M_PI * h;
                    const Eigen::Matrix3d rotMtr = Eigen::AngleAxis<double>(rotAngRad,
                        Eigen::Vector3d(rotAxis[0], rotAxis[1], rotAxis[2])).toRotationMatrix();
                    
                    const Eigen::Vector3d x(dbcI[1] - rotCenter[0], dbcI[2] - rotCenter[1], dbcI[3] - rotCenter[2]);
                    const Eigen::Vector3d rotx = rotMtr * x;
                    dbcI[1] = rotx[0] + rotCenter[0];
                    dbcI[2] = rotx[1] + rotCenter[1];
                    dbcI[3] = rotx[2] + rotCenter[2];
                }
            }

            dbcI[1] += v[0] * h;
            dbcI[2] += v[1] * h;
            if constexpr (dim == 3) {
                dbcI[3] += v[2] * h;
            }
        }
    });
}

}

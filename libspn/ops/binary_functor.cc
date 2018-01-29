#include "tensorflow/core/kernels/cwise_ops.h"

//#include "tensorflow/core/framework/op.h"
//#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/framework/tensor_types.h"
//#include "tensorflow/core/framework/variant_op_registry.h"
//#include "tensorflow/core/platform/logging.h"
//#include "tensorflow/core/util/bcast.h"

#define EIGEN_USE_THREADS

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif

namespace functor {

template <typename D, typename Out, typename Rhs>
void Assign(const D& d, Out out, Rhs rhs) {
  out.device(d) = rhs;
}

// Partial specialization of BinaryFunctor<Device=CPUDevice, Functor, NDIMS>
// for functors with with no error checking.
template <typename Functor, int NDIMS>
struct BinaryFunctor<CPUDevice, Functor, NDIMS, false> {
  void operator()(const CPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1, bool* error) {
    Assign(d, out, in0.binaryExpr(in1, typename Functor::func()));
  }

  void Left(const CPUDevice& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in, bool* error) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_left<Tout, Tin, Binary> Unary;
    Assign(d, out, in.unaryExpr(Unary(scalar.data())));
  }

  void Right(const CPUDevice& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar, bool* error) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_right<Tout, Tin, Binary> Unary;
    Assign(d, out, in.unaryExpr(Unary(scalar.data())));
  }

#if !defined(EIGEN_HAS_INDEX_LIST)
  inline Eigen::DSizes<int, 2> NByOne(int n) {
    return Eigen::DSizes<int, 2>(n, 1);
  }
  inline Eigen::DSizes<int, 2> OneByM(int m) {
    return Eigen::DSizes<int, 2>(1, m);
  }
#else
  inline Eigen::IndexList<int, Eigen::type2index<1>> NByOne(int n) {
    Eigen::IndexList<int, Eigen::type2index<1>> ret;
    ret.set(0, n);
    return ret;
  }
  inline Eigen::IndexList<Eigen::type2index<1>, int> OneByM(int m) {
    Eigen::IndexList<Eigen::type2index<1>, int> ret;
    ret.set(1, m);
    return ret;
  }
#endif

  void BCast(const CPUDevice& dev,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1,
             bool* error) {
    typedef typename Functor::in_type T;
    typename Functor::func func;
    if ((NDIMS == 2) && Functor::use_bcast_optimization &&
        use_bcast_optimization<T>::value) {
      // Optimize for speed by using Eigen::type2index and avoid
      // .broadcast() when we know its a no-op.
      //
      // Here, we need to handle 6 cases depending on how many "1"
      // exist in in0 and in1's shapes (4 numbers in total). It's not
      // possible that two shapes have more than 2 1s because those
      // are simplified to NDIMS==1 case.
      //
      // Because this optimization increases the binary size for each
      // Functor (+, -, *, /, <, <=, etc.), type and ndim combination.
      // we only apply such optimization for selected ops/types/ndims.
      //
      // Because NDIMS, Functor::use_broadcast_optimization and
      // use_broadcast_optimization<T> are compile-time constant, gcc
      // does a decent job avoiding generating code when conditions
      // are not met.
      const int a = in0.dimension(0);  // in0 is shape [a, b]
      const int b = in0.dimension(1);
      const int c = in1.dimension(0);  // in1 is shape [c, d]
      const int d = in1.dimension(1);
      if ((a == 1) && (d == 1)) {
        auto lhs = in0.reshape(OneByM(b)).broadcast(NByOne(c));
        auto rhs = in1.reshape(NByOne(c)).broadcast(OneByM(b));
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
      if ((b == 1) && (c == 1)) {
        auto lhs = in0.reshape(NByOne(a)).broadcast(OneByM(d));
        auto rhs = in1.reshape(OneByM(d)).broadcast(NByOne(a));
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
      if (a == 1) {
        auto lhs = in0.reshape(OneByM(b)).broadcast(NByOne(c));
        auto rhs = in1;
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
      if (b == 1) {
        auto lhs = in0.reshape(NByOne(a)).broadcast(OneByM(d));
        auto rhs = in1;
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
      if (c == 1) {
        auto lhs = in0;
        auto rhs = in1.reshape(OneByM(d)).broadcast(NByOne(a));
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
      if (d == 1) {
        auto lhs = in0;
        auto rhs = in1.reshape(NByOne(c)).broadcast(OneByM(b));
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }

      const bool bcast0_all_one = AllOne<NDIMS>(bcast0);
      const bool bcast1_all_one = AllOne<NDIMS>(bcast1);
      if (bcast0_all_one && !bcast1_all_one) {
        auto lhs = in0;  // No need to do broadcast for in0
        auto rhs = in1.broadcast(bcast1);
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }

      if (!bcast0_all_one && bcast1_all_one) {
        auto lhs = in0.broadcast(bcast0);
        auto rhs = in1;  // No need to do broadcast for in1
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
    }

    // Fallback path. Always work and probably slower.
    auto lhs = in0.broadcast(bcast0);
    auto rhs = in1.broadcast(bcast1);
    Assign(dev, out, lhs.binaryExpr(rhs, func));
  }
};

// Version of BinaryFunctor with error handling.
template <typename Functor, int NDIMS>
struct BinaryFunctor<CPUDevice, Functor, NDIMS, true> {
  void operator()(const CPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1, bool* error) {
    Assign(d, out, in0.binaryExpr(in1, typename Functor::func(error)));
  }

  void Left(const CPUDevice& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in, bool* error) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_left<Tout, Tin, Binary> Unary;
    Assign(d, out, in.unaryExpr(Unary(scalar.data(), error)));
  }

  void Right(const CPUDevice& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar, bool* error) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_right<Tout, Tin, Binary> Unary;
    Assign(d, out, in.unaryExpr(Unary(scalar.data(), error)));
  }

  void BCast(const CPUDevice& dev,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1,
             bool* error) {
    typename Functor::func func(error);
    auto lhs = in0.broadcast(bcast0);
    auto rhs = in1.broadcast(bcast1);
    Assign(dev, out, lhs.binaryExpr(rhs, func));
  }
};

// Partial specialization of UnaryFunctor<Device=CPUDevice, Functor>.
template <typename Functor>
struct UnaryFunctor<CPUDevice, Functor> {
  void operator()(const CPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in) {
    Assign(d, out, in.unaryExpr(typename Functor::func()));
  }
};

// Partial specialization of ApproximateEqual<Device=CPUDevice, T>.
template <typename T>
struct ApproximateEqual<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstFlat x,
                  typename TTypes<T>::ConstFlat y, T tolerance,
                  typename TTypes<bool>::Flat z) {
    auto diff = x - y;
    z.device(d) = diff.abs() <= tolerance;
  }
};

}  // end namespace functor

};


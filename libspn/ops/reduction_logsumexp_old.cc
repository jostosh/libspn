#define EIGEN_USE_THREADS

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/kernels/cwise_ops.h"

// TODO below gives compile error, but implements BinaryFunctor for CPU
// TODO if it's not included, we don't get the definition of BinaryFunctor<ThreadPoolDevice,...> in the .so file
// TODO this leads to runtime errors!

// TODO perhaps a workaround is to make a local file that implements a BinaryFunctor, (CHECK, that works)
// Including it conflicts with numeric op, which is also needed itself...
// #include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/kernels/reduction_ops.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/bcast.h"


// cwise_ops_common for functor::sub
#include "tensorflow/core/kernels/reduction_ops_common.h"
// #include "tensorflow/core/kernels/reduction_ops_mmon.h"

namespace tensorflow
{
using shape_inference::ShapeHandle;
using shape_inference::InferenceContext;


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


typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL


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

// For operations where the output is a reduction function along some
// dimensions of the input.
template <typename Device, class T, typename Tperm>
class ReductionLogSumExpOp : public OpKernel {
 public:
  explicit ReductionLogSumExpOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    const DataType dt = DataTypeToEnum<T>::v();
    const DataType pt = DataTypeToEnum<Tperm>::v();
    OP_REQUIRES_OK(ctx, ctx->MatchSignature({dt, pt}, {dt}));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& data = ctx->input(0);
    const Tensor& axes = ctx->input(1);
    VLOG(1) << "data shape: " << data.shape().DebugString();
    VLOG(1) << "axes      : " << axes.SummarizeValue(10);

    ReductionHelper helper;
    ReductionHelper maxhelper;
    OP_REQUIRES_OK(ctx, helper.Simplify(data, axes, keep_dims_));
    OP_REQUIRES_OK(ctx, maxhelper.Simplify(data, axes, true));
    CHECK_GE(helper.ndims(), 0);
    CHECK_GE(maxhelper.ndims(), 0);

    if (helper.ndims() == 0 ||
        (helper.ndims() == 1 && !helper.reduce_first_axis())) {
      // Special case. Reduces nothing.  It is unclear why this is
      // necessary, but tests fail without it.  Look into why this
      // case occurs.
      Tensor out;
      if (!out.CopyFrom(data, helper.out_shape())) {
        ctx->SetStatus(errors::Internal("Error during reduction copy."));
      }
      ctx->set_output(0, out);
      return;
    }

    // We must allocate temp tensors using the same alloc attr as
    // output(0) because it is returned as output(0) in the end.
    const AllocatorAttributes alloc_attr = ctx->output_alloc_attr(0);

    // A temporary tensor whose size matches the size of the reduced
    // output.
    Tensor tmp_out;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(ctx->expected_output_dtype(0),
                                helper.out_reshape(), &tmp_out, alloc_attr));
    Tensor max_out;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(ctx->expected_output_dtype(0),
                                maxhelper.out_reshape(), &max_out, alloc_attr));

    CHECK_EQ(helper.reduce_first_axis(), 0);

    auto max_flat = max_out.flat<T>();
    auto tmp_flat = tmp_out.flat<T>();

    typedef Eigen::internal::SumReducer<T> SumReducer;
    typedef Eigen::internal::MaxReducer<T> MaxReducer;
    typedef functor::ReduceFunctor<Device, SumReducer> SumFunctor;
    typedef functor::ReduceFunctor<Device, MaxReducer> MaxFunctor;
    typedef functor::sub<T> SubtractFunctor;

    Constants<Device> constants;
    const Device& d = ctx->eigen_device<Device>();
    MaxReducer max_reducer;
    SumReducer sum_reducer;
    SubtractFunctor subtracter;

    Tensor max;
    Tensor subtracted;

    // taken from tensorflow/core/kernels/cwise_ops_common.h
    // TODO this line might be important
    bool error = false;
    bool* const error_ptr = SubtractFunctor::has_errors ? &error : nullptr;

    typedef typename SubtractFunctor::in_type Tin;    // Input scalar data type.
    typedef typename SubtractFunctor::out_type Tout;  // Output scalar data type.

    if (tmp_out.NumElements() == 0) {
      // Nothing to do, fall through to final reshaping.
    }
    else if ((helper.ndims() == 2) && !helper.reduce_first_axis()) {
      // Can be viewed as a reduction of a matrix along 2nd dimension.
      MaxFunctor::Reduce(ctx, helper.out<T, 1>(&max_out), helper.in<T, 2>(data),
                         constants.kOne, max_reducer);

      // Broadcasted shape should match input's shape
      if (!max.CopyFrom(max_out, maxhelper.out_shape())) {
          ctx->SetStatus(errors::Internal("Error during reduction copy."));
      }

      // Below taken from tensorflow/core/kernels/cwise_ops_common.cc
      BCast bcast(BCast::FromShape(data.shape()), BCast::FromShape(max.shape()));
      if (!bcast.IsValid()) {
       ctx->SetStatus(errors::InvalidArgument("Incompatible shapes: ",
                                              data.shape().DebugString(), " vs. ",
                                              max_out.shape().DebugString()));
       return;
      }
      const TensorShape subtracted_shape = BCast::ToShape(bcast.output_shape());
      OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(ctx->expected_output_dtype(0),
        subtracted_shape, &subtracted, alloc_attr));
//      auto max_eig = max.template shaped<Tin, 2>(bcast.y_reshape());
//      auto subtracted_reshaped = subtracted.shaped<Tin, 2>(bcast.result_shape());
//      auto data_reshaped = data.shaped<Tin, 2>(bcast.x_reshape());
//
//      auto rhs = max_eig.reshape(NByOne(max_eig.dimension(0))).broadcast(OneByM(data_reshaped.dimension(1)));
//
//      subtracted_reshaped.device(d) = data_reshaped.binaryExpr(rhs, Eigen::internal::scalar_difference_op<T>());
//      subtracted_reshaped.device(d) = subtracted_reshaped.unaryExpr(Eigen::internal::scalar_exp_op<T>());

      const Tensor &const_max = max;

      auto arg1 = subtracted.shaped<Tout, 2>(bcast.result_shape());
      auto arg2 = data.shaped<Tin, 2>(bcast.x_reshape());
      auto arg3 = BCast::ToIndexArray<2>(bcast.x_bcast());
      auto arg4 = const_max.shaped<Tin, 2>(bcast.y_reshape());
      auto arg5 = BCast::ToIndexArray<2>(bcast.y_bcast());
      functor::BinaryFunctor<Device, SubtractFunctor, 2>().BCast(
          d, subtracted.shaped<Tout, 2>(bcast.result_shape()),
          data.shaped<Tin, 2>(bcast.x_reshape()),
          BCast::ToIndexArray<2>(bcast.x_bcast()),
          const_max.shaped<Tin, 2>(bcast.y_reshape()),
          BCast::ToIndexArray<2>(bcast.y_bcast()), error_ptr);

      SumFunctor::Reduce(ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 2>(subtracted),
                         constants.kOne, sum_reducer);
      auto out_normal = tmp_out.flat<T>();
      out_normal.device(d) = out_normal.unaryExpr(Eigen::internal::scalar_log_op<T>());
      auto max_flat = max.flat<T>();

      out_normal.device(d) = out_normal.binaryExpr(max_flat, Eigen::internal::scalar_sum_op<T>());
    }

    Tensor out;
    if (!out.CopyFrom(tmp_out, helper.out_shape())) {
      ctx->SetStatus(errors::Internal("Error during reduction copy."));
    }
    ctx->set_output(0, out);
  }

 private:
  // True if the number of dimensions should be maintained.
  bool keep_dims_;
};

REGISTER_OP("LogSumExp")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);


#define REGISTER_CPU_KERNELS(type)                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("LogSumExp")                                                        \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int32>("Tidx"),                                      \
      ReductionLogSumExpOp<CPUDevice, type, int32>);                           \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("LogSumExp")                                                        \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int64>("Tidx"),                                      \
      ReductionLogSumExpOp<CPUDevice, type, int64>);
//TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
REGISTER_CPU_KERNELS(float)
REGISTER_CPU_KERNELS(double)
#undef REGISTER_CPU_KERNELS

//
//#if GOOGLE_CUDA
//
//#define REGISTER_GPU_KERNELS(type)                                             \
//  REGISTER_KERNEL_BUILDER(                                                     \
//      Name("LogSumExp")                                                        \
//          .Device(DEVICE_GPU)                                                  \
//          .TypeConstraint<type>("T")                                           \
//          .TypeConstraint<int32>("Tidx")                                       \
//          .HostMemory("reduction_indices"),                                    \
//      ReductionLogSumExpOp<GPUDevice, type, int32>);                           \
//  REGISTER_KERNEL_BUILDER(                                                     \
//      Name("LogSumExp")                                                        \
//          .Device(DEVICE_GPU)                                                  \
//          .TypeConstraint<type>("T")                                           \
//          .TypeConstraint<int64>("Tidx")                                       \
//          .HostMemory("reduction_indices"),                                    \
//      ReductionLogSumExpOp<GPUDevice, type, int64>);
//TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
//TF_CALL_complex64(REGISTER_GPU_KERNELS);
//TF_CALL_complex128(REGISTER_GPU_KERNELS);
//#undef REGISTER_GPU_KERNELS
//
//// A special GPU kernel for int32.
//// TODO(b/25387198): Also enable int32 in device memory. This kernel
//// registration requires all int32 inputs and outputs to be in host memory.
//REGISTER_KERNEL_BUILDER(
//    Name("LogSumExp")
//        .Device(DEVICE_GPU)
//        .TypeConstraint<int32>("T")
//        .TypeConstraint<int32>("Tidx")
//        .HostMemory("input")
//        .HostMemory("output")
//        .HostMemory("reduction_indices"),
//    ReductionLogSumExpOp<CPUDevice, int32, int32>);
//REGISTER_KERNEL_BUILDER(
//    Name("LogSumExp")
//        .Device(DEVICE_GPU)
//        .TypeConstraint<int32>("T")
//        .TypeConstraint<int64>("Tidx")
//        .HostMemory("input")
//        .HostMemory("output")
//        .HostMemory("reduction_indices"),
//    ReductionLogSumExpOp<CPUDevice, int32, int64>);
//
//#endif
//
//#ifdef TENSORFLOW_USE_SYCL
//#define REGISTER_SYCL_KERNELS(type)                                        \
//  REGISTER_KERNEL_BUILDER(Name("LogSumExp")                                \
//                              .Device(DEVICE_SYCL)                         \
//                              .TypeConstraint<type>("T")                   \
//                              .TypeConstraint<int32>("Tidx")               \
//                              .HostMemory("reduction_indices"),            \
//                          ReductionLogSumExpOp<SYCLDevice, type, int32>);  \
//  REGISTER_KERNEL_BUILDER(Name("LogSumExp")                                \
//                              .Device(DEVICE_SYCL)                         \
//                              .TypeConstraint<type>("T")                   \
//                              .TypeConstraint<int64>("Tidx")               \
//                              .HostMemory("reduction_indices"),            \
//                          ReductionLogSumExpOp<SYCLDevice, type, int64>);
//REGISTER_SYCL_KERNELS(float);
//REGISTER_SYCL_KERNELS(double);
//
//REGISTER_KERNEL_BUILDER(
//    Name("LogSumExp")
//        .Device(DEVICE_SYCL)
//        .TypeConstraint<int32>("T")
//        .TypeConstraint<int32>("Tidx")
//        .HostMemory("input")
//        .HostMemory("output")
//        .HostMemory("reduction_indices"),
//    ReductionLogSumExpOp<CPUDevice, int32, int32>);
//REGISTER_KERNEL_BUILDER(
//    Name("LogSumExp")
//        .Device(DEVICE_SYCL)
//        .TypeConstraint<int32>("T")
//        .TypeConstraint<int64>("Tidx")
//        .HostMemory("input")
//        .HostMemory("output")
//        .HostMemory("reduction_indices"),
//    ReductionLogSumExpOp<CPUDevice, int32, int64>);
//#undef REGISTER_SYCL_KERNELS
//#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow

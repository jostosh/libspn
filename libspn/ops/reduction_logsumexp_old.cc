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
//#include "tensorflow/core/kernels/reduction_gpu_kernels.cu.h"
#include "binary_functor.h"


// cwise_ops_common for functor::sub
#include "tensorflow/core/kernels/reduction_ops_common.h"
// #include "tensorflow/core/kernels/reduction_ops_mmon.h"

namespace tensorflow
{
using shape_inference::ShapeHandle;
using shape_inference::InferenceContext;
using shape_inference::DimensionHandle;
using shape_inference::Dimension;

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
template <typename Device, class T>
class ReductionLogSumExpOp : public OpKernel {
 public:
  explicit ReductionLogSumExpOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), keep_dims_(true) {
    const DataType dt = DataTypeToEnum<T>::v();
  }

  void Compute(OpKernelContext* ctx) override {
    const DataType dt = DataTypeToEnum<T>::v();
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

    typedef Eigen::internal::SumReducer<T> SumReducer;
    typedef Eigen::internal::MaxReducer<T> MaxReducer;
    typedef functor::ReduceFunctor<Device, SumReducer> SumFunctor;
    typedef functor::ReduceFunctor<Device, MaxReducer> MaxFunctor;

    typedef functor::log<T> LogFunctor;
    typedef functor::exp<T> ExpFunctor;
    typedef functor::sub<T> SubtractFunctor;
    typedef functor::add<T> AddFunctor;

    Constants<Device> constants;
    const Device& d = ctx->eigen_device<Device>();
    MaxReducer max_reducer;
    SumReducer sum_reducer;

	Tensor max;
    Tensor subtracted;

    if (tmp_out.NumElements() == 0) {
      // Nothing to do, fall through to final reshaping.
    }
    else if ((helper.ndims() == 2) && !helper.reduce_first_axis()) {
      // Can be viewed as a reduction of a matrix along 2nd dimension.
      MaxFunctor::Reduce(ctx, helper.out<T, 1>(&max_out), helper.in<T, 2>(data),
                         constants.kOne, max_reducer);

      // Below taken from tensorflow/core/kernels/cwise_ops_common.cc
      BCast bcast(BCast::FromShape(data.shape()), BCast::FromShape(maxhelper.out_shape()));
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

      // Subtract max
      const Tensor &const_max = max_out;
	  bool error = false;
	  bool* const error_ptr = SubtractFunctor::has_errors ? &error : nullptr;
	  typedef typename SubtractFunctor::in_type Tin;    // Input scalar data type.
	  typedef typename SubtractFunctor::out_type Tout;  // Output scalar data type.
	  functor::BinaryFunctor<Device, SubtractFunctor, 2>().BCast(
          d, subtracted.shaped<Tout, 2>(bcast.result_shape()),
          data.shaped<Tin, 2>(bcast.x_reshape()),
          BCast::ToIndexArray<2>(bcast.x_bcast()),
          const_max.shaped<Tin, 2>(bcast.y_reshape()),
          BCast::ToIndexArray<2>(bcast.y_bcast()), error_ptr);

	  // Take exp of subtracted
	  // Select functor, construct it, and call it
      const Tensor &const_subtracted = subtracted;
      functor::UnaryFunctor<Device, ExpFunctor>()(d, subtracted.flat<T>(), const_subtracted.flat<T>());

      // Sum the subtracted and store in tmp_out
      SumFunctor::Reduce(ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 2>(subtracted),
                         constants.kOne, sum_reducer);

	  // Finally, we take the log
	  const Tensor &const_tmp_out = tmp_out;
      functor::UnaryFunctor<Device, LogFunctor>()(d, tmp_out.flat<T>(), const_tmp_out.flat<T>());

      // Then, we add the maxes
      typedef typename AddFunctor::in_type Tin_add;    // Input scalar data type.
      typedef typename AddFunctor::out_type Tout_add;  // Output scalar data type.
      bool error_add = false;
      bool* const error_add_ptr = AddFunctor::has_errors ? &error : nullptr;
      functor::BinaryFunctor<Device, AddFunctor, 1>()(
		  d, tmp_out.flat<T>(),
		  const_tmp_out.flat<T>(),
		  const_max.flat<T>(),
		  error_add_ptr
	  );
    }

    Tensor out;
    if (!out.CopyFrom(tmp_out, helper.out_shape())) {
      ctx->SetStatus(errors::Internal("Error during reduction copy."));
    }
    ctx->set_output(0, out);
  }

private:
    bool keep_dims_;
};

REGISTER_OP("LogSumExp")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .SetShapeFn([](InferenceContext* ctx) {
        ShapeHandle in_shape;
        TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 2, &in_shape));
        
        DimensionHandle out_last_dim;
        out_last_dim = ctx->MakeDim(1);

        ShapeHandle out_shape;
        TF_RETURN_IF_ERROR(
            ctx->ReplaceDim(in_shape, -1, out_last_dim, &out_shape));
        ctx->set_output(0, out_shape);

        return Status::OK();
    });



#define REGISTER_CPU_KERNELS(type)                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("LogSumExp")                                                        \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T"),                                         \
      ReductionLogSumExpOp<CPUDevice, type>);
//TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
REGISTER_CPU_KERNELS(float)
REGISTER_CPU_KERNELS(double)
#undef REGISTER_CPU_KERNELS



}  // namespace tensorflow

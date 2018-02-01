

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "reduction_logsumexp_functor.h"

namespace tensorflow
{
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

using shape_inference::ShapeHandle;
using shape_inference::InferenceContext;
using shape_inference::DimensionHandle;
using shape_inference::Dimension;

namespace functor {
template <typename Device, typename T>
struct LogsumexpFunctorBase {
  void operator()(const Device& d,
                  typename TTypes<T>::ConstMatrix &logits,
                  typename TTypes<T>::Matrix &max_logits,
                  typename TTypes<T>::Matrix &max_logits_safe,
                  typename TTypes<T>::Matrix &out) {
    LogsumexpEigenImpl<Device, T>::Compute(
      d, logits, max_logits, max_logits_safe, out);
  }
};
template <typename T>
struct LogsumexpFunctor<CPUDevice, T> : LogsumexpFunctorBase<CPUDevice, T> {};

}  // namespace functor

template <typename Device, typename T>
class LogsumexpOp : public OpKernel {
 public:
  explicit LogsumexpOp(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in_ = context->input(0);
    auto logits_in = logits_in_.flat_inner_dims<T>();

    // Making sure we have the right number of rows and columns
    const int rank = logits_in_.dims();
    const int lastdim = rank - 1;
    int rows = logits_in_.shape().dim_size(0);
    const int cols = logits_in_.shape().dim_size(lastdim);
    if (rank == 3)
        rows *= logits_in_.shape().dim_size(1);

    TensorShape out_shape(logits_in_.shape());
    out_shape.set_dim(lastdim, 1);
    Tensor* out = nullptr;

    Tensor max_logits, max_logits_safe;

    // Allocate output and temporary tensors
    OP_REQUIRES_OK(context,
      context->allocate_output(0, out_shape, &out));
    OP_REQUIRES_OK(context,
      context->allocate_temp(DataTypeToEnum<T>::value, out_shape, &max_logits));
    OP_REQUIRES_OK(context,
      context->allocate_temp(DataTypeToEnum<T>::value, out_shape,
        &max_logits_safe));

    const Device &device = context->eigen_device<Device>();
    if (logits_in_.NumElements() > 0) {
      functor::LogsumexpFunctor<Device, T> functor;

      // Feed Eigen matrices to functor
      auto out_matrix = out->flat_inner_dims<T>();
      auto max_logits_matrix = max_logits.flat_inner_dims<T>();
      auto max_logits_safe_matrix = max_logits_safe.flat_inner_dims<T>();
      functor(device, logits_in, max_logits_matrix, max_logits_safe_matrix,
        out_matrix);
    }
  }
};


REGISTER_OP("ReduceLogsumexp")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .SetShapeFn([](InferenceContext* ctx) {
        ShapeHandle in_shape;
        TF_RETURN_IF_ERROR(ctx->WithRankAtLeast(ctx->input(0), 2, &in_shape));
        TF_RETURN_IF_ERROR(ctx->WithRankAtMost(ctx->input(0), 3, &in_shape));

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
      Name("ReduceLogsumexp")                                                  \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T"),                                          \
      LogsumexpOp<CPUDevice, type>);
TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);
//REGISTER_CPU_KERNELS(float)
//REGISTER_CPU_KERNELS(double)
#undef REGISTER_CPU_KERNELS
//
// #if GOOGLE_CUDA
//
// #define REGISTER_GPU_KERNELS(type)                                             \
//   REGISTER_KERNEL_BUILDER(                                                     \
//       Name("ReduceLogsumexp")                                                  \
//           .Device(DEVICE_GPU)                                                  \
//           .TypeConstraint<type>("T"),                                          \
//       LogsumexpOp<GPUDevice, type>);
// //TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
// REGISTER_GPU_KERNELS(float)
// REGISTER_GPU_KERNELS(double)
// #undef REGISTER_GPU_KERNELS
//
// #endif

}

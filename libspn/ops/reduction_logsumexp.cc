#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/kernels/reduction_ops.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// cwise_ops_common for functor::sub
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/kernels/reduction_ops_common.h"
// #include "tensorflow/core/kernels/reduction_ops_mmon.h"

namespace tensorflow
{
using shape_inference::ShapeHandle;
using shape_inference::InferenceContext;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

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

    // Tensor sub_out;
    // OP_REQUIRES_OK(
    //     ctx, ctx->allocate_temp(ctx->expected_output_dtype(0),
    //                             maxhelper.out_reshape(), &sub_out, alloc_attr));


    auto max_flat = max_out.flat<T>();
    auto tmp_flat = tmp_out.flat<T>();

    typedef Eigen::internal::SumReducer<T> SumReducer;
    typedef Eigen::internal::MaxReducer<T> MaxReducer;
    typedef functor::ReduceFunctor<Device, SumReducer> SumFunctor;
    typedef functor::ReduceFunctor<Device, MaxReducer> MaxFunctor;
    //typedef functor::sub<T> SubtractFunctor;
    typedef functor::sub<T> SubtractFunctor;

    Constants<Device> constants;
    const Device& d = ctx->eigen_device<Device>();
    MaxReducer max_reducer;
    SumReducer sum_reducer;
    SubtractFunctor subtracter;

/*
    taken from tensorflow/core/util/bcast.h

        BCast takes the shape of two tensors and computes a few vectors of
        int32 that are useful for the caller to reshape the tensors, apply
        the right broadcasts to them, compute the broadcasted operation,
        and possibly the gradients. In a nutshell, the caller is expected
        to compute the broadcasted operation as following:
    //
    //   BCast b(x.shape(), y.shape());
    //   output = x.reshape(b.x_reshape()).broadcast(b.x_bcast())
    //            _op_
    //            y.reshape(b.y_reshape()).broadcast(b.y_bcast())
*/


    Tensor max;
    Tensor* subtracted;

    // taken from tensorflow/core/kernels/cwise_ops_common.h
    // TODO this line might be important
    bool error = false;
    bool* const error_ptr = SubtractFunctor::has_errors ? &error : nullptr;

    if (tmp_out.NumElements() == 0) {
      // Nothing to do, fall through to final reshaping.
    }
    // else if (data.NumElements() == 0) {
    //   // Degenerate reduction where the input is empty but the output is
    //   // nonempty (thus tmp_out.NumElements() > 0), and we must fill the output
    //   // with identity elements.  Example: tf.reduce_sum(tf.zeros((0, 3)), [0]).
    //   // Eigen sometimes crashes in this case, so we do it manually.
    //   MaxFunctor::FillIdentity(d, max_flat, max_reducer);
    //   SumFunctor::FillIdentity(d, tmp_flat, sum_reducer);
    // } else if ((helper.ndims() == 1) && helper.reduce_first_axis()) {
    //   // Reduce to a scalar.
    //   MaxFunctor::Reduce(ctx, helper.out<T, 0>(&max_out), helper.in<T, 1>(data),
    //                      constants.kZero, max_reducer);
    //   // Broadcasted shape should match input's shape
    //   if (!max.CopyFrom(max_out, helper.out_shape())) {
    //     ctx->SetStatus(errors::Internal("Error during reduction copy."));
    //   }
    //   subtracter(d, max_flat, )
    //   SumFunctor::Reduce(ctx, helper.out<T, 0>(&tmp_out), helper.in<T, 1>(data),
    //                      constants.kZero, sum_reducer);
    // }
    // else if ((helper.ndims() == 2) && helper.reduce_first_axis()) {
    //   // Can be viewed as a reduction of a matrix along 1st dimension.
    //   MaxFunctor::Reduce(ctx, helper.out<T, 1>(&max_out), helper.in<T, 2>(data),
    //                      constants.kZero, max_reducer);
    //   SumFunctor::Reduce(ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 2>(data),
    //                      constants.kZero, sum_reducer);
    //}
    else if ((helper.ndims() == 2) && !helper.reduce_first_axis()) {
      // Can be viewed as a reduction of a matrix along 2nd dimension.
      MaxFunctor::Reduce(d, helper.out<T, 1>(&max_out), helper.in<T, 2>(data),
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
      OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                                {0, 1}, 0, subtracted_shape, &subtracted));

      // Taken from
      functor::BinaryFunctor<Device, SubtractFunctor, 2>().BCast(
          eigen_device, subtracted->shaped<T, 2>(bcast.result_shape()),
          data.template shaped<T, 2>(bcast.x_reshape()),
          BCast::ToIndexArray<2>(bcast.x_bcast()),
          max.template shaped<T, 2>(bcast.y_reshape()),
          BCast::ToIndexArray<2>(bcast.y_bcast()), error_ptr);

      SumFunctor::Reduce(d, helper.out<T, 1>(&tmp_out), helper.in<T, 2>(*subtracted),
                         constants.kOne, sum_reducer);
    }
    // TODO implement something as beneath
    // else {
    //   SetUnimplementedError(ctx);
    // }
    // if (Functor::has_errors && error) {
    //   SetComputeError(ctx);
    // }

    
    // else if ((helper.ndims() == 3) && helper.reduce_first_axis()) {
    //   // Can be viewed as a reduction of a 3D tensor along 1st and 3rd
    //   // dimensions.
    //   MaxFunctor::Reduce(ctx, helper.out<T, 1>(&max_out), helper.in<T, 3>(data),
    //                      constants.kZeroTwo, max_reducer);
    //   SumFunctor::Reduce(ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 3>(data),
    //                      constants.kZeroTwo, sum_reducer);
    // } else if ((helper.ndims() == 3) && !helper.reduce_first_axis()) {
    //   // Can be viewed as a reduction of a 3D tensor along 2nd dimension.
    //   MaxFunctor::Reduce(ctx, helper.out<T, 2>(&max_out), helper.in<T, 3>(data),
    //                      constants.kOne, max_reducer);
    //   // Can be viewed as a reduction of a 3D tensor along 2nd dimension.
    //   SumFunctor::Reduce(ctx, helper.out<T, 2>(&tmp_out), helper.in<T, 3>(data),
    //                      constants.kOne, sum_reducer);
    // } else {
    //   // If we don't hit one of the cases above, transpose the data so that
    //   // all reduced dimensions are last and reuse the 2-D -> 1-D case.
    //
    //   // TODO(jostosh) not sure what to do here
    //   Tensor data_reshaped;
    //   CHECK(data_reshaped.CopyFrom(data, helper.data_reshape()));
    //   Tensor shuffled;
    //   OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
    //                                          helper.shuffled_shape(), &shuffled,
    //                                          alloc_attr));
    //   OP_REQUIRES_OK(
    //       ctx, DoTranspose(d, data_reshaped, helper.permutation(), &shuffled));
    //   const int64 unreduced = tmp_out.NumElements();
    //   const int64 reduced = shuffled.NumElements() / unreduced;
    //   const Tensor& const_shuffled = shuffled;
    //   Functor::Reduce(ctx, tmp_out.flat<T>(),
    //                   const_shuffled.shaped<T, 2>({unreduced, reduced}),
    //                   constants.kOne, reducer);
    // }

    // Set the real output using the contents of the reduction but the
    // real expected output shape.  The number of elements should
    // match between the two shapes.
    Tensor out;
    if (!out.CopyFrom(tmp_out, helper.out_shape())) {
      ctx->SetStatus(errors::Internal("Error during reduction copy."));
    }
    // TODO used to be in there won't compile
//    if (ctx->track_allocations()) {
//      ctx->record_temp_memory_size(-static_cast<int64>(out.AllocatedBytes()));
//    }
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

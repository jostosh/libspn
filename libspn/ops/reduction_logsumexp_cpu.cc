
#define EIGEN_USE_THREADS

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/numeric_op.h"
//#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/common_shape_fns.h"
//#include "tensorflow/core/kernels/reduction_ops.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/bcast.h"


// cwise_ops_common for functor::sub
//#include "tensorflow/core/kernels/reduction_ops_common.h"
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



REGISTER_OP("LogSumExpLibSpn")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);


template <typename Device, typename OUT_T, typename IN_T,
          typename ReductionAxes, typename Reducer>
void ReduceEigenImpl(const Device& d, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer) {
  out.device(d) = in.reduce(reduction_axes, reducer);
}


template <typename Reducer>
struct Identity {
  static auto identity(const Reducer& reducer)
      -> decltype(reducer.initialize()) {
    return reducer.initialize();
  }
};

template <typename Device, typename OUT_T, typename Reducer>
void FillIdentityEigenImpl(const Device& d, OUT_T out, const Reducer& reducer) {
  out.device(d) = out.constant(Identity<Reducer>::identity(reducer));
}

template <typename Device, typename Reducer>
struct ReduceFunctorBase {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer) {
    const Device& d = ctx->eigen_device<Device>();
    ReduceEigenImpl(d, out, in, reduction_axes, reducer);
  }

  template <typename OUT_T>
  static void FillIdentity(const Device& d, OUT_T out,
                           const Reducer& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <typename Reducer>
struct ReduceFunctor<CPUDevice, Reducer>
        : ReduceFunctorBase<CPUDevice, Reducer>{};


class ReductionHelper {
 public:
  ReductionHelper() : reduce_first_axis_(false) {}

  Status Simplify(const Tensor& data, const Tensor& axis, const bool keep_dims);

  // We need to do roughly:
  //   tmp_out = allocate(out_reshape())
  //   tmp_out.reshape(out_reshape) = data.reshape(data_reshape).reduce(axes)
  //   out = tmp_out.reshape(out_shape)

  // The reduction result must be allocated with this shape.
  TensorShape out_reshape() const;

  // The final output shape must be allocated with this shape.
  TensorShape out_shape() const;

  // The reduction is on a reshaped tensor of this rank.
  int ndims() const { return data_reshape_.size(); }

  // True if need to reduce the 0-th dimension.
  bool reduce_first_axis() const { return reduce_first_axis_; }

  // The output is reshaped.
  template <typename T, int N>
  typename TTypes<T, N>::Tensor out(Tensor* out) {
    return out->shaped<T, N>(out_reshape_);
  }

  // The input is reshaped.
  template <typename T, int N>
  typename TTypes<T, N>::ConstTensor in(const Tensor& data) {
    return data.shaped<T, N>(data_reshape_);
  }

  // Shape of shuffled input
  TensorShape data_reshape() const {
    TensorShape shape;
    for (auto s : data_reshape_) shape.AddDim(s);
    return shape;
  }

  // Shape with all reduction dimensions at the end
  TensorShape shuffled_shape();

  // Permutation of reduced dims needed to put reduction dimensions at the end
  gtl::InlinedVector<int32, 8> permutation();

 private:
  bool reduce_first_axis_;  // True if need to reduce the 0-th dimension.
  gtl::InlinedVector<int64, 4> data_reshape_;  // Reshape data before reduction.
  gtl::InlinedVector<int64, 4> out_shape_;     // The final output shape.
  gtl::InlinedVector<int64, 4> out_reshape_;   // Reshape output for reduction.
};

TensorShape ReductionHelper::out_reshape() const {
  TensorShape shape;
  for (auto size : out_reshape_) shape.AddDim(size);
  return shape;
}

// The final output shape must be allocated with this shape.
TensorShape ReductionHelper::out_shape() const {
  TensorShape shape;
  for (auto size : out_shape_) shape.AddDim(size);
  return shape;
}

TensorShape ReductionHelper::shuffled_shape() {
  const int dims = data_reshape_.size();
  TensorShape shape;
  for (int i = reduce_first_axis_; i < dims; i += 2) {
    shape.AddDim(data_reshape_[i]);
  }
  for (int i = !reduce_first_axis_; i < dims; i += 2) {
    shape.AddDim(data_reshape_[i]);
  }
  return shape;
}

gtl::InlinedVector<int32, 8> ReductionHelper::permutation() {
  const int dims = data_reshape_.size();
  const int unreduced_dims = (dims + !reduce_first_axis_) / 2;
  gtl::InlinedVector<int32, 8> perm(dims);
  for (int i = 0; i < unreduced_dims; i++) {
    perm[i] = 2 * i + reduce_first_axis_;
  }
  for (int i = unreduced_dims; i < dims; i++) {
    perm[i] = 2 * (i - unreduced_dims) + !reduce_first_axis_;
  }
  return perm;
}

Status ReductionHelper::Simplify(const Tensor& data, const Tensor& axis,
                                 const bool keep_dims) {
  // bitmap[i] indicates whether to reduce data along i-th axis.
  gtl::InlinedVector<bool, 4> bitmap(data.dims(), false);
  auto axis_vec = axis.flat<int32>();
  for (int64 i = 0; i < axis.NumElements(); ++i) {
    int32 index = axis_vec(i);
    if (index < -data.dims() || index >= data.dims()) {
      return errors::InvalidArgument("Invalid reduction dimension (", index,
                                     " for input with ", data.dims(),
                                     " dimension(s)");
    }
    index = (index + data.dims()) % data.dims();
    bitmap[index] = true;
  }

  // Output tensor's dim sizes.
  out_shape_.clear();
  for (int i = 0; i < data.dims(); ++i) {
    if (!bitmap[i]) {
      // If we are not reducing along dimension i.
      out_shape_.push_back(data.dim_size(i));
    } else if (keep_dims) {
      // We are reducing along dimension i, but we want to keep the
      // same number of dimensions, so we set the dimension of i to
      // '1'.
      out_shape_.push_back(1);
    }
  }

  // Depending on bitmap[i] and bitmap[i-1], we can collapse axis of
  // the input data before doing the reduction on the resulting
  // tensor.  The shape of the reduction is a reshape of the final
  // output.

  // We'll skip the leading 1s.
  int dim_index = 0;
  for (; dim_index < data.dims(); ++dim_index) {
    if (data.dim_size(dim_index) != 1) break;
  }
  if (dim_index >= data.dims()) {
    // Special case. The input is essentially a scalar.
    reduce_first_axis_ = true;
  } else {
    // Starting from the (dim_index)-th dimension, dimensions
    // alternates between runs that need to be reduced and runs that
    // don't.
    //
    // NOTE: If a dimension has size 1, we group it as the current
    // run so that we can minimize the number of runs.
    //
    // E.g., when we want to reduce a tensor of shape [2, 1, 3, 1,
    // 5] by axes = [1, 4], we should treat the tensor as a [6, 5]
    // and reduce by axes = [1] (i.e., the output is shape [6]).
    reduce_first_axis_ = bitmap[dim_index];
    data_reshape_.push_back(data.dim_size(dim_index));
    ++dim_index;
    for (; dim_index < data.dims(); ++dim_index) {
      const auto size = data.dim_size(dim_index);
      if (size == 1) {
        bitmap[dim_index] = bitmap[dim_index - 1];
      }
      if (bitmap[dim_index - 1] != bitmap[dim_index]) {
        // Starts a new run of reduce or !reduce.
        data_reshape_.push_back(size);
      } else {
        // Continue a run of reduce or !reduce.
        data_reshape_.back() *= size;
      }
    }
    // If reduce_first_axis_ is true (input's dimension 0, 2, 4, etc
    // are reduced), data_reshape_[1, 3, 5, ...]  is out_reshape_,
    // otherwise, data_reshape_[0, 2, 4, ...] is.
    for (size_t i = reduce_first_axis_ ? 1 : 0; i < data_reshape_.size();
         i += 2) {
      out_reshape_.push_back(data_reshape_[i]);
    }
  }

  return Status::OK();
}


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
    OP_REQUIRES_OK(ctx, helper.Simplify(data, axes, keep_dims_));
    CHECK_GE(helper.ndims(), 0);
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

    CHECK_EQ(helper.reduce_first_axis(), 0);

    typedef Eigen::internal::SumReducer<T> SumReducer;
    typedef Eigen::internal::MaxReducer<T> MaxReducer;
    typedef ReduceFunctor<SumReducer> SumFunctor;
    typedef ReduceFunctor<MaxReducer> MaxFunctor;
    Constants<Device> constants;
    const Device& d = ctx->eigen_device<Device>();
    MaxReducer max_reducer;
    SumReducer sum_reducer;

    Tensor max;
    const Eigen::IndexList<Eigen::type2index<1>> kOne;
    SumFunctor::Reduce(ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 2>(data),
                       kOne, sum_reducer);

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

REGISTER_KERNEL_BUILDER(                                                     \
      Name("LogSumExpLibSpn")                                                        \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<float>("T")                                           \
          .TypeConstraint<int64>("Tidx"),                                      \
      ReductionLogSumExpOp<CPUDevice, float, int64>);


}  // namespace tensorflow

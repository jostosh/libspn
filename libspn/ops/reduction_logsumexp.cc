
#define EIGEN_USE_THREADS

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/bcast.h"


namespace tensorflow
{
using shape_inference::ShapeHandle;
using shape_inference::InferenceContext;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL


template <typename Device>
struct Constants {
  // Derive Index type. int (32-bit) or long (64-bit) depending on the
  // compile-time configuration. "float" here is not relevant.
  // TODO(zhifengc): Moves the definition to TTypes.
  typedef TTypes<float>::Tensor::Index Index;
  Eigen::array<Index, 1> kZero;
  Eigen::array<Index, 1> kOne;
  Eigen::array<Index, 2> kZeroTwo;

  Constants() {
    kZero[0] = 0;
    kOne[0] = 1;
    kZeroTwo[0] = 0;
    kZeroTwo[1] = 2;
  }
};

#if defined(EIGEN_HAS_INDEX_LIST)
struct ConstantsBase {
  const Eigen::IndexList<Eigen::type2index<0>> kZero;
  const Eigen::IndexList<Eigen::type2index<1>> kOne;
  const Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<2>> kZeroTwo;
};
template<> struct Constants<CPUDevice> : ConstantsBase{};
#ifdef TENSORFLOW_USE_SYCL
template<> struct Constants<SYCLDevice> : ConstantsBase{};
#endif // TENSORFLOW_USE_SYCL
#endif // EIGEN_HAS_INDEX_LIST


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
    const Tensor& axis = ctx->input(1);

    TensorShape max_shape(data.shape());
    TensorShape out_shape(data.shape());
    const DataType dt = DataTypeToEnum<T>::v();

    typedef Eigen::internal::MaxReducer<T> MaxReducer;

    auto axis_vec = axis.flat<int32>();

    const Device& d = ctx->eigen_device<Device>();

    MaxReducer reducer;

    out_shape.set_dim(axis_vec(0), 1);
    max_shape.set_dim(axis_vec(0), 1);
    Constants<Device> constants;

    Tensor max(dt, out_shape);

    //TODO This is taken from ReduceEigenImpl, could be different for GPU
    max.tensor<T, 2>().device(d) = data.tensor<T, 2>().reduce(constants.kOne, reducer);

    BCast bcast(data.shape().dim_sizes(), max_shape.dim_sizes());

    // TODO below doesn't work
//    auto data_bcast = data.reshape(bcast.x_reshape()).broadcast(bcast.x_bcast());
//    auto  max_bcast =  max.reshape(bcast.y_reshape()).broadcast(bcast.y_bcast());

    ctx->set_output(0, max);
  }

 private:
  // True if the number of dimensions should be maintained.
  bool keep_dims_;
};

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


//#if GOOGLE_CUDA
//#define EIGEN_USE_GPU
//
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
//
//REGISTER_GPU_KERNELS(float)
//REGISTER_GPU_KERNELS(double)
////TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
////TF_CALL_complex64(REGISTER_GPU_KERNELS);
////TF_CALL_complex128(REGISTER_GPU_KERNELS);
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

//#endif

}
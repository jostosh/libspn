
#if GOOGLE_CUDA

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "reduction_logsumexp_op_gpu.cu.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;
using shape_inference::Dimension;

#define EIGEN_USE_GPU

#define REGISTER_GPU_KERNELS(type)                                             \
 REGISTER_KERNEL_BUILDER(                                                      \
     Name("ReduceLogsumexp")                                                   \
         .Device(DEVICE_GPU)                                                   \
         .TypeConstraint<type>("T"),                                           \
     LogSumExpOpGPU<type>);

TF_CALL_float(REGISTER_GPU_KERNELS);
TF_CALL_double(REGISTER_GPU_KERNELS);
TF_CALL_half(REGISTER_GPU_KERNELS);

#undef REGISTER_GPU_KERNELS

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA

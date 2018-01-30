/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/core/util/cuda_kernel_helper.h"

#include "tensorflow/core/kernels/reduction_gpu_kernels.cu.h"
#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

namespace {
//
template <typename T>
__global__ void ReplaceInfOrNanWithZero(T* data,
    const int num_rows, const int num_cols) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  data[tid] = (isfinite(data[tid])) ? data[tid] : 0.0;
}

template <typename T>
__global__ void SubtractAndExpKernel(
    CudaLaunchConfig &cfg,
    T* left,
    const T* right,
    const int num_cols) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  left[tid] = exp(left[tid] - ldg(right + tid / num_cols));
}


template <typename T>
struct SubtractAndExpFunctor {
  __host__ __device__ SubtractAndExpFunctor(const T* logits,
                                            const T* max_logits,
                                            const int num_cols)
      : logits_(logits), max_logits_(max_logits), num_cols_(num_cols) {}
  __host__ __device__ T operator()(const int gid) const {
    // Assuming input is 2D [(num_rows_) x num_cols_], we can find the
    // index of the corresponding max logit by dividing the offset given
    // by gid by num_cols_
    return exp(logits_[gid] - ldg(max_logits_ + gid / num_cols_));
  }

  const T* logits_;
  const T* max_logits_;
  const int num_cols_;
};


template <typename T>
__global__ void LogAndAddKernel(const T* left, const T* right, T* out,
        const int numel) {
    CUDA_1D_KERNEL_LOOP(x, numel)
    {
      //out[x] = log(ldg(left + x)) + ldg(right + x);
    }
}


template <typename T, typename Op, typename InputIter>
void DoRowReduction(OpKernelContext* context, T* output, InputIter input,
                    int rows, int cols) {
  typedef const Eigen::array<TTypes<float>::Tensor::Index, 1>& ReductionAxes;
  Constants<GPUDevice> constants;

  Op op;

  functor::ReduceImpl<T, Op, T*, InputIter, ReductionAxes>(
      context, output, input, 2, rows, cols, 1, 1, constants.kOne, op);
}

}  // namespace

template <typename T>
class LogSumExpOpGPU : public OpKernel {
 public:
  explicit LogSumExpOpGPU(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in_ = context->input(0);
    auto logits_in = logits_in_.matrix<T>();
    const int rows = logits_in_.shape().dim_size(0);
    const int cols = logits_in_.shape().dim_size(1);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits_in_.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));
    // TODO need to set output shape differently
    TensorShape out_shape(logits_in_.shape());
    out_shape.set_dim(1, 1);

    const cudaStream_t& cu_stream = GetCudaStream(context);
    Tensor *out;
    context->allocate_output(0, out_shape, &out);
    if (logits_in_.NumElements() > 0) {

      // Three tensors
      Tensor max_logits;
      Tensor sum_probs;

      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::value,
                                            out_shape, &max_logits));
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::value,
                                            out_shape, &sum_probs));

      DoRowReduction<T, cub::Max, const T*>(
          context, const_cast<T*>(max_logits.flat<T>().data()),
          reinterpret_cast<const T*>(logits_in_.flat<T>().data()), rows, cols);
      //
      // // ReplaceInfOrNanWithZero<<<numBlocks, numThreads, 0, cu_stream>>>(
      // //     const_cast<T*>(max_logits.flat<T>().data()), rows, cols);

      const GPUDevice &d = context->eigen_device<GPUDevice>();
      CudaLaunchConfig config = GetCudaLaunchConfig(logits_in_.NumElements(), d);

      cub::CountingInputIterator<int> counting_iterator(0);
            typedef cub::TransformInputIterator<T, SubtractAndExpFunctor<T>,
                                                cub::CountingInputIterator<int>>
                InputIterType;

            InputIterType input_itr(
                counting_iterator,
                SubtractAndExpFunctor<T>(
                    reinterpret_cast<const T*>(logits_in_.flat<T>().data()),
                    reinterpret_cast<const T*>(max_logits.flat<T>().data()), cols));

      DoRowReduction<T, cub::Sum, InputIterType>(
        context, const_cast<T*>(sum_probs.flat<T>().data()), input_itr, rows,
        cols);

      // So problem lies here ...
      config = GetCudaLaunchConfig(out->NumElements(), d);
      LogAndAddKernel<<<config.block_count, config.thread_per_block, 0, cu_stream>>>(
            reinterpret_cast<const T*>(sum_probs.flat<T>().data()),
            reinterpret_cast<const T*>(max_logits.flat<T>().data()),
            const_cast<T*>(out->flat<T>().data()),
            rows);
    }
  }

};

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA

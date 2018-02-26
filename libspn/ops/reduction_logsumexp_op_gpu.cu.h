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

// TODO one of these includes tends to produce a lot of Eigen warnings
#include "tensorflow/core/kernels/reduction_gpu_kernels.cu.h"
#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

namespace {
template <typename T>
__global__ void ReplaceInfWithZero(T* data, CudaLaunchConfig clc) {
  // Replaces any -inf or +inf with zero
  CUDA_1D_KERNEL_LOOP(x, clc.virtual_thread_count)
  {
    if (isinf(data[x]))
      data[x] = static_cast<T>(0);
  }
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
__global__ void LogAddAssignKernel(CudaLaunchConfig clc,
  const T* max_logits, T* out, const int numel) {
    // Takes logarithm of out, adds the max logit to it and writes the result
    // to out
    CUDA_1D_KERNEL_LOOP(x, clc.virtual_thread_count)
    {
      out[x] = log(out[x]) + ldg(max_logits + x);
    }
}


template <typename T, typename OpFunctor, typename InputIter>
void DoRowReduction(OpKernelContext* context, T* output, InputIter input,
                    int rows, int cols) {
  // Performs reduction over rows using some functor Op (e.g. max or sum)
  typedef const Eigen::array<TTypes<float>::Tensor::Index, 1>& ReductionAxes;
  Constants<GPUDevice> constants;

  // Construct the operator functor
  OpFunctor opFunctor;
  functor::ReduceImpl<T, OpFunctor, T*, InputIter, ReductionAxes>(
      context, output, input, 2, rows, cols, 1, 1, constants.kOne, opFunctor);
}

}  // namespace

template <typename T>
class LogSumExpOpGPU : public OpKernel {
 public:
  explicit LogSumExpOpGPU(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Computes the logsumexp operation, referring to f(x) = log(sum(exp(x)))
    // The incoming tensor is reduced over the last axis
    //
    // In practice, we compute f(x) = log(sum(exp(x - mx))) + mx, where mx
    // is max(x), to improve numerical stability
    const Tensor& logits_in_ = context->input(0);
    auto logits_in = logits_in_.flat_inner_dims<T>();

    // Making sure we have the right number of rows and columns
    const int rank = logits_in_.dims();
    const int lastdim = rank - 1;
    const int cols = logits_in_.shape().dim_size(lastdim);
    const int rows = logits_in_.NumElements() / cols;

    // Setting output shape and allocating it
    TensorShape out_shape(logits_in_.shape());
    out_shape.set_dim(lastdim, 1);
    Tensor *out;
    context->allocate_output(0, out_shape, &out);

    const cudaStream_t& cu_stream = GetCudaStream(context);
    const GPUDevice &d = context->eigen_device<GPUDevice>();
    if (logits_in_.NumElements() > 0) {

      // Temporary max logit tensor
      Tensor max_logits;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::value,
                                            out_shape, &max_logits));

      // Perform max reduction over rows, storing in max_logits
      DoRowReduction<T, cub::Max, const T*>(
          context, const_cast<T*>(max_logits.flat<T>().data()),
          reinterpret_cast<const T*>(logits_in_.flat<T>().data()), rows, cols);

      // Making sure we don't subtract infinite numbers
      CudaLaunchConfig config = GetCudaLaunchConfig(max_logits.NumElements(), d);
      ReplaceInfWithZero
        <<<config.block_count, config.thread_per_block, 0, cu_stream>>>(
          reinterpret_cast<T*>(max_logits.flat<T>().data()), config);

      // Setting up an iterator that will subtract the max and exponentialize
      // the result. This acts as a kind of 'placeholder' for the next
      // reduction operation, where the value is 'fetched'
      config = GetCudaLaunchConfig(logits_in_.NumElements(), d);
      cub::CountingInputIterator<int> counting_iterator(0);
      typedef cub::TransformInputIterator<T, SubtractAndExpFunctor<T>,
          cub::CountingInputIterator<int>> InputIterType;
      InputIterType input_itr(
          counting_iterator,
          SubtractAndExpFunctor<T>(
              reinterpret_cast<const T*>(logits_in_.flat<T>().data()),
              reinterpret_cast<const T*>(max_logits.flat<T>().data()),
              cols));

      // Now take the sum
      DoRowReduction<T, cub::Sum, InputIterType>(
        context, const_cast<T*>(out->flat<T>().data()), input_itr, rows,
        cols);

      // Obtain the output by computing y(a) = log(a) + mx
      config = GetCudaLaunchConfig(out->NumElements(), d);
      LogAddAssignKernel
        <<<config.block_count, config.thread_per_block, 0, cu_stream>>>(
            config,
            reinterpret_cast<const T*>(max_logits.flat<T>().data()),
            const_cast<T*>(out->flat<T>().data()),
            rows);
    }
  }

};

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA

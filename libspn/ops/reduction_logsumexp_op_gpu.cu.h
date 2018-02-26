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
    return exp(ldg(logits_ + gid) - ldg(max_logits_ + gid / num_cols_));
  }

  const T* logits_;
  const T* max_logits_;
  const int num_cols_;
};


template <typename T>
__global__ void SubtractAndExpKernel(const T* logits,
                                     T* max_logits,
                                     const int num_cols,
                                     T* out,
                                     const int total_size)
{
  // Uses shared mem, but turns out not to yield any significant speed-up...
  // extern __shared__ __align__(sizeof(T)) unsigned char max_logits_shared[];
  // T *smem = reinterpret_cast<T *>(max_logits_shared);
  // Determine the 'row' in shared memory
  //int tid, block_start, block_offset, row, max_idx;
  int max_idx;
  CUDA_1D_KERNEL_LOOP(x, total_size)
  {
    // Determine the 'row' in shared memory
    // tid = threadIdx.x;
    // block_start = x - tid;
    // block_offset = block_start % num_cols;
    // row = (block_offset + tid) / num_cols;
    //
    // if (tid == 0 || (block_offset + tid) % num_cols == 0)
    // {
    //   // Read the max value, and set it to 0 if infinite
    //   T val = max_logits[x / num_cols];
    //   val = isinf(val) ? static_cast<T>(0) : val;
    //   max_logits[x / num_cols] = val;
    //   smem[row] = val;
    // }

    max_idx = x / num_cols;
    if (x % num_cols == 0)
    {
      T m = max_logits[max_idx];
      m = isinf(m) ? static_cast<T>(0) : m;
      max_logits[max_idx] = m;
    }
    __syncthreads();

    // Subtract and exponentialize
    out[x] = exp(ldg(logits + x) - max_logits[max_idx]);
  }
}

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

      Tensor exp_logits_sub_max;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::value,
                                            logits_in_.shape(),
                                            &exp_logits_sub_max));


      // Perform max reduction over rows, storing in max_logits
      DoRowReduction<T, cub::Max, const T*>(
          context, const_cast<T*>(max_logits.flat<T>().data()),
          reinterpret_cast<const T*>(logits_in_.flat<T>().data()), rows, cols);

      // Making sure we don't subtract infinite numbers
      // CudaLaunchConfig config = GetCudaLaunchConfig(max_logits.NumElements(), d);
      // ReplaceInfWithZero
      //   <<<config.block_count, config.thread_per_block, 0, cu_stream>>>(
      //     reinterpret_cast<T*>(max_logits.flat<T>().data()), config);

      // Subtracts max and exponentialize. Also replaces any infs in max with 0
      CudaLaunchConfig config = GetCudaLaunchConfig(logits_in_.NumElements(), d);
      const int smem_len = config.thread_per_block / cols + 1;
      SubtractAndExpKernel
        <<<config.block_count, config.thread_per_block, 0,//smem_len * sizeof(T),
           cu_stream>>>(
          reinterpret_cast<const T*>(logits_in_.flat<T>().data()),
          const_cast<T*>(max_logits.flat<T>().data()),
          cols,
          const_cast<T*>(exp_logits_sub_max.flat<T>().data()),
          rows * cols);


      // Setting up an iterator that will subtract the max and exponentialize
      // the result. This acts as a kind of 'placeholder' for the next
      // reduction operation, where the value is 'fetched'
      // config = GetCudaLaunchConfig(logits_in_.NumElements(), d);
      // cub::CountingInputIterator<int> counting_iterator(0);
      // typedef cub::TransformInputIterator<T, SubtractAndExpFunctor<T>,
      //     cub::CountingInputIterator<int>> InputIterType;
      // InputIterType input_itr(
      //     counting_iterator,
      //     SubtractAndExpFunctor<T>(
      //         reinterpret_cast<const T*>(logits_in_.flat<T>().data()),
      //         reinterpret_cast<const T*>(max_logits.flat<T>().data()),
      //         cols));

      // Now take the sum
      DoRowReduction<T, cub::Sum, const T*>(
        context, const_cast<T*>(out->flat<T>().data()),
        reinterpret_cast<const T*>(exp_logits_sub_max.flat<T>().data()), rows,
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

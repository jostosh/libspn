#ifndef TENSORFLOW_USEROPS_REDUCTION_LOGSUMEXP_FUNCTOR_H_
#define TENSORFLOW_USEROPS_REDUCTION_LOGSUMEXP_FUNCTOR_H_

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow
{
  typedef Eigen::ThreadPoolDevice CPUDevice;
  typedef Eigen::GpuDevice GPUDevice;

  namespace functor
  {
    template <typename Device, typename T>
    struct LogsumexpFunctor {
      // Computes the logsumexp along the columns
      //
      // logits: matrix log probabilities [rows x cols]
      // out: output matrix [rows x 1]
      void operator()(const Device &d,
        typename TTypes<T>::ConstMatrix &logits,
        typename TTypes<T>::Matrix &out);
    };

    // This should work for both GPU and CPU devices
    // Heavily inspired by the implementation of the softmax functors as in
    // tensorflow/core/kernels/softmax_op_functor.h
    template <typename Device, typename T>
    struct LogsumexpEigenImpl {
      static void Compute(const Device& d,
                     typename TTypes<T>::ConstMatrix &logits,
                     typename TTypes<T>::Matrix &shifted_logits,
                     typename TTypes<T>::Matrix &max_logits,
                     typename TTypes<T>::Matrix &out)
      {
        const int rowDim = 0;
        const int colDim = 1;

        const int rows = logits.dimension(rowDim);
        const int cols = logits.dimension(colDim);

        #if !defined(EIGEN_HAS_INDEX_LIST)
            Eigen::DSizes<int, 1> along_cols(colDim);
            Eigen::DSizes<int, 2> rows_x_one(rows, 1);
            Eigen::DSizes<int, 2> one_x_cols(1, cols);
        #else
            Eigen::IndexList<Eigen::type2index<colDim> > along_cols;
            Eigen::IndexList<int, Eigen::type2index<1> > rows_x_one;
            rows_x_one.set(0, rows);
            Eigen::IndexList<Eigen::type2index<1>, int> one_x_cols;
            one_x_cols.set(1, cols);
        #endif
        max_logits.device(d) = logits.maximum(along_cols)
                                     .eval()
                                     .reshape(rows_x_one);
        out.device(d) = (logits - max_logits.broadcast(one_x_cols))
            .exp()
            .sum(along_cols)
            .log()
            .reshape(rows_x_one) + max_logits;
      }
    };

  }
}

#endif

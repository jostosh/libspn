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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#define REGISTER_GPU_KERNELS(type)                                             \
 REGISTER_KERNEL_BUILDER(                                                      \
     Name("ReduceLogsumexp")                                                   \
         .Device(DEVICE_GPU)                                                   \
         .TypeConstraint<type>("T"),                                           \
     LogSumExpOpGPU<type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
//TF_CALL_complex64(REGISTER_GPU_KERNELS);
//TF_CALL_complex128(REGISTER_GPU_KERNELS);
// REGISTER_GPU_KERNELS(float);
// REGISTER_GPU_KERNELS(double);
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA
}  // end namespace tensorflow

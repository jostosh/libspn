#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
from context import libspn as spn
import time
import argparse
import colorama as col
import sys
import scipy.misc
import itertools
from libspn.tests.profiler import profile_report
col.init()

op_module = tf.load_op_library('/home/jos/spn/libspn/libspn/ops/libspn_ops.so')


def print1(str, file):
    if file:
        print(str, file=file)
    print(col.Fore.YELLOW + str + col.Style.RESET_ALL)


def print2(str, file):
    if file:
        print(str, file=file)
    print(col.Fore.BLUE + str + col.Style.RESET_ALL)


def print3(str, file):
    if file:
        print(str, file=file)
    print(col.Fore.RED + str + col.Style.RESET_ALL)


class Ops:

    def noop(params):
        return params

    def logsum_old(params):
        return spn.utils.math.reduce_log_sum_v2(params)

    def logsum_tf(params):
        return tf.reduce_logsumexp(params, axis=-1, keepdims=True)

    def logsum_custom(params):
        return op_module.reduce_logsumexp(params)


class OpTestResult:
    """Result of a single test of a single op."""

    def __init__(self, op_name, on_gpu, graph_size, setup_time,
                 run_times, output_correct, dtype):
        self.op_name = op_name
        self.on_gpu = on_gpu
        self.graph_size = graph_size
        self.setup_time = setup_time
        self.run_times = run_times
        self.output_correct = output_correct
        self.dtype = dtype


class TestResults:
    """Results for a single test for multiple ops and devices."""

    def __init__(self, test_name, cpu_results, gpu_results):
        self.test_name = test_name
        self.cpu_results = cpu_results
        self.gpu_results = gpu_results

    def print(self, file):
        def get_header(dev):
            return ("%3s %15s %5s %8s %11s %15s %14s %10s" %
                    (dev, 'op', 'size', 'dtype', 'setup_time',
                     'first_run_time', 'rest_run_time', 'correct'))

        def get_res(res):
            """Helper function printing a single result."""
            return ("%18s: %5d %8s %11.2f %15.2f %14.2f %10s" %
                    (res.op_name, res.graph_size, res.dtype.name,
                     res.setup_time * 1000, res.run_times[0] * 1000,
                     np.mean(res.run_times[1:]) * 1000,
                     res.output_correct))

        # Print results
        print1("\n-----------------------", file)
        print1("%s" % self.test_name, file)
        print1("-----------------------", file)
        print1(get_header("CPU"), file)
        for res in sorted(self.cpu_results, key=lambda x: (x.dtype.name, x.op_name)):
            print1(get_res(res), file)
        print1(get_header("GPU"), file)
        for res in sorted(self.gpu_results, key=lambda x: (x.dtype.name, x.op_name)):
            print1(get_res(res), file)


class PerformanceTest:

    def __init__(self, num_param_rows, num_param_cols, num_param_slices, out_size,
                 num_ops, num_runs, dtype,
                 without_cpu, without_gpu, log_devs, file, profile, profiles_dir):
        self.num_param_rows = num_param_rows
        self.num_param_cols = num_param_cols
        self.num_param_slices = num_param_slices
        self.out_size = out_size
        self.num_ops = num_ops
        self.num_runs = num_runs
        self.dtype = dtype
        self.without_cpu = without_cpu
        self.without_gpu = without_gpu
        self.log_devs = log_devs
        self.file = file

        self.profile = profile
        self.profiles_dir = profiles_dir

        print1("Params:", file)
        print1("- num_param_rows=%s" % num_param_rows, file)
        print1("- num_param_cols=%s" % num_param_cols, file)
        print1("- out_size=%s" % out_size, file)
        print1("- num_ops=%s" % num_ops, file)
        print1("- num_runs=%s" % num_runs, file)
        print1("- dtype=%s" % dtype, file)
        print1("", file=file)

    def _run_op_test(self, op_fun, params, on_gpu, dtype):
        """Run a single test for a single op."""
        # Preparations
        op_name = op_fun.__name__
        device_name = '/gpu:0' if on_gpu else '/cpu:0'
        params = np.asarray(params, dtype=dtype.as_numpy_dtype())
        # Print
        print2("--> %s: on_gpu=%s, params_shape=%s, dtype=%s"
               % (op_name, on_gpu, params.shape, dtype),
               self.file)
        # Compute true output with numpy
        true_out = scipy.misc.logsumexp(params, axis=-1, keepdims=True)
        # Create graph
        tf.reset_default_graph()
        with tf.device(device_name):
            # Create input
            # We cannot use a constant here, since the operation will be pre-computed
            # on a CPU for some cases (e.g. for int64 indices)
            # To ensure that data is copied only once, we add an identity op
            # which is served the input data and connected to all ops
            params_pl = tf.placeholder(dtype=self.dtype)
            params_op = tf.identity(params_pl)
            # Create ops
            start_time = time.time()
            ops = op_fun(params_op)
            for _ in range(self.num_ops - 1):
                # The tuple ensures that the next op waits for the output
                # of the previous op, effectively stacking the ops
                # but using the original input every time
                ops = op_fun(tf.tuple([params_op, ops])[0])
            setup_time = time.time() - start_time
        # Get num of graph ops
        graph_size = len(tf.get_default_graph().get_operations())
        # Run multiple times
        output_correct = True
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=False,
                log_device_placement=self.log_devs)) as sess:
            run_times = []
            for n in range(self.num_runs):
                # Run
                start_time = time.time()
                out = sess.run(ops, feed_dict={params_pl: params})
                run_times.append(time.time() - start_time)
                # Test value
                try:
                    np.testing.assert_allclose(out, true_out, rtol=1e-5)
                    # np.testing.assert_array_almost_equal(out, true_out)
                except AssertionError:
                    output_correct = False

            if self.profile:
                # Create a suitable filename suffix
                fnm_suffix = op_name
                fnm_suffix += ("_GPU" if on_gpu else "_CPU")
                # Create a profiling report
                profile_report(sess, ops, {params_pl: params}, self.profiles_dir,
                               "sum_value_varying_sizes", fnm_suffix)
        # Return stats
        return OpTestResult(op_name, on_gpu, graph_size, setup_time,
                            run_times, output_correct, dtype)

    def _run_test(self, test_name, op_funs, params):
        """Run a single test for multiple ops and devices."""
        cpu_results = []
        gpu_results = []
        for dtype, op_fun in itertools.product([tf.float64, tf.float32], op_funs):
            if not self.without_cpu:
                cpu_results.append(
                    self._run_op_test(op_fun, params, on_gpu=False, dtype=dtype))
            if not self.without_gpu:
                gpu_results.append(
                    self._run_op_test(op_fun, params, on_gpu=True, dtype=dtype))
        return TestResults(test_name, cpu_results, gpu_results)

    def _run_2d(self):
        """Run all 2D tests."""
        results = []

        # # 1 index
        # params = np.random.rand(self.num_param_rows, 1)
        # r = self._run_test('2d_1index',
        #                    [Ops.reduce_logsum, Ops.reduce_logsum_tf, Ops.reduce_logsum_pyfun],
        #                    params)
        # results.append(r)

        # Passthrough
        params = np.random.rand(self.num_param_rows, self.num_param_cols)
        r = self._run_test('2d_passthrough',
                           [Ops.logsum_custom, Ops.logsum_old, Ops.logsum_tf],
                           params)
        results.append(r)

        return results

    def _run_3d(self):
        """Run all 3D tests."""
        results = []

        # Passthrough
        params = np.random.rand(
            self.num_param_slices, self.num_param_rows, self.num_param_cols
        )
        r = self._run_test('3d_passthrough',
                           [Ops.logsum_custom, Ops.logsum_old, Ops.logsum_tf],
                           params)
        results.append(r)

        return results

    def run(self):
        """Run all tests."""
        print1("Running tests:", self.file)
        results = []
        results += self._run_2d()
        results += self._run_3d()

        # Print results
        for res in results:
            res.print(self.file)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-param-rows', default=10000, type=int,
                        help="Num of rows of params")
    parser.add_argument('--num-param-cols', default=1000, type=int,
                        help="Num of cols of params")
    parser.add_argument('--num-param-slices', default=10, type=int,
                        help='Num of planes of params')
    parser.add_argument('--out-size', default=100, type=int,
                        help="Size of the output")
    parser.add_argument('--num-ops', default=200, type=int,
                        help="Num of ops used for tests")
    parser.add_argument('--num-runs', default=250, type=int,
                        help="Number of times each test is run")
    parser.add_argument('--log-devices', action='store_true',
                        help="Log on which device op is run. Affects run time!")
    parser.add_argument('--without-cpu', action='store_true',
                        help="Do not run CPU tests")
    parser.add_argument('--without-gpu', action='store_true',
                        help="Do not run GPU tests")
    parser.add_argument('--save-to', default='', type=str,
                        help="Save results to file")
    parser.add_argument('--profile', default=False, action='store_true',
                        help="Run test one more time and profile")
    parser.add_argument('--profiles-dir', default='profiles', type=str,
                        help="Run test one more time and profile")
    dtype = tf.float32
    args = parser.parse_args()

    # Needed to generate indices for partially optimized cases
    if args.num_param_cols % 10:
        sys.exit('ERROR: num_param_cols must be divisible by 10')

    # Open a file
    f = None
    if args.save_to:
        f = open(args.save_to, 'w')

    try:
        t = PerformanceTest(args.num_param_rows, args.num_param_cols,
                            args.num_param_slices,
                            args.out_size, args.num_ops,
                            args.num_runs, dtype,
                            args.without_cpu, args.without_gpu,
                            args.log_devices, f, args.profile, args.profiles_dir)
        t.run()
    except Exception as e:
        print("Error: ", e)
    finally:
        if f is not None:
            f.close()


if __name__ == '__main__':
    main()

import itertools
import subprocess
from collections import OrderedDict
from argparse import ArgumentParser
from copy import deepcopy


def configs():
    generative = dict(
        unsupervised=True,
        learning_algo="em",
        batch_size=4,
        completion=True,
        completion_by_marginal=True,
        dist='normal',
        sample_prob=0.5,
        log_weights=False,
        sum_num_c0=64, sum_num_c1=64, sum_num_c2=64, sum_num_c3=64, sum_num_c4=64,
        estimate_scale=True,
        normalize_data=True,
        num_epochs=100,
        stop_epsilon=1e-2
    )

    generative_olivetti_poon = dict(
        unsupervised=True,
        learning_algo='em',
        batch_size=16,
        num_epochs=25,
        completion=True,
        completion_by_marginal=True,
        dist='normal',
        log_weights=False,
        sum_num_c0=32, sum_num_c1=32, sum_num_c2=32, sum_num_c3=32, sum_num_c4=32,
        normalize_data=True,
        equidistant_means=False,
        estimate_scale=False,
        fixed_mean=True,
        fixed_variance=True,
        weight_init_min=1.0,
        weight_init_max=1.0,
        minimal_value_multiplier=1e-4,
        stop_epsilon=1e-1,
        update_period_unit="step",
        update_period_value=8,
    )

    small_olivetti = dict(
        sum_num_c0=2, sum_num_c1=2, sum_num_c2=2, sum_num_c3=2, sum_num_c4=2,
        not_depthwise=True, first_depthwise=True, num_components=4,
        batch_size=64, update_period_value=2, num_epochs=25, equidistant_means=False,
        additive_smoothing=1e-2, minimal_value_multiplier=0.0
    )

    soft_em = dict(
        sum_num_c0=32, sum_num_c1=32, sum_num_c2=32, sum_num_c3=32, sum_num_c4=32,
        not_depthwise=False, num_components=16,
        learning_algo='soft_em', batch_size=32, update_period_value=1,
        minimal_value_multiplier=1e-4, stop_epsilon=1e-3, num_epochs=100
    )

    small_caltech = dict(
        sum_num_c0=16, sum_num_c1=16, sum_num_c2=16, sum_num_c3=16, sum_num_c4=16,
        not_depthwise=False, first_depthwise=False, num_components=4,
        batch_size=16, update_period_value=8, num_epochs=25, equidistant_means=False,
        additive_smoothing=1e-2, minimal_value_multiplier=0.0, stop_epsilon=0.1
    )

    generative_olivetti_caltech = dict(
        unsupervised=True,
        learning_algo='em',
        batch_size=16,
        num_epochs=25,
        completion=True,
        completion_by_marginal=True,
        dist='normal',
        log_weights=False,
        sum_num_c0=16, sum_num_c1=16, sum_num_c2=16, sum_num_c3=16, sum_num_c4=16,
        normalize_data=True,
        equidistant_means=False,
        estimate_scale=False,
        fixed_mean=True,
        fixed_variance=True,
        weight_init_min=1.0,
        weight_init_max=1.0,
        minimal_value_multiplier=1e-2,
        stop_epsilon=1e-1,
        update_period_unit="step",
        update_period_value=4,
        num_components=32,
        first_depthwise=True
    )

    generative_mnist = dict(
        unsupervised=True,
        learning_algo='em',
        batch_size=16,
        num_epochs=25,
        completion=True,
        completion_by_marginal=True,
        dist='normal',
        log_weights=False,
        sum_num_c0=32, sum_num_c1=32, sum_num_c2=64, sum_num_c3=64, sum_num_c4=64,
        normalize_data=True,
        equidistant_means=False,
        estimate_scale=False,
        fixed_mean=True,
        fixed_variance=True,
        weight_init_min=1.0,
        weight_init_max=1.0,
        minimal_value_multiplier=1e-4,
        stop_epsilon=1e-2,
        update_period_unit="step",
        update_period_value=4
    )

    discriminative_mnist = dict(
        unsupervised=False,
        learning_algo='adam',
        batch_size=64,
        num_epochs=400,
        eval_batch_size=128,
        completion=False,
        completion_by_marginal=False,
        dist='normal',
        log_weights=True,
        sum_num_c0=64, sum_num_c1=64, sum_num_c2=64, sum_num_c3=128, sum_num_c4=128,
        normalize_data=True,
        normalize_batch_wise=False,
        equidistant_means=True,
        estimate_scale=False,
        fixed_mean=False,
        fixed_variance=True,
        reparameterized_weights=True,
        lr_decay_rate=1.0,
        variance_init=1.0,
        learning_rate=1e-3,
        minimal_value_multiplier=1e-4,
        stop_epsilon=1e-8,
        input_dropout=0.2,
        dropout_rate=0.2,
        first_depthwise=True,
        num_components=32
    )

    cauchy = dict(
        dist='cauchy'
    )

    exp_l2 = dict(
        learning_rate=1e-2,
        exp_l2=1e-3
    )

    export_completions = dict(
        export_completions=True,
        num_epochs=25
    )

    generative_additive_smoothing = dict(
        additive_smoothing=1.0,
        minimal_value_multiplier=0.0
    )

    generative_mpe = dict(
        value_inf_type='mpe'
    )

    generative_reset_per5 = dict(
        reset_per_epoch=5
    )

    generative_additive_smoothing1 = dict(
        additive_smoothing=1.0,
        minimal_value_multiplier=0.0
    )

    generative_l0_prior = dict(
        l0_prior_factor=1e-1
    )

    generative_l0_prior01 = dict(
        l0_prior_factor=0.01
    )

    generative_use_unweighted = dict(
        use_unweighted=True
    )

    generative_sampling = dict(
        sample_prob=0.5,
        sample_path=True
    )

    generative_sampling_05 = dict(
        sample_prob=0.5,
        sample_path=True
    )

    singleton_batch = dict(
        batch_size=1
    )

    discriminative = dict(
        unsupervised=False,
        learning_algo='amsgrad',
        batch_size=32, 
        dist='cauchy',
        log_weights=True,
        sum_num_c0=32, sum_num_c1=64, sum_num_c2=64, sum_num_c3=128, sum_num_c4=128,
        normalize_data=True,
        num_epochs=500,
        fixed_variance=True
    )
    
    generative_grid = OrderedDict(
        use_unweighted=[True, False],
        sample_path=[True, False],
        equidistant_means=[True, False]
    )
    
    discriminative_grid = OrderedDict(
        dropconnect_keep_prob=[1.0, 0.9],
        input_dropout=[0.0, 0.1],
    )
    
    augmentation = dict(
        width_shift_range=2 / 28,
        height_shift_range=2/28,
        zoom_range=0.1,
        rotation_range=5,
        shear_range=0.1
    )

    
    mnist_common = dict(dataset='mnist')
    fashion_mnist_common = dict(dataset='fashion_mnist')
    olivetti_common = dict(dataset='olivetti', num_components=4)
    caltech_common = dict(dataset='caltech', num_components=4)
    cifar10_common = dict(dataset='cifar10')
    return dict(
        mnist=[
            # (combine_dicts(generative_mnist, mnist_common, generative_additive_smoothing, export_completions, dict(name="MNISTCompletionBaseExport")), None),
            (combine_dicts(generative_mnist, mnist_common, small_caltech, dict(name="MNISTBase2")), None),
            (combine_dicts(generative_mnist, mnist_common, generative_use_unweighted, small_caltech, dict(name="MNISTCompletionUnweighted2")), None),
            # (combine_dicts(discriminative_mnist, mnist_common, dict(name="MNISTClassAdam")), None),
            # (combine_dicts(discriminative_mnist, mnist_common, exp_l2, dict(name="MNISTClassAdamExpL2")), None),
            # (combine_dicts(discriminative, mnist_common, augmentation, dict(name="MNIST_Discriminative_Augmented")), discriminative_grid)
        ],
        fashion_mnist=[
            (combine_dicts(generative_mnist, fashion_mnist_common, small_caltech, dict(name="FMNISTCompletionBase2")), None),
            (combine_dicts(generative_mnist, fashion_mnist_common, generative_use_unweighted, small_caltech,
                           dict(name="FMNISTCompletionUnweighted2")), None),
            # (combine_dicts(discriminative_mnist, fashion_mnist_common, dict(name="FMNISTAdam")), None),
        ],
        olivetti=[
            # (combine_dicts(generative_olivetti_poon, olivetti_common, small_olivetti, dict(name="BaseFair3")), None),
            # (combine_dicts(generative_olivetti_poon, olivetti_common, small_olivetti,
            #                generative_use_unweighted, export_completions, dict(name="UnweightedFair3Export")), None),
            # (combine_dicts(generative_olivetti_poon, olivetti_common, small_olivetti, cauchy,
            #                generative_use_unweighted, dict(name="OlivettiCauchy")), None),
            # (combine_dicts(generative_olivetti_poon, olivetti_common, small_olivetti, cauchy,
            #                dict(name="OlivettiCauchyBase")), None),
            (combine_dicts(generative_olivetti_poon, olivetti_common, small_olivetti, soft_em,
                           dict(name="OlivettiSoftEM")), None),
            # (combine_dicts(generative_olivetti_poon, olivetti_common, small_olivetti,
            #                generative_l0_prior, generative_additive_smoothing,
            #                dict(name="L0PriorFair")), None),
            # (combine_dicts(generative_olivetti_poon, olivetti_common, small_olivetti, export_completions,
            #                generative_use_unweighted, dict(name="UnweightedFairExport")), None),
            # (combine_dicts(generative_olivetti_poon, olivetti_common, small_olivetti,
            #                generative_l0_prior, dict(name="L0Prior")), None),
            # (combine_dicts(generative_olivetti_poon, olivetti_common,
            #                generative_l0_prior01, dict(name="L0Prior01")), None),
            # (combine_dicts(generative_olivetti_poon, olivetti_common,
            #                generative_mpe, dict(name="MPE")), None),
            # (combine_dicts(generative_olivetti_poon, olivetti_common,
            #                generative_reset_per5, dict(name="ResetPer5")), None),
        ],
        caltech=[
            (combine_dicts(generative_olivetti_poon, caltech_common, small_caltech, dict(name="CaltechBase3")), None),
            (combine_dicts(generative_olivetti_poon, caltech_common, small_caltech, generative_use_unweighted,
                           dict(name="CaltechUnweighted3")), None),
            # (combine_dicts(generative_olivetti_caltech, caltech_common, generative_reset_per5,
            #                dict(name="CaltechResetPer5")), None),
        ],
        cifar10=[
            # (combine_dicts(discriminative, cifar10_common, dict(name="Cifar10_Discriminative")), discriminative_grid),
            # (combine_dicts(discriminative, cifar10_common, augmentation,
            #                dict(name="Cifar10_Generative_Augmented")), discriminative_grid),
            (combine_dicts(discriminative_mnist, cifar10_common, dict(name="Cifar10")), None),
        ]
    )



def combine_dicts(*ds):
    out_dict = dict()
    for d in ds:
        out_dict.update(d)
    return out_dict


def key_val_to_arg(k, v):
    if isinstance(v, bool):
        if v:
            return "--{}".format(k)
        else:
            return ''
    return "--{}={}".format(k, v)


def abbreviate(n):
    return ''.join([s[0] for s in n.split('_')]).upper()


def experiment_group(defaults, grid_params):
    name_prefix = defaults['name']
    defaults = deepcopy(defaults)

    if grid_params is None:

        for _ in range(args.repeat):
            cmd = ['python3', './train.py'] + [key_val_to_arg(k, v) for k, v in defaults.items()]
            cmd = [c for c in cmd if len(c)]

            print("Running\n", ' '.join(cmd))
            process = subprocess.Popen(cmd)
            try:
                stdout, stderr = process.communicate()
            except KeyboardInterrupt as e:
                raise e
            finally:
                try:
                    process.terminate()
                except OSError:
                    pass
        return

    del defaults['name']
    for _ in range(args.repeat):
        for gp in itertools.product(*list(grid_params.values())):
            name_suffix = '_'.join(["{}={}".format(abbreviate(k), v) for k, v in zip(grid_params.keys(), gp)])
            cmd = ['python3', './train.py'] + [key_val_to_arg(k, v) for k, v in defaults.items()] + \
                [key_val_to_arg(k, v) for k, v in zip(grid_params.keys(), gp)] + ["--name={}_{}".format(name_prefix, name_suffix)]
            cmd = [c for c in cmd if len(c)]
            print("Running\n", ' '.join(cmd))
            process = subprocess.Popen(cmd)
            try:
                stdout, stderr = process.communicate()
            except KeyboardInterrupt as e:
                raise e
            finally:
                try:
                    process.terminate()
                except OSError:
                    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-subset", choices=['mnist', 'fashion_mnist', 'caltech', 'olivetti', 'cifar10'], 
        default=None, nargs='+')
    parser.add_argument("--repeat", default=1, type=int)
    args = parser.parse_args()
        
    if args.dataset_subset:
        cs = configs()
        for dataset in args.dataset_subset:
            for defaults, grid_params in cs[dataset]:
                experiment_group(defaults, grid_params)
    else:
        for cs in configs().values():
            for defaults, grid_params in cs:
                experiment_group(defaults, grid_params)

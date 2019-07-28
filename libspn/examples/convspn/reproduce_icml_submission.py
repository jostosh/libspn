import itertools
import subprocess
from collections import OrderedDict
from argparse import ArgumentParser
from copy import deepcopy


def configs():
    generative = dict(
        unsupervised=True,
        learning_algo="em",
        batch_size=1,
        completion=True,
        completion_by_marginal=True,
        dist='normal',
        sample_prob=0.5,
        log_weights=False,
        sum_num_c0=64, sum_num_c1=64, sum_num_c2=64, sum_num_c3=64, sum_num_c4=64,
        estimate_scale=True,
        normalize_data=True,
        num_epochs=100
    )

    generative_olivetti_poon = dict(
        unsupervised=True,
        # first_local_sum=True,
        # first_depthwise=True,
        learning_algo='em',
        batch_size=16,
        num_epochs=5,
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
    )

    generative_train_leafs = dict(
        fixed_mean=False,
        fixed_variance=False,
        equidistant_means=True
    )

    generative_initial_accum = dict(
        initial_accum_value=1e-4
    )

    generative_additive_smoothing = dict(
        additive_smoothing=1e-4,
        initial_accum_value=0.0
    )

    generative_l0_prior = dict(
        l0_prior_factor=1.0
    )

    generative_use_unweighted = dict(
        use_unweighted=True
    )

    generative_sampling = dict(
        sample_prob=0.25,
        sample_path=True,
        # batch_size=12
    )

    generative_olivetti_grid = OrderedDict(
        l0_prior_factor=[1e-1, 1.0, 10.0]
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
    
    mnist_common = dict(dataset='mnist', num_components=2)
    fashion_mnist_common = dict(dataset='fashion_mnist', num_components=4)
    olivetti_common = dict(dataset='olivetti', num_components=4)
    caltech_common = dict(dataset='caltech', num_components=8)
    cifar10_common = dict(dataset='cifar10', num_components=32, first_depthwise=True)
    
    return dict(
        mnist=[
            (combine_dicts(generative, mnist_common, dict(name="MNIST_Generative")), generative_grid),
            (combine_dicts(discriminative, mnist_common, dict(name="MNIST_Discriminative")), discriminative_grid),
            (combine_dicts(discriminative, mnist_common, augmentation, dict(name="MNIST_Discriminative_Augmented")), discriminative_grid)
        ],
        fashion_mnist=[
            (combine_dicts(generative, fashion_mnist_common, dict(name="FashionMNIST_Generative")),
             generative_grid),
            (combine_dicts(discriminative, fashion_mnist_common, dict(name="FashionMNIST Discriminative")),
             discriminative_grid),
            (combine_dicts(discriminative, fashion_mnist_common, augmentation,
                           dict(name="FashionMNIST_Discriminative_Augmented")), discriminative_grid)
        ],
        olivetti=[
            # (combine_dicts(generative_olivetti_poon, olivetti_common,
            #                generative_additive_smoothing, dict(name="Base")), None),
            # (combine_dicts(generative_olivetti_poon, olivetti_common,
            #                generative_additive_smoothing, generative_l0_prior,
            #                dict(name="L0Prior")), None),
            (combine_dicts(generative_olivetti_poon, olivetti_common, generative_additive_smoothing,
                           generative_sampling, dict(name="Sampling")), None),
            (combine_dicts(generative_olivetti_poon, olivetti_common, generative_additive_smoothing,
                           generative_use_unweighted, dict(name="Unweighted")), None),
            (combine_dicts(generative_olivetti_poon, olivetti_common, generative_additive_smoothing,
                           generative_train_leafs, dict(name="TrainLeaves")), None),
            (combine_dicts(generative_olivetti_poon, olivetti_common,
                           generative_additive_smoothing, generative_l0_prior,
                           generative_use_unweighted, generative_sampling,
                           dict(name="AllNoTrainLeaves")), None),
            (combine_dicts(generative_olivetti_poon, olivetti_common,
                           generative_additive_smoothing, generative_l0_prior,
                           generative_use_unweighted,
                           generative_train_leafs, dict(name="AllNoSampling")), None),
            (combine_dicts(generative_olivetti_poon, olivetti_common,
                           generative_additive_smoothing, generative_l0_prior,
                           generative_sampling,
                           generative_train_leafs, dict(name="AllNoUnweighted")), None),
            (combine_dicts(generative_olivetti_poon, olivetti_common,
                           generative_additive_smoothing,
                           generative_sampling, generative_use_unweighted,
                           generative_train_leafs, dict(name="AllNoL0Prior")), None),
            (combine_dicts(generative_olivetti_poon, olivetti_common,
                           generative_additive_smoothing, generative_l0_prior,
                           generative_use_unweighted, generative_sampling,
                           generative_train_leafs, dict(name="All")), None)
        ],
        caltech=[
            (combine_dicts(generative, caltech_common, dict(name="Caltech")), generative_grid)
        ],
        cifar10=[
            (combine_dicts(discriminative, cifar10_common, dict(name="Cifar10_Discriminative")), discriminative_grid),
            (combine_dicts(discriminative, cifar10_common, augmentation,
                           dict(name="Cifar10_Generative_Augmented")), discriminative_grid)
        ]
    )


def combine_dicts(*ds):
    return {k: v for k, v in itertools.chain(*[d.items() for d in ds])}


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

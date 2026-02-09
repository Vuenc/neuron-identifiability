from typing import Dict


checkpoint_directories_by_architecture: Dict[str, Dict[str, str]] = {
    "mlp": {
        "mlp_symmetry0": "outputs/2025-12-17/12-18-16_mlp_mnist_sym-0__3pb1rw8b__kk13eg0q",
        "mlp_symmetry1_kappa0": "outputs/2025-12-17/14-20-51_mlp_mnist_sym-1__m3z8jvf9__iwzo2u22",
        "mlp_symmetry1_kappa1": "outputs/2025-12-17/12-30-55_mlp_mnist_sym-1__3pb1rw8b__4b0gt1xm",
        "mlp_symmetry1_kappaPerLayer": "outputs/2025-12-17/12-46-23_mlp_mnist_sym-1__3pb1rw8b__8fvvk8f3",
        # "mlp_symmetry2": "outputs/2025-12-17/13-01-05_mlp_mnist_sym-2__3pb1rw8b__48q9fddu",
        "mlp_symmetry3_kappa0": "outputs/2025-12-17/13-16-26_mlp_mnist_sym-3__3pb1rw8b__tq3utpfq",
        "mlp_symmetry3_kappa1": "outputs/2025-12-17/13-30-38_mlp_mnist_sym-3__3pb1rw8b__ruzzxkpy",
        "mlp_symmetry3_kappaPerLayer": "outputs/2025-12-17/13-45-10_mlp_mnist_sym-3__3pb1rw8b__e4vv3n8v",
    },
    "mlp-kappa-sweep": {
        "mlp_symmetry1_kappa0.0": "outputs/2025-12-17/14-20-51_mlp_mnist_sym-1__m3z8jvf9__iwzo2u22",
        'mlp_symmetry1_kappa0.01': 'outputs/2026-01-26/15-47-14_mlp_mnist_sym-1__f7w54qp9__fx14wg6s',
        'mlp_symmetry1_kappa0.02': 'outputs/2026-01-26/16-14-09_mlp_mnist_sym-1__f7w54qp9__aw5dyme9',
        'mlp_symmetry1_kappa0.05': 'outputs/2026-01-26/16-31-41_mlp_mnist_sym-1__f7w54qp9__5cpb540s',
        'mlp_symmetry1_kappa0.1': 'outputs/2026-01-26/16-22-57_mlp_mnist_sym-1__f7w54qp9__k799iq97',
        'mlp_symmetry1_kappa0.2': 'outputs/2026-01-26/16-40-28_mlp_mnist_sym-1__f7w54qp9__ghm6ruvt',
        'mlp_symmetry1_kappa0.5': 'outputs/2026-01-26/15-55-57_mlp_mnist_sym-1__f7w54qp9__ot3dcy8q',
        "mlp_symmetry1_kappa1.0": "outputs/2025-12-17/12-30-55_mlp_mnist_sym-1__3pb1rw8b__4b0gt1xm",
        'mlp_symmetry1_kappa2.0': 'outputs/2026-01-26/16-49-17_mlp_mnist_sym-1__f7w54qp9__jhejafgl',
        'mlp_symmetry1_kappa5.0': 'outputs/2026-01-26/16-04-48_mlp_mnist_sym-1__f7w54qp9__05dtz7bk',
    },
    "resnet": {
        "resnet_symmetry0": "outputs/2025-12-18/19-14-23_resnet_cifar_sym-0__r3aiubzb__au3i07iw",
        "resnet_symmetry1_kappa0": "outputs/2025-12-18/23-27-30_resnet_cifar_sym-1__r3aiubzb__6xrlc0ln",
        "resnet_symmetry1_kappa2": "outputs/2025-12-19/17-19-27_resnet_cifar_sym-1__38uctfm6__2y80rlyo",
        "resnet_symmetry3_kappa2": "outputs/2026-01-26/11-22-32_resnet_cifar_sym-3__fr9ukbz7__9jpjp7my",
    },
    "resnet-kappa-sweep": {
        "resnet_symmetry1_kappa0": "outputs/2025-12-18/23-27-30_resnet_cifar_sym-1__r3aiubzb__6xrlc0ln",
        'resnet_symmetry1_kappa0.01': 'outputs/2026-01-26/13-52-11_resnet_cifar_sym-1__tascbzwm__yf4rx8oz',
        'resnet_symmetry1_kappa0.02': 'outputs/2026-01-26/18-17-45_resnet_cifar_sym-1__tascbzwm__kkyvk6ui',
        'resnet_symmetry1_kappa0.05': 'outputs/2026-01-26/20-30-30_resnet_cifar_sym-1__tascbzwm__y9v2v7lt',
        'resnet_symmetry1_kappa0.1': 'outputs/2026-01-26/19-23-52_resnet_cifar_sym-1__tascbzwm__w2b7ed3p',
        'resnet_symmetry1_kappa0.2': 'outputs/2026-01-26/21-36-42_resnet_cifar_sym-1__tascbzwm__x1i2fcek',
        'resnet_symmetry1_kappa0.5': 'outputs/2026-01-26/14-58-20_resnet_cifar_sym-1__tascbzwm__wqf0d2y0',
        'resnet_symmetry1_kappa1': 'outputs/2026-01-26/16-05-05_resnet_cifar_sym-1__tascbzwm__0krlyere',
        "resnet_symmetry1_kappa2": "outputs/2025-12-19/17-19-27_resnet_cifar_sym-1__38uctfm6__2y80rlyo",
        'resnet_symmetry1_kappa5': 'outputs/2026-01-26/17-11-39_resnet_cifar_sym-1__tascbzwm__7zxpysr8',
    }
}

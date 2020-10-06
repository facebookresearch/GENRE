import fb_sweep.sweep as sweep
from fb_sweep.sweep import hyperparam


def get_grid(args):
    grid = []

    total_num_udpates = 20000
    warmup_updates = 500
    num_data_loaders = 0
    arch = "bart_large"
    task = "translation"
    #     task = "noisy_translation"
    criterion = "label_smoothed_cross_entropy"
    #     criterion = "cross_entropy"

    adam_eps = 1e-08
    weight_decay = 0.01
    update_freq = 4 if args.num_nodes == 1 else 1

    restore_file = (
        "/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/models/kilt/checkpoint.pt"
    )
    # "/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/models/el/checkpoint_aidayago.pt"
    # "/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/models/kilt/checkpoint.pt"

    #     restore_file = "/checkpoint/ndecao/2020-07-29/new_fairseq_query2title_blink_constrained.bart_large.ls0.1.mt2048.uf1.mu200000.dr0.1.atdr0.1.actdr0.0.wd0.01.adam.beta9999.eps1e-08.clip0.1.lr3e-05.warm500.fp16.ngpu32/checkpoint_last.pt"

    #     restore_file = "/checkpoint/ndecao/2020-07-28/new_fairseq_query2title_blink.bart_large.ls0.1.mt2048.uf1.mu200000.dr0.1.atdr0.1.actdr0.0.wd0.01.adam.beta9999.eps1e-08.clip0.1.lr3e-05.warm500.fp16.ngpu32/checkpoint_best_blink.pt"

    #     restore_file = "/checkpoint/ndecao/globalel_wiki_abs.bart_large.ls0.1.mt2048.uf1.mu200000.dr0.1.atdr0.1.actdr0.0.wd0.01.adam.beta9999.eps1e-08.clip0.1.lr3e-05.warm500.fp16.ngpu64/checkpoint_last.pt"

    grid += [hyperparam("--restore-file", restore_file)]

    # model settings
    grid += [
        hyperparam("--arch", arch, save_dir_key=lambda val: val),
        hyperparam("--task", task),
        hyperparam("--criterion", criterion),
        hyperparam("--source-lang", "source"),
        hyperparam("--target-lang", "target"),
        hyperparam("--truncate-source"),
        hyperparam("--label-smoothing", 0.1, save_dir_key=lambda val: f"ls{val}"),
    ]

    grid += [
        hyperparam("--max-tokens", 2048, save_dir_key=lambda val: f"mt{val}"),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam(
            "--max-update", total_num_udpates, save_dir_key=lambda val: f"mu{val}"
        ),
        hyperparam("--required-batch-size-multiple", 1),
    ]
    # regularization
    grid += [
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"actdr{val}"),
        hyperparam("--weight-decay", weight_decay, save_dir_key=lambda val: f"wd{val}"),
    ]

    # optimization settings
    grid += [
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.999)", save_dir_key=lambda val: "beta9999"),
        hyperparam("--adam-eps", adam_eps, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.1, save_dir_key=lambda val: f"clip{val}"),
    ]

    # lr scheduler
    grid += [
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", 3e-05, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", total_num_udpates),
        hyperparam(
            "--warmup-updates", warmup_updates, save_dir_key=lambda val: f"warm{val}"
        ),
    ]
    grid += [
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("--ddp-backend", "no_c10d"),
    ]

    # data loading settings
    grid += [hyperparam("--num-workers", num_data_loaders)]

    # validation and checkpoint settings
    grid += [
        # hyperparam("--no-save"),
        # hyperparam("--no-epoch-checkpoints"),
        hyperparam("--reset-meters"),
        hyperparam("--reset-optimizer"),
    ]

    grid += [
        hyperparam("--share-all-embeddings"),
        hyperparam("--layernorm-embedding"),
        hyperparam("--share-decoder-input-output-embed"),
    ]

    # logging settings
    grid += [
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 10),
    ]

    if args.local:
        grid += [hyperparam("--log-format", "json"), hyperparam("--log-interval", 1)]

    grid += [hyperparam("--patience", 200)]
    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)

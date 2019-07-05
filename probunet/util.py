import argparse
import os
import shutil
import numpy as np
import torch
from trixi.util import Config, GridSearch


def check_attributes(object_, attributes):

    missing = []
    for attr in attributes:
        if not hasattr(object_, attr):
            missing.append(attr)
    if len(missing) > 0:
        return False
    else:
        return True


def set_seeds(seed, cuda=True):
    if not hasattr(seed, "__iter__"):
        seed = (seed, seed, seed)
    np.random.seed(seed[0])
    torch.manual_seed(seed[1])
    if cuda: torch.cuda.manual_seed_all(seed[2])


def make_onehot(array, labels=None, axis=1, newaxis=False):

    # get labels if necessary
    if labels is None:
        labels = np.unique(array)
        labels = list(map(lambda x: x.item(), labels))

    # get target shape
    new_shape = list(array.shape)
    if newaxis:
        new_shape.insert(axis, len(labels))
    else:
        new_shape[axis] = new_shape[axis] * len(labels)

    # make zero array
    if type(array) == np.ndarray:
        new_array = np.zeros(new_shape, dtype=array.dtype)
    elif torch.is_tensor(array):
        new_array = torch.zeros(new_shape, dtype=array.dtype, device=array.device)
    else:
        raise TypeError("Onehot conversion undefined for object of type {}".format(type(array)))

    # fill new array
    n_seg_channels = 1 if newaxis else array.shape[axis]
    for seg_channel in range(n_seg_channels):
        for l, label in enumerate(labels):
            new_slc = [slice(None), ] * len(new_shape)
            slc = [slice(None), ] * len(array.shape)
            new_slc[axis] = seg_channel * len(labels) + l
            if not newaxis:
                slc[axis] = seg_channel
            new_array[tuple(new_slc)] = array[tuple(slc)] == label

    return new_array


def match_to(x, ref, keep_axes=(1,)):

    target_shape = list(ref.shape)
    for i in keep_axes:
        target_shape[i] = x.shape[i]
    target_shape = tuple(target_shape)
    if x.shape == target_shape:
        pass
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() == 2:
        while x.dim() < len(target_shape):
            x = x.unsqueeze(-1)

    x = x.expand(*target_shape)
    x = x.to(device=ref.device, dtype=ref.dtype)

    return x


def make_slices(original_shape, patch_shape):

    working_shape = original_shape[-len(patch_shape):]
    splits = []
    for i in range(len(working_shape)):
        splits.append([])
        for j in range(working_shape[i] // patch_shape[i]):
            splits[i].append(slice(j*patch_shape[i], (j+1)*patch_shape[i]))
        rest = working_shape[i] % patch_shape[i]
        if rest > 0:
            splits[i].append(slice((j+1)*patch_shape[i], (j+1)*patch_shape[i] + rest))

    # now we have all slices for the individual dimensions
    # we need their combinatorial combinations
    slices = list(itertools.product(*splits))
    for i in range(len(slices)):
        slices[i] = [slice(None), ] * (len(original_shape) - len(patch_shape)) + list(slices[i])

    return slices


def coordinate_grid_samples(mean, std, factor_std=5, scale_std=1.):

    relative = np.linspace(-scale_std*factor_std, scale_std*factor_std, 2*factor_std+1)
    positions = np.array([mean + i * std for i in relative]).T
    axes = np.meshgrid(*positions)
    axes = map(lambda x: list(x.ravel()), axes)
    samples = list(zip(*axes))
    samples = list(map(np.array, samples))

    return samples


def get_default_experiment_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str, help="Working directory for experiment.")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to a config file.")
    parser.add_argument("-v", "--visdomlogger", action="store_true", help="Use visdomlogger.")
    parser.add_argument("-tx", "--tensorboardxlogger", type=str, default=None)
    parser.add_argument("-tl", "--telegramlogger", action="store_true")
    parser.add_argument("-dc", "--default_config", type=str, default="DEFAULTS", help="Select a default Config")
    parser.add_argument("-ad", "--automatic_description", action="store_true")
    parser.add_argument("-r", "--resume", type=str, default=None, help="Path to resume from")
    parser.add_argument("-irc", "--ignore_resume_config", action="store_true", help="Ignore Config in experiment we resume from.")
    parser.add_argument("-test", "--test", action="store_true", help="Run test instead of training")
    parser.add_argument("-g", "--grid", type=str, help="Path to a config for grid search")
    parser.add_argument("-s", "--skip_existing", action="store_true", help="Skip configs for which an experiment exists, only for grid search")
    parser.add_argument("-m", "--mods", type=str, nargs="+", default=None, help="Mods are Config stubs to update only relevant parts for a certain setup.")
    parser.add_argument("-ct", "--copy_test", action="store_true", help="Copy test files to original experiment.")

    return parser


def run_experiment(experiment, configs, args, mods=None, **kwargs):

    # set a few defaults
    if "explogger_kwargs" not in kwargs:
        kwargs["explogger_kwargs"] = dict(folder_format="{experiment_name}_%Y%m%d-%H%M%S")
    if "explogger_freq" not in kwargs:
        kwargs["explogger_freq"] = 1
    if "resume_save_types" not in kwargs:
        kwargs["resume_save_types"] = ("model", "simple", "th_vars", "results")

    config = Config(file_=args.config) if args.config is not None else Config()
    config.update_missing(configs[args.default_config].deepcopy())
    if args.mods is not None and mods is not None:
        for mod in args.mods:
            config.update(mods[mod])
    config = Config(config=config, update_from_argv=True)

    # GET EXISTING EXPERIMENTS TO BE ABLE TO SKIP CERTAIN CONFIGS
    if args.skip_existing:
        existing_configs = []
        for exp in os.listdir(args.base_dir):
            try:
                existing_configs.append(Config(file_=os.path.join(args.base_dir, exp, "config", "config.json")))
            except Exception as e:
                pass

    if args.grid is not None:
        grid = GridSearch().read(args.grid)
    else:
        grid = [{}]

    for combi in grid:

        config.update(combi)

        if args.skip_existing:
            skip_this = False
            for existing_config in existing_configs:
                if existing_config.contains(config):
                    skip_this = True
                    break
            if skip_this:
                continue

        if "backup_every" in config:
            kwargs["save_checkpoint_every_epoch"] = config["backup_every"]

        loggers = {}
        if args.visdomlogger:
            loggers["v"] = ("visdom", {}, 1)
        if args.tensorboardxlogger is not None:
            if args.tensorboardxlogger == "same":
                loggers["tx"] = ("tensorboard", {}, 1)
            else:
                loggers["tx"] = ("tensorboard", {"target_dir": args.tensorboardxlogger}, 1)

        if args.telegramlogger:
            kwargs["use_telegram"] = True

        if args.automatic_description:
            difference_to_default = Config.difference_config_static(config, configs["DEFAULTS"]).flat(keep_lists=True, max_split_size=0, flatten_int=True)
            description_str = ""
            for key, val in difference_to_default.items():
                val = val[0]
                description_str = "{} = {}\n{}".format(key, val, description_str)
            config.description = description_str

        exp = experiment(config=config,
                         base_dir=args.base_dir,
                         resume=args.resume,
                         ignore_resume_config=args.ignore_resume_config,
                         loggers=loggers,
                         **kwargs)

        trained = False
        if args.resume is None or args.test is False:
            exp.run()
            trained = True
        if args.test:
            exp.run_test(setup=not trained)
            if isinstance(args.resume, str) and exp.elog is not None and args.copy_test:
                for f in glob.glob(os.path.join(exp.elog.save_dir, "test*")):
                    if os.path.isdir(f):
                        shutil.copytree(f, os.path.join(args.resume, "save", os.path.basename(f)))
                    else:
                        shutil.copy(f, os.path.join(args.resume, "save"))
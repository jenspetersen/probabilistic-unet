from probunet.experiment.probabilistic_unet_future_segmentation import ProbabilisticUNetFutureSegmentation

import os
import numpy as np
import time
import torch
from torch import nn, optim, distributions
from trixi.util import Config
from batchgenerators.transforms import (
    MirrorTransform,
    SpatialTransform,
    CenterCropTransform,
    SegLabelSelectionBinarizeTransform,
    Compose
)
from batchgenerators.dataloading import MultiThreadedAugmenter

from probunet.model import ProbabilisticSegmentationNet, InjectionUNet3D, InjectionConvEncoder3D
from probunet.eval import Evaluator, dice
from probunet.util import (
    get_default_experiment_parser,
    run_experiment,
    make_onehot as make_onehot_segmentation,
    coordinate_grid_samples
)
from probunet import data



DESCRIPTION = "Segmentation with a Probabilistic U-Net. .test() will give results for upper bound in paper, .test_future() will give results for lower bound."


def make_defaults(patch_size=112,
                  in_channels=4,
                  latent_size=3,
                  labels=[0, 1, 2, 3]):

    if hasattr(patch_size, "__iter__"):
        if len(patch_size) > 1:
            patch_size = tuple(patch_size)
        else:
            patch_size = patch_size[0]
    if not hasattr(patch_size, "__iter__"):
        patch_size = tuple([patch_size, ] * 3)

    DEFAULTS = Config(

        # Base
        name=os.path.basename(__file__).split(".")[0],
        description=DESCRIPTION,
        n_epochs=50000,
        batch_size=2,
        batch_size_val=1,
        patch_size=patch_size,
        in_channels=in_channels,
        out_channels=len(labels),
        latent_size=latent_size,
        seed=1,
        device="cuda",

        # Data
        split_val=3,
        split_test=4,
        data_module=data,
        data_dir=None,  # we're setting data_module.data_dir if this is given
        mmap_mode="r",
        npz=False,
        debug=0,  # 1 selects (10, 5, 5) patients, 2 a single batch
        train_on_all=False,  # adds val and test to training set
        generator_train=data.RandomBatchGenerator,
        generator_val=data.LinearBatchGenerator,
        transforms_train={
            0: {
                "type": SpatialTransform,
                "kwargs": {
                    "patch_size": patch_size,
                    "patch_center_dist_from_border": patch_size[0] // 2,
                    "do_elastic_deform": False,
                    "p_el_per_sample": 0.2,
                    "p_rot_per_sample": 0.3,
                    "p_scale_per_sample": 0.3
                },
                "active": True
            },
            1: {
                "type": MirrorTransform,
                "kwargs": {"axes": (0, 1, 2)},
                "active": True
            },
            2: {
                "type": SegLabelSelectionBinarizeTransform,
                "kwargs": {"label": [1, 2, 3]},
                "active": False
            }
        },
        transforms_val={
            0: {
                "type": CenterCropTransform,
                "kwargs": {"crop_size": patch_size},
                "active": False
            },
            1: {
                "type": SegLabelSelectionBinarizeTransform,
                "kwargs": {"label": [1, 2, 3]},
                "active": False
            },
            2: {
                "type": SpatialTransform,
                "kwargs": {
                    "patch_size": patch_size,
                    "patch_center_dist_from_border": patch_size[0] // 2,
                    "do_elastic_deform": False,
                    "do_rotation": False,
                    "do_scale": True,
                    "p_scale_per_sample": 1,
                    "scale": (1.25, 1.25)
                },
                "active": False
            }
        },
        augmenter_train=MultiThreadedAugmenter,
        augmenter_train_kwargs={
            "num_processes": 11,
            "num_cached_per_queue": 6,
            "pin_memory": True
        },
        augmenter_val=MultiThreadedAugmenter,
        augmenter_val_kwargs={
            "num_processes": 2,
            "pin_memory": True
        },

        # Model
        model=ProbabilisticSegmentationNet,
        model_kwargs={
            "in_channels": in_channels,
            "out_channels": len(labels),
            "num_feature_maps": 24,
            "latent_size": latent_size,
            "depth": 5,
            "latent_distribution": distributions.Normal,
            "task_op": InjectionUNet3D,
            "task_kwargs": {
                "output_activation_op": nn.LogSoftmax,
                "output_activation_kwargs": {"dim": 1},
                "activation_kwargs": {"inplace": True}
            },
            "prior_op": InjectionConvEncoder3D,
            "prior_kwargs": {
                "in_channels": in_channels,
                "out_channels": latent_size * 2,
                "depth": 5,
                "block_depth": 2,
                "num_feature_maps": 24,
                "feature_map_multiplier": 2,
                "activation_kwargs": {"inplace": True},
                "norm_depth": 2,
            },
            "posterior_op": InjectionConvEncoder3D,
            "posterior_kwargs": {
                "in_channels": in_channels + len(labels),
                "out_channels": latent_size * 2,
                "depth": 5,
                "block_depth": 2,
                "num_feature_maps": 24,
                "feature_map_multiplier": 2,
                "activation_kwargs": {"inplace": True},
                "norm_depth": 2,
            },
        },
        model_init_weights_args=[nn.init.kaiming_uniform_, 0],
        model_init_bias_args=[nn.init.constant_, 0],

        # Learning
        optimizer=optim.Adam,
        optimizer_kwargs={"lr": 1e-4},
        scheduler=optim.lr_scheduler.StepLR,
        scheduler_kwargs={"step_size": 200, "gamma": 0.985},
        criterion_segmentation=nn.NLLLoss,
        criterion_segmentation_kwargs={"reduction": "sum"},
        criterion_latent=distributions.kl_divergence,
        criterion_latent_kwargs={},
        criterion_latent_init=False,
        criterion_segmentation_seg_onehot=False,
        criterion_segmentation_weight=1.0,
        criterion_latent_weight=1.0,
        criterion_segmentation_seg_dtype=torch.long,

        # Logging
        backup_every=1000,
        validate_every=1000,
        validate_subset=0.1,  # validate only this percentage randomly
        show_every=10,
        validate_metrics=["Dice"],
        labels=labels,
        evaluator=Evaluator,
        evaluator_kwargs={
            "label_values": list(labels) + [tuple(labels[1:])],
            "label_names": {
                0: "Background",
                1: "Edema",
                2: "Enhancing",
                3: "Necrosis",
                tuple(labels[1:]): "Whole Tumor"
            },
            "nan_for_nonexisting": True
        },
        val_save_output=False,
        val_example_samples=10,
        val_save_images=False,
        latent_plot_range=[-5, 5],
        test_on_val=True,
        test_save_output=False,
        test_future=True,
        test_std_factor=3,
        test_std_scale=1.

    )

    TASKMEAN = Config(
        criterion_segmentation_kwargs={"reduction": "elementwise_mean"}
    )

    ELASTIC = Config(
        transforms_train={0: {"kwargs": {"do_elastic_deform": True}}}
    )

    NONORM = Config(
        model_kwargs={
            "prior_kwargs": {"norm_depth": 0},
            "posterior_kwargs": {"norm_depth": 0}
        }
    )

    FULLNORM = Config(
        model_kwargs={
            "prior_kwargs": {"norm_depth": "full"},
            "posterior_kwargs": {"norm_depth": "full"}
        }
    )

    BATCHNORM = Config(
        model_kwargs={
            "prior_kwargs": {"norm_op": nn.BatchNorm3d},
            "posterior_kwargs": {"norm_op": nn.BatchNorm3d},
            "task_kwargs": {"norm_op": nn.BatchNorm3d}
        }
    )

    WHOLETUMOR = Config(
        transforms_train={2: {"active": True}},
        transforms_val={1: {"active": True}},
        out_channels=2,
        labels=[0, 1],
        model_kwargs={
            "out_channels": 2,
            "posterior_kwargs": {"in_channels": in_channels + 2}
        },
        evaluator_kwargs={
            "label_values": [0, 1],
            "label_names": {
                0: "Background",
                1: "Whole Tumor"
            }
        }
    )

    ENHANCING = Config(
        transforms_train={2: {
            "kwargs": {"label": 2},
            "active": True
        }},
        transforms_val={1: {
            "kwargs": {"label": 2},
            "active": True
        }},
        out_channels=2,
        labels=[0, 1],
        model_kwargs={
            "out_channels": 2,
            "posterior_kwargs": {"in_channels": in_channels + 2}
        },
        evaluator_kwargs={
            "label_values": [0, 1],
            "label_names": {
                0: "Background",
                1: "Whole Tumor"
            }
        }
    )

    NOAUGMENT = Config(
        transforms_train={
            0: {
                "kwargs": {
                    "p_el_per_sample": 0,
                    "p_rot_per_sample": 0,
                    "p_scale_per_sample": 0
                }
            },
            1: {"active": False}
        }
    )

    LOWAUGMENT = Config(
        transforms_train={
            0: {
                "kwargs": {
                    "p_el_per_sample": 0.,
                    "p_rot_per_sample": 0.15,
                    "p_scale_per_sample": 0.15
                }
            }
        }
    )

    NOBG = Config(
        criterion_segmentation_kwargs={"ignore_index": 0}
    )

    VALIDATEPATCHED = Config(
        transforms_val={2: {"active": True}}
    )

    MODS = {

        "TASKMEAN": TASKMEAN,
        "ELASTIC": ELASTIC,
        "NONORM": NONORM,
        "FULLNORM": FULLNORM,
        "BATCHNORM": BATCHNORM,
        "WHOLETUMOR": WHOLETUMOR,
        "ENHANCING": ENHANCING,
        "NOAUGMENT": NOAUGMENT,
        "LOWAUGMENT": LOWAUGMENT,
        "NOBG": NOBG,
        "VALIDATEPATCHED": VALIDATEPATCHED

    }

    return {"DEFAULTS": DEFAULTS}, MODS


class ProbabilisticUNetSegmentation(ProbabilisticUNetFutureSegmentation):

    def setup_data(self):

        c = self.config

        self.data_train = c.data_module.load(c.mmap_mode, subjects=c.subjects_train, npz=c.npz)
        self.data_val = c.data_module.load(c.mmap_mode, subjects=c.subjects_val, npz=c.npz)
        self.data_test = c.data_module.load(c.mmap_mode, subjects=c.subjects_test, npz=c.npz)

        self.generator_train = c.generator_train(
            self.data_train, c.batch_size, 3,
            number_of_threads_in_multithreaded=c.augmenter_train_kwargs.num_processes)
        self.generator_val = c.generator_val(
            self.data_val, c.batch_size_val, 3,
            number_of_threads_in_multithreaded=c.augmenter_val_kwargs.num_processes)
        self.generator_test = c.generator_val(
            self.data_test, c.batch_size_val, 3,
            number_of_threads_in_multithreaded=c.augmenter_val_kwargs.num_processes)

    def process_data(self, data, epoch):

        c = self.config

        if c.debug < 2 or epoch == 0:

            input_ = torch.from_numpy(data["data"][:, c.in_channels:2*c.in_channels]).to(dtype=torch.float32, device=c.device)
            gt_segmentation = make_onehot_segmentation(torch.from_numpy(data["seg"][:, -2:-1]).to(dtype=torch.float32, device=c.device), c.labels)

            if c.debug == 2:
                self._memo_batch = (input_, gt_segmentation, data)

        else:

            return self._memo_batch

        return input_, gt_segmentation, data

    def process_data_future(self, data, epoch):

        c = self.config

        if c.debug < 2 or epoch == 0:

            input_ = torch.from_numpy(data["data"][:, c.in_channels:2*c.in_channels]).to(dtype=torch.float32, device=c.device)
            gt_segmentation = make_onehot_segmentation(torch.from_numpy(data["seg"][:, -1:]).to(dtype=torch.float32, device=c.device), c.labels)

            if c.debug == 2:
                self._memo_batch = (input_, gt_segmentation, data)

        else:

            return self._memo_batch

        return input_, gt_segmentation, data

    def test(self):

        info = self.validate_make_default_info()
        info["coords"]["Metric"] = info["coords"]["Metric"] + ["Reference KL",
                                                               "Reference Reconstruction NLL",
                                                               "Reference Reconstruction Dice",
                                                               "Prior Maximum Dice",
                                                               "Prior Best Volume Dice"]

        if self.config.test_on_val:
            augmenter = self.augmenter_val
        else:
            augmenter = self.augmenter_test
        test_scores, info = self.test_inner(augmenter, [], info)
        test_scores = np.array(test_scores)

        self.elog.save_numpy_data(test_scores, "test.npy")
        self.elog.save_dict(info, "test.json")

        if self.config.test_future:
            self.test_future()

    def test_inner(self, augmenter, scores, info, future=False):

        c = self.config

        with torch.no_grad():

            self.model.eval()

            for data in augmenter:

                if future:
                    input_, gt_segmentation, data = self.process_data_future(data, 0)
                else:
                    input_, gt_segmentation, data = self.process_data(data, 0)
                prediction = self.model(input_, gt_segmentation, make_onehot=False).cpu()

                self.model.encode_posterior(input_, gt_segmentation, make_onehot=False)
                reference_reconstruction = self.model.reconstruct(out_device="cpu")
                gt_segmentation = torch.argmax(gt_segmentation.cpu(), 1, keepdim=False)
                reference_kl = distributions.kl_divergence(self.model.posterior, self.model.prior)

                # sample latent space for volumes and dice scores
                prior_mean = self.model.prior.loc.cpu().numpy()
                prior_mean = prior_mean.reshape(prior_mean.shape[0], -1)
                prior_std = self.model.prior.scale.cpu().numpy()
                prior_std = prior_std.reshape(prior_std.shape[0], -1)
                batch_samples = []
                for b in range(prior_mean.shape[0]):
                    batch_samples.append(coordinate_grid_samples(prior_mean[b], prior_std[b], c.test_std_factor, c.test_std_scale))

                if len(batch_samples) > 1:
                    batch_samples = list(zip(*batch_samples))
                    batch_samples = list(map(np.stack, batch_samples))
                else:
                    batch_samples = batch_samples[0]
                    batch_samples = list(map(lambda x: x[np.newaxis, ...], batch_samples))

                volumes = np.zeros((prior_mean.shape[0], len(batch_samples)))
                dice_scores = np.zeros((prior_mean.shape[0], len(batch_samples)))

                for s, sample in enumerate(batch_samples):
                    sample = torch.tensor(sample)
                    sample_prediction = self.model.reconstruct(sample=sample, out_device="cpu")
                    sample_prediction = torch.argmax(sample_prediction, 1, keepdim=False).numpy()
                    volumes_current = []
                    dice_scores_current = []
                    for b in range(sample_prediction.shape[0]):
                        volumes_current.append(np.sum(sample_prediction[b] != 0))
                        dice_scores_current.append(dice(sample_prediction[b] != 0, data["seg"][b, -1] != 0))
                    volumes[:, s] = volumes_current
                    dice_scores[:, s] = dice_scores_current

                max_dice = np.max(dice_scores, 1)
                dice_for_best_volume = []
                for b in range(volumes.shape[0]):
                    idx = np.argmin(np.abs(volumes[b] - np.sum(data["seg"][b, -1] != 0)))
                    dice_for_best_volume.append(dice_scores[b, idx])

                for s, subject in enumerate(data["subject"]):

                    name = "{}_t{}".format(subject, data["timestep"][s])
                    if name in info["coords"]["Subject and Timestep"]:
                        continue
                    else:
                        info["coords"]["Subject and Timestep"].append(name)

                    # regular evaluation
                    summary = data.copy()
                    summary["data"] = data["data"][s:s+1]
                    summary["seg"] = data["seg"][s:s+1]
                    if future:
                        summary["seg"] = summary["seg"][:, -1:]
                    else:
                        summary["seg"] = summary["seg"][:, -2:-1]
                    summary["prediction"] = prediction[s:s+1]

                    if c.test_save_output:
                        self.elog.save_numpy_data(summary["prediction"].numpy(), "test/{}_prediction.npy".format(name))

                    # regular results like in validation
                    current_score = self.validate_score(summary, 0)

                    # surprise / information gain
                    current_reference_kl = reference_kl[s:s+1].sum().item()

                    # ability to reconstruct groundtruth
                    current_reference_nll = nn.NLLLoss(reduction="sum")(reference_reconstruction[s:s+1], gt_segmentation[s:s+1]).item()
                    current_reference_dice = self.validate_score({"prediction": reference_reconstruction[s:s+1], "seg": summary["seg"]}, 0)
                    current_reference_dice = current_reference_dice[:, self.evaluator.metrics.index("Dice")]

                    # create arrays, repeating NLL and KL for each label
                    current_reference_kl = np.array([current_reference_kl, ] * len(c.evaluator_kwargs.label_values))
                    current_reference_nll = np.array([current_reference_nll, ] * len(c.evaluator_kwargs.label_values))
                    current_max_dice = np.array([max_dice[s], ] * len(c.evaluator_kwargs.label_values))
                    current_dice_best_volume = np.array([dice_for_best_volume[s], ] * len(c.evaluator_kwargs.label_values))

                    current_score_extended = np.stack([current_reference_kl,
                                                       current_reference_nll,
                                                       current_reference_dice,
                                                       current_max_dice,
                                                       current_dice_best_volume], 1)
                    current_score = np.concatenate([current_score, current_score_extended], 1)

                    scores.append(current_score)

                del input_, gt_segmentation

        return scores, info

    def test_future(self):

        info = self.validate_make_default_info()
        info["coords"]["Metric"] = info["coords"]["Metric"] + ["Future KL",
                                                               "Future Reconstruction NLL",
                                                               "Future Reconstruction Dice",
                                                               "Prior Maximum Dice",
                                                               "Prior Best Volume Dice"]

        # our regular generators only produce 1 timestep, so we create this here manually
        if self.config.test_on_val:
            test_data = self.data_val
        else:
            test_data = self.data_test

        generator = self.config.generator_val(
            test_data, self.config.batch_size_val, 3,
            number_of_threads_in_multithreaded=self.config.augmenter_val_kwargs.num_processes)
        transforms = []
        for t in sorted(self.config.transforms_val.keys()):
            if self.config.transforms_val[t]["active"]:
                cls = self.config.transforms_val[t]["type"]
                kwargs = self.config.transforms_val[t]["kwargs"]
                transforms.append(cls(**kwargs))

        augmenter = self.config.augmenter_val(generator,
                                              Compose(transforms),
                                              **self.config.augmenter_val_kwargs)
        test_scores, info = self.test_inner(augmenter, [], info, future=True)
        test_scores = np.array(test_scores)

        self.elog.save_numpy_data(test_scores, "test_future.npy")
        self.elog.save_dict(info, "test_future.json")


if __name__ == '__main__':

    parser = get_default_experiment_parser()
    parser.add_argument("-p", "--patch_size", type=int, nargs="+", default=112)
    parser.add_argument("-in", "--in_channels", type=int, default=4)
    parser.add_argument("-lt", "--latent_size", type=int, default=3)
    parser.add_argument("-lb", "--labels", type=int, nargs="+", default=[0, 1, 2, 3])
    args, _ = parser.parse_known_args()
    DEFAULTS, MODS = make_defaults(patch_size=args.patch_size, in_channels=args.in_channels, latent_size=args.latent_size, labels=args.labels)
    run_experiment(ProbabilisticUNetSegmentation,
                   DEFAULTS,
                   args,
                   mods=MODS,
                   explogger_kwargs=dict(folder_format="{experiment_name}_%Y%m%d-%H%M%S"),
                   globs=globals(),
                   resume_save_types=("model", "simple", "th_vars", "results"))

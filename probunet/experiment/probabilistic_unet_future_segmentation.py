from probunet.experiment.tumorgrowthexperiment import TumorGrowthExperiment

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



DESCRIPTION = "Future segmentation with a Probabilistic U-Net. .test() will give results for 'proposed' in paper."


def make_defaults(patch_size=112,
                  in_channels=4,
                  latent_size=3,
                  input_timesteps=2,
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
        input_timesteps=input_timesteps,
        seed=1,
        device="cuda",

        # Data
        split_val=3,
        split_test=4,
        data_module=data,
        data_dir=None,
        mmap_mode="r",
        npz=False,
        train_on_all=False,
        debug=0,  # 1 selects (10, 5, 5) patients, 2 a single batch
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
            "in_channels": in_channels * input_timesteps,
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
                "in_channels": in_channels * input_timesteps,
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
                "in_channels": in_channels * input_timesteps + len(labels),
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
            "posterior_kwargs": {"in_channels": in_channels * input_timesteps + 2}
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
            "posterior_kwargs": {"in_channels": in_channels * input_timesteps + 2}
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


class ProbabilisticUNetFutureSegmentation(TumorGrowthExperiment):

    def setup_loss_and_eval(self):

        c = self.config

        self.criterion_segmentation = c.criterion_segmentation(**c.criterion_segmentation_kwargs)
        if c.criterion_latent_init:
            self.criterion_latent = c.criterion_latent(**c.criterion_latent_kwargs)
        else:
            self.criterion_latent = c.criterion_latent
        self.evaluator = c.evaluator(**c.evaluator_kwargs)

    def train(self, epoch):

        c = self.config

        t0 = time.time()

        self.model.reset()
        self.model.train()
        self.optimizer.zero_grad()
        self.scheduler.step(epoch=epoch)

        data = next(self.augmenter_train)
        input_, gt_segmentation, data = self.process_data(data, epoch)
        prediction = self.model(input_, gt_segmentation, make_onehot=False)
        if not c.criterion_segmentation_seg_onehot:
            gt_segmentation = torch.argmax(gt_segmentation, 1, keepdim=False)

        loss_segmentation = self.criterion_segmentation(prediction, gt_segmentation).sum()
        loss_latent = self.criterion_latent(self.model.posterior, self.model.prior).sum()
        loss = c.criterion_segmentation_weight * loss_segmentation + c.criterion_latent_weight * loss_latent
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        training_time = time.time() - t0

        loss_segmentation = loss_segmentation.detach().cpu()
        loss_latent = loss_latent.detach().cpu()
        prediction = prediction.detach().cpu()

        data["loss_segmentation"] = loss_segmentation
        data["loss_latent"] = loss_latent
        data["prediction"] = prediction
        data["training_time"] = training_time

        self.train_log(data, epoch)

        self.model.reset()
        del input_, gt_segmentation, data

    def process_data(self, data, epoch):

        c = self.config

        if c.debug < 2 or epoch == 0:

            input_ = torch.from_numpy(data["data"][:, :-c.in_channels]).to(dtype=torch.float32, device=c.device)
            gt_segmentation = make_onehot_segmentation(torch.from_numpy(data["seg"][:, -1:]).to(dtype=torch.float32, device=c.device), c.labels)

            if c.debug == 2:
                self._memo_batch = (input_, gt_segmentation, data)

        else:

            return self._memo_batch

        return input_, gt_segmentation, data

    def train_log(self, summary, epoch):

        _backup = (epoch + 1) % self.config.backup_every == 0
        _show = (epoch + 1) % self.config.show_every == 0

        self.elog.show_text("{}/{}: {:.3f}".format(epoch, self.config.n_epochs, summary["training_time"]), name="Training Time")
        self.elog.show_text("{}/{}: {:.3f}".format(epoch, self.config.n_epochs, summary["loss_segmentation"].item()), name="Segmentation Loss")
        self.elog.show_text("{}/{}: {:.3f}".format(epoch, self.config.n_epochs, summary["loss_latent"].item()), name="Latent Loss")

        self.add_result(summary["loss_segmentation"].item(), "loss_loss_segmentation", epoch, "Loss", plot_result=_show, plot_running_mean=True)
        self.add_result(summary["loss_latent"].item(), "loss_latent", epoch, "Loss", plot_result=_show, plot_running_mean=True)

        self.make_images(summary,
                         epoch,
                         save=_backup,
                         show=_show,
                         validate=False)
        self.make_dist_plot(summary, epoch, save=_backup, show=_show, validate=False)

    def validate(self, epoch):

        c = self.config

        if (epoch+1) % c.validate_every == 0:

            with torch.no_grad():

                t0 = time.time()
                self.model.eval()

                info = self.validate_make_default_info()
                validation_scores = []

                for d, data in enumerate(self.augmenter_val):

                    # randomly decide whether to look at this batch, but with fallback so we don't validate nothing
                    if c.validate_subset not in (False, None, 1.):
                        rand_number = np.random.rand()
                        if rand_number < 1 - c.validate_subset:
                            if not (d * c.batch_size_val >= len(self.generator_val.possible_sets) - 1 and len(validation_scores) == 0):
                                continue

                    input_, gt_segmentation, data = self.process_data(data, epoch)
                    prediction = self.model(input_, gt_segmentation, make_onehot=False).cpu()

                    # iterate over batch and evaluate each instance separately
                    for s, subject in enumerate(data["subject"]):

                        name = "{}_{}".format(subject, data["timestep"][s])
                        if name in info["coords"]["Subject and Timestep"]:
                            continue
                        else:
                            info["coords"]["Subject and Timestep"].append(name)

                        summary = data.copy()
                        summary["data"] = data["data"][s:s+1]
                        summary["seg"] = data["seg"][s:s+1]
                        summary["prediction"] = prediction[s:s+1]

                        if c.val_save_images:
                            self.make_images(summary,
                                             epoch,
                                             label=name,
                                             save=True,
                                             show=False,
                                             validate=True)

                        if c.val_save_output:
                            epoch_str = c.epoch_str_template.format(epoch)
                            self.elog.save_numpy_data(summary["prediction"].numpy(), "validation/{}/{}_prediction.npy".format(epoch_str, name))

                        current_score = self.validate_score(summary, epoch)
                        validation_scores.append(current_score)

                    del input_, gt_segmentation

                validation_time = time.time() - t0
                validation_scores = np.array(validation_scores)

                data["validation_time"] = validation_time
                data["validation_scores"] = validation_scores
                data["validation_info"] = info

                self.validate_log(data, epoch)

                # draw a few different samples for the last data item
                if c.val_example_samples >= 2:
                    data = self.validate_draw_random_samples(self.augmenter_val)
                    self.make_images_samples(data,
                                             epoch,
                                             save=True,
                                             show=True,
                                             validate=True)

                self.model.reset()

    def validate_draw_random_samples(self, augmenter):

        c = self.config

        # in 2D, the last batch will have little tumor, so select a random one
        rand_position = np.random.randint(len(augmenter.generator.possible_sets))
        data = augmenter.generator.make_batch(rand_position)
        data = augmenter.transform(**data)
        input_, gt_segmentation, data = self.process_data(data, 0)

        # encode prior and generate activations, then draw actual samples
        mean_prediction = self.model(input_, gt_segmentation, make_onehot=False).cpu()
        samples = self.model.sample_prior(c.val_example_samples, "cpu")
        samples = torch.stack(samples, 0).transpose(0, 1).contiguous()
        v = list(samples.shape)[1:]
        v[0] = samples.shape[0] * samples.shape[1]
        samples = samples.view(*v)

        data["prediction"] = mean_prediction
        data["samples"] = samples

        return data

    def make_images_samples(self,
                            summary,
                            epoch,
                            label=None,
                            save=False, show=True, validate=False,
                            axis=0):

        if save is False and show is False:
            return

        patch_size = summary["samples"].shape[2:]
        dim = len(patch_size)

        predicted_seg = torch.argmax(summary["samples"], 1, keepdim=True).float()

        if dim == 2:

            segs = predicted_seg

        elif dim == 3:

            slc = [slice(None), ] * 5
            slc[axis+2] = patch_size[axis] // 2
            segs = predicted_seg[tuple(slc)]

        else:

            raise ValueError("Data must have 2 or 3 spatial dimensions, but found {}".format(dim))

        name = "samples"
        if label is not None:
            name = label + "_" + name
        if validate:
            name = "val/" + name
        image_args = {
            "normalize": True,
            "range": (min(self.config.labels), max(self.config.labels)),
            "nrow": self.config.val_example_samples,
            "pad_value": 1
        }

        if show and self.vlog is not None:
            self.vlog.show_image_grid(segs, name, image_args=image_args)
        if save and self.elog is not None:
            name = self.config.epoch_str_template.format(epoch) + "_" + name
            self.elog.show_image_grid(segs, name, image_args=image_args)

    def make_images(self,
                    summary,
                    epoch,
                    label=None,
                    save=False, show=True, validate=False):

        if save is False and show is False:
            return

        input_data = torch.from_numpy(summary["data"]).float()
        predicted_seg = torch.argmax(summary["prediction"], 1, keepdim=True).float().cpu()
        reference_seg = torch.from_numpy(summary["seg"]).float()

        if reference_seg.dim() == 4:
            reference_seg = reference_seg.unsqueeze(1)

        batch_size = input_data.shape[0]
        patch_size = tuple(input_data.shape[2:])
        dim = len(patch_size)
        _range = (0., float(self.config.out_channels - 1))

        if dim == 3:

            for i in range(batch_size):

                image_list = []
                image_list.append(input_data[i:i+1, :, patch_size[0] // 2, :, :].transpose(0, 1))
                image_list.append(input_data[i:i+1, :, :, patch_size[1] // 2, :].transpose(0, 1))
                image_list.append(input_data[i:i+1, :, :, :, patch_size[2] // 2].transpose(0, 1))
                image_list = torch.cat(image_list, 0)

                name = "input_data_{}".format(i)
                if label is not None:
                    name = label + "_" + name
                if validate:
                    name = "val/" + name

                if show and hasattr(self, "vlog") and self.vlog is not None:
                    self.vlog.show_image_grid(image_list, name,
                                              image_args={"normalize": True,
                                                          "scale_each": True,
                                                          "nrow": input_data.shape[1],
                                                          "pad_value": 1})
                if save:
                    name = self.config.epoch_str_template.format(epoch) + "_" + name
                    self.elog.show_image_grid(image_list, name,
                                              image_args={"normalize": True,
                                                          "scale_each": True,
                                                          "nrow": input_data.shape[1],
                                                          "pad_value": 1})

                seg_list = []
                for j in range(reference_seg.shape[1]):
                    seg_list.append(reference_seg[i:i+1, j:j+1, patch_size[0] // 2, :, :].float().transpose(0, 1))
                for j in range(predicted_seg.shape[1]):
                    seg_list.append(predicted_seg[i:i+1, j:j+1, patch_size[0] // 2, :, :].float().transpose(0, 1))
                for j in range(reference_seg.shape[1]):
                    seg_list.append(reference_seg[i:i+1, j:j+1, :, patch_size[1] // 2, :].float().transpose(0, 1))
                for j in range(predicted_seg.shape[1]):
                    seg_list.append(predicted_seg[i:i+1, j:j+1, :, patch_size[1] // 2, :].float().transpose(0, 1))
                for j in range(reference_seg.shape[1]):
                    seg_list.append(reference_seg[i:i+1, j:j+1, :, :, patch_size[2] // 2].float().transpose(0, 1))
                for j in range(predicted_seg.shape[1]):
                    seg_list.append(predicted_seg[i:i+1, j:j+1, :, :, patch_size[2] // 2].float().transpose(0, 1))
                seg_list = torch.cat(seg_list, 0)

                name = "segmentation_{}".format(i)
                if label is not None:
                    name = label + "_" + name
                if validate:
                    name = "val/" + name

                if show and hasattr(self, "vlog") and self.vlog is not None:
                    self.vlog.show_image_grid(seg_list, name,
                                              image_args={"normalize": True,
                                                          "range": _range,
                                                          "nrow": reference_seg.shape[1] + predicted_seg.shape[1],
                                                          "pad_value": 1})
                if save and hasattr(self, "elog"):
                    name = self.config.epoch_str_template.format(epoch) + "_" + name
                    self.elog.show_image_grid(seg_list, name,
                                              image_args={"normalize": True,
                                                          "range": _range,
                                                          "nrow": reference_seg.shape[1] + predicted_seg.shape[1],
                                                          "pad_value": 1})

        else:

            for i in range(input_data.shape[1]):

                image_list = input_data[:, i:i+1]

                name = "input_data_{}".format(i)
                if label is not None:
                    name = label + "_" + name
                if validate:
                    name = "val/" + name

                if show and hasattr(self, "vlog") and self.vlog is not None:
                    self.vlog.show_image_grid(image_list, name,
                                              image_args={"normalize": True,
                                                          "scale_each": True,
                                                          "nrow": int(np.sqrt(image_list.shape[0])),
                                                          "pad_value": 1})
                if save and hasattr(self, "elog"):
                    name = self.config.epoch_str_template.format(epoch) + "_" + name
                    self.elog.show_image_grid(image_list, name,
                                              image_args={"normalize": True,
                                                          "scale_each": True,
                                                          "nrow": int(np.sqrt(image_list.shape[0])),
                                                          "pad_value": 1})

            seg_list = []
            for i in range(reference_seg.shape[0]):
                seg_list.append(reference_seg[i:i+1, :].float().transpose(0, 1))
                seg_list.append(predicted_seg[i:i+1, :].float().transpose(0, 1))
            seg_list = torch.cat(seg_list, 0)

            nrow_min = reference_seg.shape[1] + predicted_seg[1]
            nrow = (int(np.sqrt(seg_list.shape[0])) // nrow_min + 1) * nrow_min

            name = "segmentation"
            if label is not None:
                name = label + "_" + name
            if validate:
                name = "val/" + name

            if show and hasattr(self, "vlog") and self.vlog is not None:
                self.vlog.show_image_grid(seg_list, name,
                                          image_args={"normalize": True,
                                                      "range": _range,
                                                      "nrow": nrow,
                                                      "pad_value": 1})
            if save and hasattr(self, "elog"):
                name = self.config.epoch_str_template.format(epoch) + "_" + name
                self.elog.show_image_grid(seg_list, name,
                                          image_args={"normalize": True,
                                                      "range": _range,
                                                      "nrow": nrow,
                                                      "pad_value": 1})

    def make_dist_plot(self, summary, epoch, label=None, save=False, show=True, validate=False):

        c = self.config

        def density_fn(x, loc, scale):
            return 1 / np.sqrt(2*np.pi*scale*scale) * np.exp(-(x-loc)*(x-loc)/(2*scale*scale))

        prior_loc = self.model.prior.loc.detach().cpu().numpy()
        prior_scale = self.model.prior.scale.detach().cpu().numpy()
        posterior_loc = self.model.posterior.loc.detach().cpu().numpy()
        posterior_scale = self.model.posterior.scale.detach().cpu().numpy()

        prior_loc = prior_loc.reshape(prior_loc.shape[0], -1)
        prior_scale = prior_scale.reshape(prior_scale.shape[0], -1)
        posterior_loc = prior_loc.reshape(posterior_loc.shape[0], -1)
        posterior_scale = prior_scale.reshape(posterior_scale.shape[0], -1)

        x_values = np.linspace(*c.latent_plot_range, 500)

        for batch in range(prior_loc.shape[0]):

            name = "latent_{}".format(batch)
            if label is not None:
                name = label + "_" + name
            if validate:
                name = "val/" + name

            data = []
            legend_names = []
            for latent_dim in range(prior_loc.shape[1]):
                data.append(density_fn(x_values, prior_loc[batch, latent_dim], prior_scale[batch, latent_dim]))
                legend_names.append("prior_{}".format(latent_dim))
                data.append(density_fn(x_values, posterior_loc[batch, latent_dim], posterior_scale[batch, latent_dim]))
                legend_names.append("posterior_{}".format(latent_dim))
            data = np.array(data).T
            data_x = np.repeat(x_values[np.newaxis, :], prior_loc.shape[1] * 2, axis=0).T

            if show and self.vlog is not None:
                self.vlog.show_lineplot(data, data_x, name=name, opts={"legend": legend_names})

            if save and self.elog is not None:
                name = self.config.epoch_str_template.format(epoch) + "_" + name
                self.elog.show_lineplot(data, data_x, name=name, opts={"legend": legend_names})

    def test(self):

        info = self.validate_make_default_info()
        info["coords"]["Metric"] = info["coords"]["Metric"] + ["Reference KL",
                                                               "Reference Reconstruction NLL",
                                                               "Reference Reconstruction Dice",
                                                               "Present KL",
                                                               "Present Reconstruction NLL",
                                                               "Present Reconstruction Dice",
                                                               "Present Future Dice",
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

    def test_inner(self, augmenter, scores, info):

        c = self.config

        with torch.no_grad():

            self.model.eval()

            for data in augmenter:

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

                present_segmentation = data["seg"][:, -2:-1]
                present_segmentation = torch.from_numpy(present_segmentation)
                present_segmentation = present_segmentation.to(dtype=torch.float32, device=c.device)
                present_segmentation = make_onehot_segmentation(present_segmentation, c.labels)

                self.model.encode_posterior(input_, present_segmentation, make_onehot=False)
                present_reconstruction = self.model.reconstruct(out_device="cpu")
                present_segmentation = torch.argmax(present_segmentation.cpu(), 1, keepdim=False)
                present_kl = distributions.kl_divergence(self.model.posterior, self.model.prior)

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
                    summary["prediction"] = prediction[s:s+1]

                    if c.test_save_output:
                        self.elog.save_numpy_data(summary["prediction"].numpy(), "test/{}_prediction.npy".format(name))

                    # regular results like in validation
                    current_score = self.validate_score(summary, 0)

                    # surprise / information gain
                    current_reference_kl = reference_kl[s:s+1].sum().item()
                    current_present_kl = present_kl[s:s+1].sum().item()

                    # ability to reconstruct groundtruth
                    current_reference_nll = nn.NLLLoss(reduction="sum")(reference_reconstruction[s:s+1], gt_segmentation[s:s+1]).item()
                    current_reference_dice = self.validate_score({"prediction": reference_reconstruction[s:s+1], "seg": summary["seg"]}, 0)
                    current_reference_dice = current_reference_dice[:, self.evaluator.metrics.index("Dice")]
                    current_present_nll = nn.NLLLoss(reduction="sum")(present_reconstruction[s:s+1], present_segmentation[s:s+1]).item()
                    current_present_dice = self.validate_score({"prediction": present_reconstruction[s:s+1], "seg": summary["seg"][:, -2:-1]}, 0)
                    current_present_dice = current_present_dice[:, self.evaluator.metrics.index("Dice")]

                    # Overlap of present and future
                    current_present_future_dice = multi_label_metric(present_segmentation[s].numpy(),
                                                                     gt_segmentation[s].numpy(),
                                                                     labels=c.evaluator_kwargs.label_values,
                                                                     metric=dice)
                    current_present_future_dice = np.array(current_present_future_dice)

                    # create arrays, repeating NLL and KL for each label
                    current_reference_kl = np.array([current_reference_kl, ] * len(c.evaluator_kwargs.label_values))
                    current_reference_nll = np.array([current_reference_nll, ] * len(c.evaluator_kwargs.label_values))
                    current_present_kl = np.array([current_present_kl, ] * len(c.evaluator_kwargs.label_values))
                    current_present_nll = np.array([current_present_nll, ] * len(c.evaluator_kwargs.label_values))
                    current_max_dice = np.array([max_dice[s], ] * len(c.evaluator_kwargs.label_values))
                    current_dice_best_volume = np.array([dice_for_best_volume[s], ] * len(c.evaluator_kwargs.label_values))

                    current_score_extended = np.stack([current_reference_kl,
                                                       current_reference_nll,
                                                       current_reference_dice,
                                                       current_present_kl,
                                                       current_present_nll,
                                                       current_present_dice,
                                                       current_present_future_dice,
                                                       current_max_dice,
                                                       current_dice_best_volume], 1)
                    current_score = np.concatenate([current_score, current_score_extended], 1)

                    scores.append(current_score)

                del input_, gt_segmentation

        return scores, info


if __name__ == '__main__':

    parser = get_default_experiment_parser()
    parser.add_argument("-p", "--patch_size", type=int, nargs="+", default=112)
    parser.add_argument("-in", "--in_channels", type=int, default=4)
    parser.add_argument("-lt", "--latent_size", type=int, default=3)
    parser.add_argument("-lb", "--labels", type=int, nargs="+", default=[0, 1, 2, 3])
    args, _ = parser.parse_known_args()
    DEFAULTS, MODS = make_defaults(patch_size=args.patch_size, in_channels=args.in_channels, latent_size=args.latent_size, labels=args.labels)
    run_experiment(ProbabilisticUNetFutureSegmentation,
                   DEFAULTS,
                   args,
                   mods=MODS,
                   explogger_kwargs=dict(folder_format="{experiment_name}_%Y%m%d-%H%M%S"),
                   globs=globals(),
                   resume_save_types=("model", "simple", "th_vars", "results"))

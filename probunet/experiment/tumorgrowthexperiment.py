import matplotlib
matplotlib.use("agg")
import torch.backends.cudnn as cudnn
import os
if "CUDNN_DETERMINISTIC" in os.environ and os.environ["CUDNN_DETERMINISTIC"] not in (0, False, "false", "FALSE", "False"):
    cudnn.benchmark = False
    cudnn.deterministic = True
else:
    cudnn.benchmark = True
    cudnn.deterministic = False
import atexit
import numpy as np
import torch
from trixi.util import ResultLogDict
from trixi.experiment import PytorchExperiment
from trixi.logger import TelegramMessageLogger
from batchgenerators.transforms import Compose

from probunet.util import check_attributes, set_seeds



class TumorGrowthExperiment(PytorchExperiment):

    def __init__(self, *args, use_telegram=False, **kwargs):

        super().__init__(*args, **kwargs)
        self.use_telegram = use_telegram

    def setup(self):

        self.cleanup()

        set_seeds(self.config.seed, "cuda" in self.config.device)

        if self.use_telegram:
            self.setup_telegram()
        self.setup_subjects()
        self.setup_data()
        self.setup_augmentation()
        self.setup_model()
        self.setup_loss_and_eval()

        self.config.epoch_str_template = "{:0" + str(len(str(self.config.n_epochs))) + "d}"
        self.clog.show_text(self.model.__repr__(), "Model")

    def cleanup(self):

        # close everything that could have open files
        self.augmenter_train = None
        self.augmenter_val = None
        self.augmenter_test = None
        self.generator_train = None
        self.generator_val = None
        self.generator_test = None
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.results.close()

    def setup_telegram(self):

        self.tlogger = TelegramMessageLogger(token="123:abc", chat_id="123")
        info_str = "{} started\n".format(self.exp_name)
        info_str += "-" * 20 + "\n"
        info_str += self.config.description
        info_str += "-" * 20 + "\n"
        self.tlogger.show_text(info_str)

        def exit_message():
            info_str = "{} ended\n".format(self.exp_name)
            info_str += "-" * 20 + "\n"
            info_str += "Status {}, {}/{} epochs\n".format(self._exp_state, self._epoch_idx + 1, self.n_epochs)
            info_str += "-" * 20 + "\n"
            info_str += self.config.description
            info_str += "-" * 20 + "\n"
            self.tlogger.show_text(info_str)

        atexit.register(exit_message)

    def setup_subjects(self):

        c = self.config

        if c.data_dir is not None:
            c.data_module.data_dir = c.data_dir

        if not check_attributes(c, ["subjects_train", "subjects_val", "subjects_test"]):
            split = c.data_module.split()
            c.subjects_val = split[c.split_val]
            c.subjects_test = split[c.split_test]
            c.subjects_train = []
            for i in range(len(split)):
                if i not in (c.split_val, c.split_test) or c.train_on_all:
                    c.subjects_train = c.subjects_train + split[i]
            c.subjects_train = sorted(c.subjects_train)

        if c.debug:
            c.subjects_train = c.subjects_train[:10]
            c.subjects_val = c.subjects_val[:5]
            c.subjects_test = c.subjects_test[:5]

    def setup_data(self):

        c = self.config

        self.data_train = c.data_module.load(c.mmap_mode, subjects=c.subjects_train, npz=c.npz)
        self.data_val = c.data_module.load(c.mmap_mode, subjects=c.subjects_val, npz=c.npz)
        self.data_test = c.data_module.load(c.mmap_mode, subjects=c.subjects_test, npz=c.npz)

        self.generator_train = c.generator_train(
            self.data_train, c.batch_size, c.input_timesteps + 1,
            number_of_threads_in_multithreaded=c.augmenter_train_kwargs.num_processes)
        self.generator_val = c.generator_val(
            self.data_val, c.batch_size_val, c.input_timesteps + 1,
            number_of_threads_in_multithreaded=c.augmenter_val_kwargs.num_processes)
        self.generator_test = c.generator_val(
            self.data_test, c.batch_size_val, c.input_timesteps + 1,
            number_of_threads_in_multithreaded=c.augmenter_val_kwargs.num_processes)

    def setup_augmentation(self):

        c = self.config

        transforms_train = []
        for t in sorted(c.transforms_train.keys()):
            if c.transforms_train[t]["active"]:
                cls = c.transforms_train[t]["type"]
                kwargs = c.transforms_train[t]["kwargs"]
                transforms_train.append(cls(**kwargs))
        self.augmenter_train = c.augmenter_train(self.generator_train,
                                                 Compose(transforms_train),
                                                 **c.augmenter_train_kwargs)

        transforms_val = []
        for t in sorted(c.transforms_val.keys()):
            if c.transforms_val[t]["active"]:
                cls = c.transforms_val[t]["type"]
                kwargs = c.transforms_val[t]["kwargs"]
                transforms_val.append(cls(**kwargs))
        self.augmenter_val = c.augmenter_val(self.generator_val,
                                             Compose(transforms_val),
                                             **c.augmenter_val_kwargs)
        self.augmenter_test = c.augmenter_val(self.generator_test,
                                              Compose(transforms_val),
                                              **c.augmenter_val_kwargs)

    def setup_model(self):

        c = self.config

        # model
        self.model = c.model(**c.model_kwargs)
        if c.model_init_weights_args is not None:
            if isinstance(c.model_init_weights_args, dict):
                for key, val in c.model_init_weights_args.items():
                    try:
                        self.model._modules[key].init_weights(*val)
                    except Exception as e:
                        print("Tried to initialize {} with {}, but got the following error:".format(key, val))
                        print(repr(e))
            elif isinstance(c.model_init_weights_args, (list, tuple)):
                self.model.init_weights(*c.model_init_weights_args)
            else:
                raise TypeError("model_init_weights_args must be dict, list or tuple, but found {}.".format(type(c.model_init_weights_args)))
        if c.model_init_bias_args is not None:
            if isinstance(c.model_init_bias_args, dict):
                for key, val in c.model_init_bias_args.items():
                    try:
                        self.model._modules[key].init_bias(*val)
                    except Exception as e:
                        print("Tried to initialize {} with {}, but got the following error:".format(key, val))
                        print(repr(e))
            elif isinstance(c.model_init_bias_args, (list, tuple)):
                self.model.init_bias(*c.model_init_bias_args)
            else:
                raise TypeError("model_init_bias_args must be dict, list or tuple, but found {}.".format(type(c.model_init_bias_args)))

        # optimizer
        self.optimizer = c.optimizer(self.model.parameters(), **c.optimizer_kwargs)
        self.scheduler = c.scheduler(self.optimizer, **c.scheduler_kwargs)

    def setup_loss_and_eval(self):

        raise NotImplementedError

    def _setup_internal(self):

        super()._setup_internal()
        self.results.close()
        mode = "w" if os.stat(os.path.join(self.elog.result_dir, "results-log.json")).st_size <= 4 else "a"
        self.results = ResultLogDict("results-log.json", base_dir=self.elog.result_dir, mode=mode, running_mean_length=self.config.show_every)
        self.elog.save_config(self.config, "config")  # default PytorchExperiment only saves self._config_raw

    def prepare(self):

        for name, model in self.get_pytorch_modules().items():
            model.to(self.config.device)

    def validate_make_default_info(self):

        info = {}
        info["dims"] = ["Subject and Timestep", "Label", "Metric"]
        info["coords"] = {"Subject and Timestep": []}
        info["coords"]["Label"] = [self.evaluator.label_names[l] for l in self.evaluator.label_values]
        info["coords"]["Metric"] = self.evaluator.metrics

        return info

    def validate_score(self, summary, epoch):

        segmentation = torch.argmax(summary["prediction"], 1, keepdim=False)[0, ...].cpu().numpy()
        reference = summary["seg"][0, -1, ...]
        if torch.is_tensor(reference):
            reference = reference.cpu().numpy()

        self.evaluator.set_reference(reference)
        self.evaluator.set_test(segmentation)
        self.evaluator.evaluate()

        return self.evaluator.to_array()

    def validate_log(self, summary, epoch):

        epoch_str = self.config.epoch_str_template.format(epoch)

        if self.evaluator.nan_for_nonexisting:
            validation_scores_mean = np.nanmean(summary["validation_scores"], 0)
        else:
            validation_scores_mean = np.mean(summary["validation_scores"], 0)

        self.elog.save_numpy_data(summary["validation_scores"], "validation/{}.npy".format(epoch_str))
        self.elog.save_dict(summary["validation_info"], "validation/{}.json".format(epoch_str))
        self.elog.show_text("{}/{}: {}".format(epoch, self.config.n_epochs, summary["validation_time"]), name="Validation Time")

        for l, label in enumerate(self.evaluator.label_values):
            label_name = self.evaluator.label_names[label].lower().replace(" ", "_")
            for metric in self.config.validate_metrics:
                m = self.evaluator.metrics.index(metric)
                output_name = "validation_{}_{}".format(label_name, metric)
                self.add_result(float(validation_scores_mean[l][m]), output_name, epoch, "Scores")

    def _end_epoch_internal(self, epoch):

        self.save_results()
        if (epoch+1) % self.config.backup_every == 0:
            self.save_temp_checkpoint()

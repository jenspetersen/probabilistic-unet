# Probabilistic U-Net & Deep Probabilistic Modeling of Glioma Growth

This repository contains two things:

1. A generic PyTorch implementation of the [Probabilistic U-Net](https://arxiv.org/abs/1806.05034) that somewhat mirrors the signature of the [official implementation](https://github.com/SimonKohl/probabilistic_unet) in Tensorflow.
2. Code (but not data unfortunately) to reproduce the results of our paper "Deep Probabilistic Modeling of Glioma Growth" that was accepted at MICCAI 2019.


# Installation

    git clone https://github.com/jenspetersen/probabilistic-unet.git
    pip install -e probabilistic-unet

will make a package called `probunet` available in your current Python environment.


# Usage

## Probabilistic U-Net

Our generic implementation of the Probabilistic U-Net is hopefully relatively straightforward to use:

    from probunet.model import ProbabilisticSegmentationNet

As you will have noticed, it also has a pretty generic name. That's because it doesn't actually require a U-Net, but can work with arbitrary segmentation architectures, as long as they:

1. Allow injection of samples in some way.
2. Provide the same API as our InjectionUNet (look at the calls to self.task_net to see requirements.)

Our encoder implementation also accepts injections, this is currently not used. Make sure to read

## Experiments

There are two (very similar) experiments in this repository:

1. `probabilistic_unet_segmentation.py` will produce results for upper and lower bound in the paper.
2. `probabilistic_unet_future_segmentation.py` will produce results for the proposed approach in the paper.

Both subclass `PytorchExperiment` from trixi for logging etc. Please check the [trixi documentation](https://trixi.readthedocs.io/en/develop/) for an intro to how those work, the most important feature is probably that all entries in the Configs are automatically exposed to the command line (for example, you might not want to output images every 10 iterations, so you could run with `--show_every 500`). We also use a concept called mods, meaning a set of modifications to the Config. These are applied via the `-m` flag. The results in the paper use the FULLNORM mod, for example.

At the moment, you won't be able to reproduce our results, because we can't publish the data (yet). But if you have your own data of longitudinal glioma growth, there's already a data loader for your convenience. Do the following:

1. Put each patient in a 5D numpy array (time, channel, x, y, z), where channels should be (T1, T1ce, T2, FLAIR). The data should be skull-stripped and z-score normalized (subtract mean, divide by std).
2. Put all patients in a folder, with file names identifier.npy
3. Create a file multi_shapes.json in the same folder that contains a dictionary {identifier: shape-tuple}. We need this to have faster access to all shapes.
4. (Optional) Create files identifier_crop.npy in the same folder that contain your data maximally cropped. The the `load()` method has an option `crop_data`. If you don't have cropped files, this needs to be False.

There is also a function to automatically do a split for you and the actual data loading during the experiment is done with the Random/LinearBatchGenerator, which use [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators). Overall, the experiments are very generic and will allow you to configure almost anything. Assuming you have your own data and you want to run the experiments we ran for the paper, you can do:

    python probabilistic_unet_segmentation.py RESULT_DIR --data_dir DATA_DIR -ad -m FULLNORM --split_val 0 --split_test 0 --test

where `-ad` automatically generates a description by checking what's different with respect to the default configuration. Obviously run the same for all your splits and also run the future segmentation. `--test` should automatically run the tests after training has finished, but we actually did this separately:

    python probabilistic_unet_segmentation.py RESULT_DIR --data_dir DATA_DIR --test --resume PATH_TO_EXISTING_EXPERIMENT

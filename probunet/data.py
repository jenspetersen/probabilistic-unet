import os
import json
import numpy as np
from batchgenerators.dataloading import SlimDataLoaderBase

# data_dir = "INSERT YOUR RELATIVE DATA DIRECTORY HERE"
# file_dir = os.path.dirname(os.path.abspath(__file__))
# data_dir = os.path.join(file_dir, data_dir)
data_dir = None


class NpzLazyDict(dict):

    def __getitem__(self, key):
        return np.load(super().__getitem__(key))[key]


def all_subjects():
    all_ = os.listdir(data_dir)
    subjects = sorted(filter(lambda x: x.endswith((".npy", ".npz")), all_))
    subjects = sorted(list(set(map(lambda x: x.replace("_crop", "").replace(".npy", "").replace(".npz", ""), subjects))))
    return subjects


def split(N=5, seed=1):

    all_ = all_subjects()
    r = np.random.RandomState(seed)
    r.shuffle(all_)
    num = len(all_) // N
    splits = []
    for i in range(N):
        if i < (N-1):
            splits.append(sorted(all_[i*num:(i+1)*num]))
        else:
            splits.append(sorted(all_[i*num:]))

    return splits


def load(mmap_mode=None,
         subjects="all",
         dtype=np.float32,
         crop_data=True,
         npz=False):
    """Load data. Note that the data need to be already registered,
    skull-stripped, and mean/std normalized.

    :param mmap_mode: Should be either None (for in-memory array) or
                      one of the numpy.memmap read-only modes
    :param subjects: 'all' or iterable of subject identifiers
    :param dtype: Data type for scan data
    :param crop_data: Automatically append "_crop" to file names
    :return: A dict of subject identifiers and np.arrays (time, channel, ...)
    """

    if subjects == "all": subjects = all_subjects()
    assert hasattr(subjects, "__iter__")
    assert type(dtype) == type

    if npz:
        data = NpzLazyDict()
    else:
        data = dict()

    for subject in subjects:
        if npz:
            fname = "{}.npz".format(subject)
            data[subject] = os.path.join(data_dir, fname)
        elif crop_data:
            data[subject] = np.load(
                os.path.join(data_dir, "{}_crop.npy".format(subject)),
                mmap_mode=mmap_mode).astype(dtype, copy=False)
        else:
            data[subject] = np.load(
                os.path.join(data_dir, "{}.npy".format(subject)),
                mmap_mode=mmap_mode).astype(dtype, copy=False)

    return data



class LinearBatchGenerator(SlimDataLoaderBase):
    """Time steps will be stacked along channel axis!

    :param data: {subject: np.array(time, channel, x, y, z)}
    :param batch_size: See parent
    :param time_size: Extract this many consecutive time steps
    """

    def __init__(self,
                 data,
                 batch_size,
                 time_size,
                 channels=(0, 1, 2, 3),
                 dtype_seg=np.int64,
                 use_default_shapes=True,
                 **kwargs):

        super(LinearBatchGenerator, self).__init__(data, batch_size, **kwargs)
        self.time_size = time_size
        self.dtype_seg = dtype_seg
        self.channels = channels
        self.use_default_shapes = use_default_shapes

        self.current_position = 0
        self.was_initialized = False
        if self.number_of_threads_in_multithreaded is None:
            self.number_of_threads_in_multithreaded = 1
        self.data_order = np.arange(len(self.possible_sets))

    @property
    def possible_sets(self):

        try:
            return self._possible_sets
        except AttributeError:
            sets = []
            if self.use_default_shapes:
                shapes = json.load(open(os.path.join(data_dir, "multi_shapes.json"), "r"))
            else:
                shapes = {key: val.shape for key, val in self._data.items()}
            for subject in sorted(self._data.keys()):
                current_timesteps = shapes[subject][0]
                if current_timesteps < self.time_size:
                    continue
                for i in range(current_timesteps - self.time_size + 1):
                    sets.append((subject, slice(i, i + self.time_size)))
            self._possible_sets = sets
            return sets

    def reset(self):

        self.current_position = self.thread_id * self.batch_size
        self.was_initialized = True

    def __len__(self):

        return int(np.ceil(len(self.possible_sets) / float(self.batch_size)))

    def generate_train_batch(self):

        if not self.was_initialized:
            self.reset()
        if self.current_position >= len(self.possible_sets):
            self.reset()
            raise StopIteration
        batch = self.make_batch(self.current_position)
        self.current_position += self.number_of_threads_in_multithreaded * self.batch_size
        return batch

    def make_batch(self, idx):

        batch_data = []
        batch_seg = []
        batch_subjects = []
        batch_timesteps = []

        while len(batch_data) < self.batch_size:

            idx = idx % len(self.data_order)

            subject, slc = self.possible_sets[self.data_order[idx]]
            current_data = self._data[subject][slc][:, self.channels, ...]

            current_data = np.concatenate(current_data)
            current_seg = self._data[subject][slc][:, 5:, ...]
            current_seg = np.concatenate(current_seg)

            batch_data.append(current_data)
            batch_seg.append(current_seg)
            batch_subjects.append(subject)
            batch_timesteps.append(slc.start)

            idx += 1

        return {"data": np.array(batch_data),
                "seg": np.array(batch_seg, dtype=self.dtype_seg),
                "subject": np.array(batch_subjects),
                "timestep": np.array(batch_timesteps)}


class RandomOrderBatchGenerator(LinearBatchGenerator):
    """Time steps will be stacked along channel axis!

    :param data: {subject: np.array(time, channel, x, y, z)}
    :param batch_size: See parent
    :param time_size: Extract this many consecutive time steps
    """

    def __init__(self, *args, infinite=True, **kwargs):

        super(RandomOrderBatchGenerator, self).__init__(*args, **kwargs)

        self.infinite = infinite
        self.num_restarted = 0

    def reset(self):

        super(RandomOrderBatchGenerator, self).reset()
        rs = np.random.RandomState(self.num_restarted)
        rs.shuffle(self.data_order)
        self.num_restarted = self.num_restarted + 1

    def generate_train_batch(self):

        if not self.was_initialized:
            self.reset()
        if self.current_position >= len(self.possible_sets):
            self.reset()
            if not self.infinite:
                raise StopIteration
        batch = self.make_batch(self.current_position)
        self.current_position += self.number_of_threads_in_multithreaded * self.batch_size
        return batch


class RandomBatchGenerator(RandomOrderBatchGenerator):
    pass
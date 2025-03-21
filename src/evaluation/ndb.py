from dataclasses import dataclass
from collections import defaultdict
from math import sqrt

import tensorflow as tf
import numpy as np
from scipy.stats import norm
from ddsp import spectral_ops
from sklearn_extra.cluster import KMedoids

from ..data.fs_utils import git_root

NDB_STATS_DIR = f"{git_root()}/ndb_stats"
_AUTOTUNE = tf.data.experimental.AUTOTUNE


@dataclass
class DatasetNDBStats:
    bin_centers: list


def l2_distance(s1, s2):
    return tf.reduce_sum(tf.square(s1 - s2))


def flatten_subsample_tf_dataset(ds, samples_fraction=0.5, dims_fraction=0.2):
    """
    note that `ds` should not be batched
    """

    ex_shape = next(iter(ds)).shape
    assert len(ex_shape) == 2  # log magnitude spectrograms
    n_dims = np.prod(ex_shape)
    keepdims = np.random.choice(n_dims, size=int(n_dims * dims_fraction))

    ds = ds.enumerate()
    ds = ds.filter(lambda idx, ex: tf.random.uniform(()) < tf.constant(samples_fraction))
    ds = ds.map(lambda idx, ex: (idx, tf.reshape(ex, [-1])))
    subsampled_ds = np.stack([ex.numpy()[keepdims] for _, ex in iter(ds)])
    labels = np.stack([idx for idx, _ in iter(ds)])
    return labels, subsampled_ds


def map_logmag(ds, fft_size=2048):
    """
    maps examples with an 'audio' key containing audio samples
    to their log magnitude spectrograms
    """

    if tf.is_tensor(ds):
        return spectral_ops.compute_logmag(ds, size=fft_size)
    else:  # should be a dataset
        def map_f(ex):
            return spectral_ops.compute_logmag(ex['audio'], size=fft_size)
        return ds.map(map_f, num_parallel_calls=_AUTOTUNE)


def get_voronoi_centers(ds, k=50):
    original_labels, subsampled_ds = flatten_subsample_tf_dataset(ds)
    k_means = KMedoids(n_clusters=k, max_iter=1000).fit(subsampled_ds)
    centers = k_means.cluster_centers_

    # retrieve original center sample indices
    is_center_mask = [np.isclose(subsampled_ds, c).all(axis=1) for c in centers]
    assert all([mask.sum() == 1] for mask in is_center_mask)

    center_indices = [tf.gather(original_labels, indices=tf.squeeze(tf.where(mask))) for mask in is_center_mask]
    return center_indices


def get_center_samples(ds, k=50):
    cluster_centers = get_voronoi_centers(ds, k=k)

    center_samples = list(
        ds
        .enumerate()
        .filter(lambda idx, ex: tf.reduce_any(tf.equal(idx, tf.convert_to_tensor(cluster_centers))))
        .map(lambda idx, ex: ex)
    )

    return center_samples


def get_closest_center(sample, center_samples):
    distances = [l2_distance(sample, center) for center in center_samples]
    return tf.argmin(distances)


def assign_samples_to_bins(ds, center_samples):
    if tf.is_tensor(ds):
        return [get_closest_center(ex, center_samples) for ex in tf.unstack(ds)]
    else:
        return list(ds.map(lambda ex: get_closest_center(ex, center_samples)))


def get_cluster_counts(cluster_assignment, k):
    cluster_counts = {i: 0 for i in range(k)}
    for c, count in zip(*np.unique(cluster_assignment, return_counts=True)):
        cluster_counts[c] = count

    return cluster_counts


# 1. map train ds to logmag
# 2. get center samples
# 3. assign samples to bins -> get cluster counts/proportions
# 4. sample model
# 5. map logmag
# 6. assign samples to bins -> get cluster counts/proportions
def two_sample_test(c1, c2, n1, n2):
    """
    Performs a two-sample test for bernoulli variables

    For reference see https://arxiv.org/pdf/1805.12462.pdf
    chapter 2 - A New Evaluation Method
    """

    p = (c1 + c2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1/n1 + 1/n2)) + 1e-6
    p1 = c1/n1
    p2 = c2/n2
    z = (p1 - p2) / se

    pval = 2 * (1 - norm.cdf(np.abs(z)))
    return pval

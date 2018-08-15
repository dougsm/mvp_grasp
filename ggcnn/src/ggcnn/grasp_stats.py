import numpy as np
from .timeit import TimeIt

# def update_batch(data, ids, count, mean, var):
#
#     if ids.ndim == 2:
#         ids = tuple(ids.T)
#
#     count_temp = np.zeros_like(count)
#     mean_temp = np.zeros_like(mean)
#     var_temp = np.zeros_like(var)
#     new_mean = np.zeros_like(count_temp[ids])
#
#     # Numba doesn't like np.add.at
#     np.add.at(count_temp, ids, 1)
#     np.add.at(mean_temp, ids, data)
#     mean_temp[ids] /= count_temp[ids]
#
#     np.add.at(var_temp, ids, (data - mean_temp[ids]) ** 2)
#     var_temp[ids] /= count_temp[ids]
#
#     new_mean[:] = (count[ids] * mean[ids] + count_temp[ids] * mean_temp[ids]) / (count[ids] + count_temp[ids])
#     var[ids] = ((count[ids] * (var[ids] + (mean[ids] - new_mean) ** 2)) +
#                count_temp[ids] * (var_temp[ids] + (mean_temp[ids] - new_mean) ** 2)) / (count[ids] + count_temp[ids])
#     mean[ids] = new_mean
#     count += count_temp

def update_batch(data, ids, count, mean, var):

    ids_r = ids[:, 0] * count.shape[1] + ids[:, 1]
    ids_r_u, ids_r_u_i = np.unique(ids_r, return_inverse=True)  # TODO: This is by far the slowest part of this function.
    ids_r_u = (ids_r_u, )
    ids_r_u_i = (ids_r_u_i, )
    ids_unr = (ids_r_u[0]//count.shape[1], ids_r_u[0]%count.shape[1])

    count_temp = np.bincount(ids_r)[ids_r_u]

    if not isinstance(data, list):
        data = [data]
        mean = [mean]
        var = [var]

    for d, m, v in zip(data, mean, var):
        mean_temp = np.bincount(ids_r, d)[ids_r_u]
        mean_temp /= count_temp

        var_temp = np.bincount(ids_r, (d - mean_temp[ids_r_u_i])**2)[ids_r_u]
        var_temp /= count_temp

        new_mean = (count[ids_unr] * m[ids_unr] + count_temp * mean_temp) / (count[ids_unr] + count_temp)

        v[ids_unr] = ((count[ids_unr] * (v[ids_unr] + (m[ids_unr] - new_mean) ** 2)) +
                      count_temp * (var_temp + (mean_temp - new_mean) ** 2)) / (count[ids_unr] + count_temp)

        m[ids_unr] = new_mean

    count[ids_unr] += count_temp
    return ids_unr


def update_batch_single_sample(data, ids, count, mean, var):
    ids_r = ids[:, 0] * count.shape[1] + ids[:, 1]
    ids_r_u, ids_r_u_i = np.unique(ids_r, return_inverse=True)  # TODO: This is by far the slowest part of this function.
    ids_r_u = (ids_r_u, )
    ids_r_u_i = (ids_r_u_i, )
    ids_unr = (ids_r_u[0]//count.shape[1], ids_r_u[0]%count.shape[1])

    sample_count = np.bincount(ids_r)[ids_r_u]
    count[ids_unr] += 1

    if not isinstance(data, list):
        data = [data]
        mean = [mean]
        var = [var]

    for d, m, v in zip(data, mean, var):
        sample_mean = np.bincount(ids_r, d)[ids_r_u]
        sample_mean /= sample_count

        delta = sample_mean - m[ids_unr]
        m[ids_unr] += delta/count[ids_unr]

        v[ids_unr] += delta * (sample_mean - m[ids_unr])

    return ids_unr


def update_histogram(data, ids, histogram):
    ids_r = ids[:, 0] * histogram.shape[1] + ids[:, 1] + np.floor(data * 10.0).astype(np.int) * histogram.shape[0] * histogram.shape[1]
    ids_r_u = np.unique(ids_r)
    ids_unr = ((ids_r_u[0]//histogram.shape[1])%histogram.shape[0], ids_r_u[0]%histogram.shape[1], ids_r_u[0]//histogram.shape[0]//histogram.shape[1])

    hist_count = np.bincount(ids_r)[ids_r_u]
    histogram[ids_unr] += hist_count


def update_histogram_angle(data, angle, ids, histogram):
    # ids_r = ids[:, 0] + \
    #         ids[:, 1] * histogram.shape[0] + \
    #         np.floor(angle/np.pi * histogram.shape[2]).astype(np.int) * histogram.shape[0] * histogram.shape[1] + \
    #         np.floor(data * histogram.shape[3]).astype(np.int) * histogram.shape[0] * histogram.shape[1] * histogram.shape[2]
    ids_r = np.ravel_multi_index(
        (
            ids[:, 0],
            ids[:, 1],
            np.floor(angle / np.pi * histogram.shape[2]).astype(np.int),
            np.floor(data * histogram.shape[3]).astype(np.int)
        ),
        histogram.shape
    )
    ids_r_u = np.unique(ids_r)
    ids_unr = np.unravel_index(ids_r_u, histogram.shape)

    hist_count = np.bincount(ids_r)[ids_r_u]
    histogram[ids_unr] += hist_count


# @jit((float32, int32, int32, float32, float32))
def update_sequential(data, ids, count, mean, m2):

    if ids.ndim == 2:
        ids = tuple(ids)

    for i, v in enumerate(data):
        ind = ids[i]
        count[ind] += 1
        delta = v - mean[ind]
        mean[ind] += delta/count[ind]
        m2[ind] += delta * (v - mean[ind])


# if __name__ == '__main__':
#     datapoints = 9000
#     bins = (5, 5)
#
#     data = np.random.randn(datapoints) + 3
#     ids = np.random.randint(0, bins[0], (datapoints, 2))
#     # print('Normal:\t', np.mean(data), np.var(data))
#
#     count = np.zeros(bins)
#     mean = np.zeros(bins)
#     var = np.zeros(bins)
#     with TimeIt('Batch'):
#             update_batch_numba(data[:datapoints//2], ids[:datapoints//2], count, mean, var)
#             update_batch_numba(data[datapoints//2:], ids[datapoints//2:], count, mean, var)
#     print('Batch: \t', count, mean, var)
#
#     count = np.zeros(bins)
#     mean = np.zeros(bins)
#     var = np.zeros(bins)
#     with TimeIt('Batch'):
#             update_batch(data[:datapoints//2], ids[:datapoints//2], count, mean, var)
#             update_batch(data[datapoints//2:], ids[datapoints//2:], count, mean, var)
#     print('Batch: \t', count, mean, var)
#
#
#     count = np.zeros(bins)
#     mean = np.zeros(bins)
#     var = np.zeros(bins)
#     with TimeIt('Sequential'):
#         update_sequential(data[:datapoints//2], ids[:datapoints//2], count, mean, var)
#         update_sequential(data[datapoints//2:], ids[datapoints//2:], count, mean, var)
#     print('Sequential:\t', count, mean, var/count)

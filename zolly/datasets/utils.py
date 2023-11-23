import numpy as np


def select_frames_index(
    index: int,
    batch_size: int,
    num_frames: int,
    interval: int,
    fix_interval: bool = True,
    temporal_successive: bool = True,
):
    """Select two batches of frame index.
        If temporal_successive: the source and target indexes will be
            successive, else will be random.
        If fix_interval: the source and target indexes will have a fixed
            interval, else will be from -interval to interval.
    """
    if temporal_successive:
        index_0_source = index
        if fix_interval:
            index_0_target = min(index_0_source + interval,
                                 num_frames - batch_size + 1)
        else:
            index_0_target = np.random.randint(
                low=max(0, index_0_source - interval),
                high=min(num_frames - batch_size + 1,
                         index_0_source + interval))
        indexes_source = np.arange(index_0_source, index_0_source + batch_size)
        indexes_target = np.arange(index_0_target, index_0_target + batch_size)
    else:
        indexes_source = np.random.randint(low=0,
                                           high=num_frames,
                                           size=batch_size)
        if fix_interval:
            indexes_target = indexes_source + interval
            indexes_target = np.clip(indexes_target, 0, num_frames - 1)
        else:
            indexes_target = np.random.randint(
                low=-interval, high=interval, size=batch_size) + indexes_source
            indexes_target = np.clip(indexes_target, 0, num_frames - 1)
    return list(indexes_source), list(indexes_target)

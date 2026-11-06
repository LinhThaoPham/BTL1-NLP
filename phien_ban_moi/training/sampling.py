"""Random oversampling chỉ dùng trong train, có lưu chỉ số mẫu được lấy lại."""
import numpy as np


def random_oversample_indices(y, seed=42):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if set(classes) != {0, 1}:
        raise ValueError("Train cần đủ hai lớp trước khi oversampling.")
    rng = np.random.RandomState(seed)
    indices = [np.arange(len(y))]
    target = int(max(counts))
    for label, count in zip(classes, counts):
        if count < target:
            indices.append(rng.choice(np.flatnonzero(y == label), target - int(count), replace=True))
    result = np.concatenate(indices)
    rng.shuffle(result)
    return result

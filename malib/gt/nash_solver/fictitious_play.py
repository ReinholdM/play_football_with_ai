# -*- coding: utf-8 -*-
import numpy as np

# reference from https://nashpy.readthedocs.io/en/latest/reference/fictitious-play.html
def fictitious_play(A, B, num_iters):
    assert A.shape == B.shape, (A.shape, B.shape)
    play_counts = [np.array([0] * dim) for dim in A.shape]
    yield play_counts

    for n_iter in range(num_iters):
        plays = []
        for mat, play_count in zip((A, B.transpose()), play_counts[::-1]):
            utilities = mat @ play_count
            plays.append(
                np.random.choice(
                    np.argwhere(utilities == np.max(utilities)).transpose()[0]
                )
            )
        for play_count, play in zip(play_counts, plays):
            play_count[play] += 1

        yield play_counts

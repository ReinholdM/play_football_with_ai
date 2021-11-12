# -*- encoding: utf-8 -*-
# -----
# Created Date: 2021/1/21
# Author: Hanjing Wang
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 MARL @ SJTU
# -----

import pytest
from .chunk import serialize, deserialize


def test_int():
    a = 24
    assert deserialize(serialize(a)) == a


def test_float():
    b = 1.4887
    assert deserialize(serialize(b)) == b


def test_bool():
    assert deserialize(serialize(True)) == True
    assert deserialize(serialize(False)) == False


def test_str():
    c = "test sample\\ 2{}[]{}#@!()%%77&98"
    assert deserialize(serialize(c)) == c


def test_list():
    l = [1, 2, 3, "23", "abc", 2.566, 3.666, True, False]
    assert deserialize(serialize(l)) == l


def test_numpy_random_array():
    import numpy as np

    arr = np.random.rand(233, 277)
    assert (deserialize(serialize(arr)) == arr).all()

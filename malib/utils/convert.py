# -*- encoding: utf-8 -*-
###
# Created Date: 11/19/2020 21:20:50
# Author: Hanjing Wang
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 MARL @ SJTU
###
import ray
import time


def get_head_node_ip():
    # return ray.nodes()[0]["NodeManagerAddress"]
    cluster_addr = ray._private.services.get_ray_address_to_use_or_die()
    ip = cluster_addr.split(":")[0]
    return ip


def utc_to_str(utc_time) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(utc_time))


def dump_dict(d, indent=4):
    ss = "{\n"
    for k, v in d.items():
        ss += " " * indent + str(k) + ": "
        if isinstance(v, dict):
            ss += dump_dict(v, indent=indent + 4)
        else:
            ss += str(v)
        ss += "\n"
    ss += "}\n"
    return ss


def grpc_struct_to_dict(any_struct, skip_fields=[]):
    res = {}
    for f in any_struct.DESCRIPTOR.fields:
        if f not in skip_fields:
            res[f.name] = getattr(any_struct, f.name)
    return res


def tensor_to_dict(input_tensor):
    import numpy as np

    res = {}
    if not isinstance(input_tensor, np.ndarray):
        raise TypeError("numpy.ndarray objects expected")
    res["Shape"] = input_tensor.shape
    res["Values"] = input_tensor
    return res


def anyof(dict_object: dict):
    return next(iter(dict_object.values()))

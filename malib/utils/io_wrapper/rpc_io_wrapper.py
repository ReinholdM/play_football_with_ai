# -*- encoding: utf-8 -*-
###
# Created Date: 12/01/2020 22:40:27
# Author: Hanjing Wang
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 MARL @ SJTU
###

from .base_io_wrapper import BaseIOWrapper


class RpcIOWrapper(BaseIOWrapper):
    def __init__(self, in_stream=None, out_stream=None):
        raise NotImplementedError

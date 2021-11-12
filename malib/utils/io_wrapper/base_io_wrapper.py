# -*- encoding: utf-8 -*-
###
# Created Date: 11/16/2020 21:29:18
# Author: Hanjing Wang
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 MARL @ SJTU
###

import abc


class BaseIOWrapper(abc.ABC):
    """Abstract base class for io wrapper.

    The wrapper serves the following purposes

    * Unified local/remote files r/w

    """

    @abc.abstractmethod
    def write(self, object):
        """
        Serialize object and write/send to target uri.
        """
        pass

    @abc.abstractmethod
    def read(self):
        pass

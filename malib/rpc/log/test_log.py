# -*- encoding: utf-8 -*-
###
# Created Date: 01/14/2021 20:00:47
# Author: Hanjing Wang
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 MARL @ SJTU
###

import os
from log_server import LoggerServer
from log_client import LoggerClient
from malib.utils.io_wrapper import StandardIOWrapper
from malib.utils import io_wrapper

server_addr = "127.0.0.1:8080"
"""
    LoggerServer code part.
    Define and start a server
"""
# Bind the address of the server
server = LoggerServer(
    port=server_addr, io_wrappers=[StandardIOWrapper()], grace=10, max_workers=10
)
server.start()


client = LoggerClient(server_addr)
msg1 = "test logger module, the first time"
status, rec_time = client.send(msg1)

msg2 = "test logger module, second time"
# [optional] information importance level
level = 1
status, rec_time = client.send(msg2, level)

server.stop()

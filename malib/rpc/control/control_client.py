# -*- encoding: utf-8 -*-
###
# Created Date: 11/20/2020 16:14:50
# Author: Hanjing Wang
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 MARL @ SJTU
###

import time
import grpc

import sys

sys.path.append("..")
from ..proto import control_pb2_grpc, control_pb2


def run(server_port):
    channel = grpc.insecure_channel(server_port)
    stub = control_pb2_grpc.ControlRPCStub(channel)

    while True:
        sig = control_pb2.BeatSignal(
            node_type="0", node_id="0", node_status="normal", send_time=time.time()
        )
        response = stub.HeatBeat(sig)
        print(
            "Control client received reply: [target: {}, action: {}, time: {}]".format(
                response.target_code, response.action_code, response.send_time
            )
        )
        time.sleep(1)

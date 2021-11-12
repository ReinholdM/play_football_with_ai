# -*- encoding: utf-8 -*-
# -----
# Created Date: 2021/1/26
# Author: Hanjing Wang
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 MARL @ SJTU
# -----
import time
import logging
import numpy as np
import multiprocessing as mp
from ExperimentServer import ExprManagerServer as Server
from ExperimentClient import ExprManagerClient as Client


def _start_server(port, logdir):
    s = Server(port, logdir=logdir)
    s.start()
    print("Server up!")
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("Keyboard interrupt!")
    s.stop()


def _start_client(port, cidx, log_level=logging.NOTSET):
    c = Client(port, nid=cidx, log_level=log_level)
    print("Client up!")

    future = c.create_table(
        primary="test_expr", secondary="run{}".format(cidx), wait_for_ready=True
    )
    while not future.done():
        continue
    key, recv_time = future.result()
    print(key, recv_time)

    # test scalar
    # test int-to-long
    int_scalars = [78, 99, 101]
    for i in int_scalars:
        c.send_scalar(tag="send_scalar_int", content=i, global_step=i)
    # test float support
    float_scalars = [3.1415, 2.7877, 0.618]
    for f in float_scalars:
        c.send_scalar(tag="send_scalar_float", content=f)

    payoff_update = [("p1", "p2", 0.244), ("s7hja", "&&&", -280281.0)]
    c.send_obj(tag="__payoff__", obj=payoff_update, global_step=2)
    # test figure
    import matplotlib.pyplot as plt

    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    plt.plot(x, y)
    c.send_figure(tag="send_figure", global_step=666, walltime=0)

    # test image
    # send as binary objects
    pic_name = "./_test_sample_{}.png".format(cidx)
    plt.savefig(pic_name)
    with open(pic_name, "rb") as img_fp:
        img_data = img_fp.read()
        c.send_image(
            tag="send_image_binary", image=img_data, serial=False, global_step=233
        )
    # send as 3D HWC tensors
    from PIL import Image

    img = Image.open(pic_name)
    img_tensor = np.array(img)
    c.send_image(
        tag="send_image_tensor", image=img_tensor, serial=True, global_step=777
    )
    img.close()
    print("Client {} stopped".format(cidx))
    import os

    os.system("rm {}".format(pic_name))


def test():
    port = "localhost:12333"
    logdir = "recv"
    server_process = mp.Process(
        target=_start_server,
        args=(
            port,
            logdir,
        ),
    )
    clients = []
    for cidx in range(5):
        client_process = mp.Process(
            target=_start_client,
            args=(port, cidx, logging.DEBUG),
        )
        clients.append(client_process)

    server_process.start()
    time.sleep(2)
    for client_process in clients:
        client_process.start()

    for client_process in clients:
        client_process.join()
    server_process.join()


def test_subprocess():
    from ExperimentServer import start_logging_server

    port = "localhost:12333"
    sp = start_logging_server(port=port)
    sp.start()

    cp = mp.Process(
        target=_start_client,
        args=(port, 0, logging.DEBUG),
    )
    cp.start()
    cp.join()
    sp.terminate()


if __name__ == "__main__":
    test()
    test_subprocess()

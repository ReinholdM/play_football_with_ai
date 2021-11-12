# -*- encoding: utf-8 -*-
# -----
# Created Date: 2021/3/18
# Author: Hanjing Wang
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 MARL @ SJTU
# -----
import time
import threading
from mongo_client import MongoClient as Client

test_table_config = {"primary": "test_expr", "secondary": "run1"}


def test_create_client():
    c = Client(nid="client:test_create_client")
    c.create_table(**test_table_config)
    time.sleep(10)


def test_logging():
    c = Client(nid="client:test_logging")
    c.create_table(**test_table_config)
    # single push
    for i in range(1):
        c.info(f"info-test-message{i}")

    # multi push
    for i in range(10):
        c.warning(f"warning-test-message{i}")
        time.sleep(0.1)
    # stress push
    for i in range(100):
        c.debug(f"debug-test-message{i}")
    # parallel push
    threads = []
    for i in range(10):
        threads.append(threads.Thread(target=c.error, args=(f"error-test-message{i}")))
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_send_scalar():
    pass


def test_send_scalars():
    pass

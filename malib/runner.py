"""
runner.py

@Organization:  
@Author: Ming Zhou
@Time: 2/18/21 8:39 PM
@Function:
"""

import copy
import signal
import pprint
import threading
import time
import traceback
from typing import Dict, Any, List

import ray

from malib import settings
from malib.utils.logger import get_logger, Log
from malib.utils.configs.formatter import DefaultConfigFormatter
from malib.utils.convert import get_head_node_ip
from malib.rpc.ExperimentManager.ExperimentServer import start_logging_server

WAIT_FOR_READY_THRESHOLD = 10
logger_server = None


def update_configs(update_dict, ori_dict=None):
    """Update global configs with"""
    ori_configs = (
        copy.copy(ori_dict)
        if ori_dict is not None
        else copy.copy(settings.DEFAULT_CONFIG)
    )
    for k, v in update_dict.items():
        # assert k in ori_configs, f"Illegal key: {k}, {list(ori_configs.keys())}"
        if isinstance(v, dict):
            ph = ori_configs[k] if isinstance(ori_configs.get(k), dict) else {}
            ori_configs[k] = update_configs(v, ph)
        else:
            ori_configs[k] = copy.copy(v)
    return ori_configs


def _exit_handler(sig, frame):
    raise SystemExit


def _terminate(recycle_funcs: List[Dict[str, Any]], waiting: bool = True):
    background_recycle_threads = []
    for call in recycle_funcs:
        background_recycle_threads.append(
            threading.Thread(target=call["func"], args=call["args"])
        )
    for thread in background_recycle_threads:
        thread.start()
    if waiting:
        for thread in background_recycle_threads:
            thread.join()
        print("Background recycling thread ended.")


def start_logger(exp_cfg, enable_tensorboard_backend, exp_infos=None):
    if enable_tensorboard_backend:
        global logger_server
        server_address = "[::]"
        server_port = settings.REMOTE_PORT or 12333
        logger_server = start_logging_server(
            port=f"{server_address}:{server_port}", logdir=settings.LOG_DIR
        )
        logger_server.start()

    # wait for the logging server to be ready
    _wait_for_ready_start_time = time.time()
    while True:
        try:
            if settings.USE_REMOTE_LOGGER:
                if settings.USE_MONGO_LOGGER:
                    server_address = settings.MONGO_IP_ADDRESS or get_head_node_ip()
                    server_port = settings.MONGO_PORT or 27017
                else:
                    server_address = settings.REMOTE_IP_ADDRESS or get_head_node_ip()
                    server_port = settings.REMOTE_PORT or 12333
            print(server_address, server_port)
            logger = get_logger(
                name="runner",
                remote=settings.USE_REMOTE_LOGGER,
                mongo=settings.USE_MONGO_LOGGER,
                host=server_address,
                port=server_port,
                info=exp_infos,
                **exp_cfg,
            )
            logger.info("Wait for server ready", wait_for_ready=True)
            return logger
        except Exception as e:
            if time.time() - _wait_for_ready_start_time > WAIT_FOR_READY_THRESHOLD:
                raise RuntimeError(
                    "Wait time exceed threshold, "
                    "task cancelled, "
                    "cannot connect to logging server, "
                    "please check the network availability!"
                )
            time.sleep(1)


def terminate_logger():
    global logger_server
    logger_server.terminate()


def run(**kwargs):
    config = locals()["kwargs"]
    global_configs = update_configs(config)

    if global_configs["training"]["interface"].get("worker_config") is None:
        global_configs["training"]["interface"]["worker_config"] = {
            "num_cpus": None,
            "num_gpus": None,
            "memory": None,
            "object_store_memory": None,
            "resources": None,
        }
    if settings.USE_REMOTE_LOGGER and settings.USE_REMOTE_LOGGER:
        infos = DefaultConfigFormatter.parse(global_configs, filter=True)
        print(f"Logged experiment information:{pprint.pformat(infos)}")
    else:
        infos = {}

    exp_cfg = {
        "expr_group": global_configs.get("group", "experiment"),
        "expr_name": f"{global_configs.get('name', 'case')}_{time.time()}",
    }

    ray.init(address=None, local_mode=False)
    resources = ray.available_resources()
    print("Total resources:", resources)

    try:
        signal.signal(signal.SIGTERM, _exit_handler)
        from malib.backend.coordinator.server import CoordinatorServer
        from malib.backend.datapool.offline_dataset_server import OfflineDataset
        from malib.backend.datapool.parameter_server import ParameterServer

        # def run_coordinator_server(coordinator_server_configs):
        logger = start_logger(
            exp_cfg,
            enable_tensorboard_backend=settings.USE_REMOTE_LOGGER
            and not settings.USE_MONGO_LOGGER,
            exp_infos=infos,
        )

        offline_dataset = OfflineDataset.options(
            name=settings.OFFLINE_DATASET_ACTOR, max_concurrency=100
        ).remote(global_configs["dataset_config"], exp_cfg)
        parameter_server = ParameterServer.options(
            name=settings.PARAMETER_SERVER_ACTOR, max_concurrency=1000
        ).remote(exp_cfg=exp_cfg, **global_configs["parameter_server"])

        coordinator_server = CoordinatorServer.options(
            name=settings.COORDINATOR_SERVER_ACTOR, max_concurrency=100
        ).remote(exp_cfg=exp_cfg, **global_configs)
        _ = ray.get(coordinator_server.start.remote())

        with Log.timer(log=settings.PROFILING, logger=logger):
            while True:
                terminate = ray.get(coordinator_server.is_terminate.remote())
                if terminate:
                    print("ALL task done")
                    break
                else:
                    time.sleep(1)
        _terminate(
            [{"func": ray.shutdown, "args": tuple()}]
            + (
                [{"func": logger_server.terminate, "args": tuple()}]
                if logger_server
                else []
            ),
            waiting=True,
        )
    except (KeyboardInterrupt, SystemExit) as e:
        print(
            "KeyboardInterrupt or fatal error detected, start background resources recycling threads ..."
        )
        _terminate(
            [
                {"func": ray.shutdown, "args": tuple()},
                {"func": offline_dataset.shutdown.remote, "args": ()},
                {"func": parameter_server.shutdown.remote, "args": ()},
            ]
            + (
                [{"func": logger_server.terminate, "args": tuple()}]
                if logger_server
                else []
            ),
            waiting=False,
        )

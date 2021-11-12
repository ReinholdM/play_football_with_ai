# -*- encoding: utf-8 -*-
# -----
# Created Date: 2021/2/20
# Author: Hanjing Wang
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 MARL @ SJTU
# -----

import enum
import os
import time
import gridfs
import psutil
import pymongo
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from collections import deque, defaultdict
from typing import Iterable

from malib.utils import errors
from malib.utils.configs.config import EXPERIMENT_MANAGER_CONFIG as CONFIG
from malib.utils.typing import Any, Dict, Tuple, Union, EventReportStatus


class DocType(enum.Enum):
    Settings = "SETTINGS"
    HeartBeat = "HEARTBEAT"
    Metric = "METRIC"
    Report = "REPORT"
    Resource = "RESOURCE"
    Payoff = "PAYOFF"
    Logging = "LOGGING"


def extract_dict(target: Dict[str, Any]) -> Dict[str, Any]:

    """
    Flatten a nested dict

    :param target: targeted nested dict
    :return: flattened dict
    """
    if target is None:
        return None

    res = {}
    for k, v in target.items():
        if isinstance(v, Dict):
            subdict = extract_dict(v)
            for sk, sv in subdict.items():
                res[f"{k}/{sk}"] = str(sv)
        else:
            res[k] = str(v)
    return res


class MongoClient:
    """
    The mongo client class exposes similar APIs as ExperimentManagerClient,
    requiring a mongodb running in the background as data storage. A mongo client
    instance will be bound to the collection
    *CONN[db_name][f"{expr_group}-{expr_name}"]*
    after calling the create_table method and then start its background heart
    beating thread.
    """

    def __init__(
        self,
        host: Any = None,
        port: Any = None,
        log_level: int = None,
        db_name: str = "expr",
        heart_beat_interval: float = 5,
        heart_beat_max_trail: int = 10,
        nid: str = "",
    ):
        """
        :param Any host: (currently only support single-database connection) url or hostname
        :param Any port: connection port
        :param int log_level: logging level
        :param str db_name: database name
        :param float heart_beat_interval: the interval(seconds) between background heart beat reports
        :param int heart_beat_max_trail: timeout=heart_beat_interval * heart_beat_max_trail,
                        the maximum number of heart beat trails waiting for reply,
                        after which an HeartbeatTimeout will be raised.
        :param str nid: node id, included in the message for source identification.

        :return None
        """
        self._server_addr = host
        self._client = pymongo.MongoClient(
            host=host or os.environ.get("MONGO_SERVER_ADDR", "localhost"),
            port=port or int(os.environ.get("MONGO_SERVER_PORT", 27017)),
        )
        self._executor = ThreadPoolExecutor()

        self._config = CONFIG.copy()
        self._config["nid"] = nid
        if log_level:
            self._config["log_level"] = log_level

        self._database = self._client[db_name]
        self._key = None  # current experiment name
        self._expr = None  # current experiment collection
        self._fs = gridfs.GridFS(self._database)

        # background heart beat thread, started after bound to a table
        self._exit_flag = False
        self._hb_thread = threading.Thread(
            target=self._heart_beat,
            args=(heart_beat_interval, heart_beat_max_trail, self._config["nid"]),
        )
        self._hb_thread.daemon = True

    @staticmethod
    def init_expr(
        database: pymongo.database.Database,
        primary: str,
        secondary: str,
        content: Dict[str, Any],
        nid: str,
    ) -> pymongo.collection.Collection:
        """
        Init a collection for information storage and create a meta information description.
        :param pymongo.database.Database database: root database
        :param str primary: experiment group
        :param str secondary: experiment name
        :param Dict[str, Any] content: experiment meta-info content
        :param str nid: creator node id
        :return: pymongo.collection.Collection
        """
        expr_key = f"{primary}-{secondary}"
        collection = database[expr_key]
        expr_doc = collection.count_documents({"type": DocType.Settings.value}, limit=1)

        if expr_doc == 0:
            res = collection.insert_one(
                {"id": nid, "type": DocType.Settings.value, **content}
            )

        return collection

    def __del__(self):
        self._exit_flag = True
        self._hb_thread.join()
        print(f"Mongo logger-{self._config['nid']} destroyed.")

    def _heart_beat(self, interval: float, max_trail: int, nid: str) -> None:
        """
        Periodically reporting the node resource utilization in the background,
        including CPU / Mem / GPU Mem.
        :param float interval: Time interval for background heart beat report
        :param int max_trail: Max number of waiting trails before a HeartbeatTimeout is raised
        :param nid: The id of the associated node of the mongo client.
        :raises errors.HeartbeatTimeout
        :return : None
        """

        class ProcStatus:
            def __init__(self):
                self.d = {
                    "heartbeat": time.time(),
                    "cpu": 0,
                    "mem": 0,
                    "gpu": None,
                }

        trails_history = deque(maxlen=max_trail)
        current_process = psutil.Process()
        has_gpu = False
        gpu_handlers = []
        try:
            import pynvml

            pynvml.nvmlInit()
            gpu_num = pynvml.nvmlDeviceGetCount()
            for gpu_id in range(gpu_num):
                gpu_handlers.append(pynvml.nvmlDeviceGetHandleByIndex(gpu_id))
        except ImportError as e:
            print("MongoClient: can not import pynvml package, gpu monitor disabled")
            has_gpu = False
        except pynvml.NVMLError as e:
            print("MongoClient: can not load nvml lib, gpu monitor disabled")
            has_gpu = False

        while not self._exit_flag:
            status = ProcStatus()
            status.d["cpu"] = current_process.cpu_percent()
            mem_summary = current_process.memory_info()
            status.d["mem"] = (
                mem_summary.rss - mem_summary.shared
                if hasattr(mem_summary, "shared")
                else mem_summary.rss
            )
            gpu_mem_usages = []
            for handler in gpu_handlers:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handler)
                gpu_mem = 0
                for proc in procs:
                    if proc.pid == current_process.pid:
                        gpu_mem += proc.usedGpuMemory
                gpu_mem_usages.append(gpu_mem)
            status.d["gpu"] = gpu_mem_usages

            doc = {
                "id": self._config["nid"],
                "type": DocType.HeartBeat.value,
                **status.d,
            }
            future = self._executor.submit(self._expr.insert_one, doc)
            trails_history.append(future)

            if len(trails_history) == max_trail:
                trail = trails_history[0]
                if not trail.done():
                    raise errors.HeartbeatTimeout(
                        f"Logger{self._config['nid']} heart beat trails waiting overtime"
                    )
                else:
                    while len(trails_history) > 1 and trails_history[0].done():
                        trails_history.popleft()

            time.sleep(interval)

    def _built_in_logging(self, level, content) -> Future:
        if self._config["log_level"] > level:
            return None

        doc = {
            "id": self._config["nid"],
            "type": DocType.Logging.value,
            "level": level,
            "msg": content,
            "time": time.time(),
        }
        return self._executor.submit(self._expr.insert_one, doc)

    def create_table(
        self, primary=None, secondary=None, extra_info: Dict[str, Any] = None
    ):
        _call_params = {
            "database": self._database,
            "primary": primary or self._config["primary"],
            "secondary": secondary or self._config["secondary"],
            "nid": self._config["nid"],
            "content": {
                "Primary": primary or self._config["primary"],
                "Secondary": secondary or self._config["secondary"],
                "ExperimentInfo": extract_dict(extra_info),
                "StartTime": time.time(),
                "CreatedBy": self._config["nid"],
            },
        }
        future = self._executor.submit(self.init_expr, **_call_params)
        while not future.done():
            continue

        self._expr = future.result()
        self._hb_thread.start()

    def info(self, content="", **kwargs) -> Future:
        return self._built_in_logging(level=logging.INFO, content=content)

    def warning(self, content="", **kwargs) -> Future:
        return self._built_in_logging(level=logging.WARNING, content=content)

    def debug(self, content="", **kwargs) -> Future:
        return self._built_in_logging(level=logging.DEBUG, content=content)

    def error(self, content="", **kwargs) -> Future:
        return self._built_in_logging(level=logging.ERROR, content=content)

    def report(
        self,
        status: EventReportStatus,
        event_id: int,
        metric: Any = None,
        info: str = "",
    ) -> Future:
        """
        Update events status of the associated node. If metric is set to be None,
        then the report is processed as a plain event update, otherwise the report
        will be viewed as throughput profiling message as well. Currently, the
        attribute "metric" is set in the malib.utils.logger.__init__.Log.data_feedback
        when PROFILING option activated.

        :param EventReportStatus status: Either representing "start" or "end"
        :param Any event_id: A random generated uuid to identify different events, especially
                        for the nested cases.
        :param Tuple[int] metric: Numerical metrics to profile the event(currently data throughput).
        :param str info: Extra infos
        :return:
        """

        """
        Event_id here are mainly used to identify parallel tasks in profiling
        """
        doc = {
            "id": self._config["nid"],
            "type": DocType.Report.value,
            "status": status,
            "event": event_id,
            "info": info,
            "time": time.time(),
        }
        if metric is not None:
            doc.update({"metric": metric})

        return self._executor.submit(self._expr.insert_one, doc)

    def send_scalar(
        self,
        tag: str = None,
        content: Union[int, float] = None,
        tag_content_dict: Dict[str, Any] = {},
        global_step: int = 0,
        walltime: float = None,
        batch_mode: bool = False,
    ) -> Future:
        """
        Send scalar logging information to database, which can be visualized in plots.
        For logging efficiency, a batch of metrics can be recorded in a single document
        in mongodb if batch_mode is set to be True. And these metrics will be visualized
        in separated plots.

        >>> examples: MongoClient.send_scalar(tag="agent_reward", content=float(3), global_step=2, walltime=time.time())
        >>> examples: MongoClient.send_scalar(tag_content_dict={
                                                        "agent_reward": float(3),
                                                        "adversary_reward": float(-2),
                                                    }, batch_mode=True, global_step=3, walltime=time.time())
        :param str tag: Metric tag.
        :param Union[int, float] content: Metric value.
        :param Dict[str, Union[int, float]] tag_content_dict: Batched metric in the form of
                                            {tag: value}, please set batch_mode=True if you want to
                                            use tag_content_dict to batch metrics recording.
        :param int global_step:
        :param float walltime:
        :param bool batch_mode: Set to be True if batching metrics is needed
        :return: Future
        """
        walltime = walltime or time.time()
        doc_identifier_region = {
            "id": self._config["nid"],
            "type": DocType.Metric.value,
        }

        """
            aggregate = False: parsed as scalar batch, will visualized separately, 
            aggregate = True for send_scalars
        """
        if batch_mode:
            doc_content_region = {
                "aggregate": False,
                "content": [
                    {"name": sub_tag, "content": field_value}
                    for sub_tag, field_value in tag_content_dict.items()
                ],
                "step": global_step,
                "time": walltime,
            }
        else:
            doc_content_region = {
                "aggregate": False,
                "name": tag,
                "content": content,
                "step": global_step,
                "time": walltime,
            }
        return self._executor.submit(
            self._expr.insert_one, {**doc_identifier_region, **doc_content_region}
        )

    def send_scalars(
        self,
        tag: Iterable[str],
        content: Dict[str, Any],
        global_step: int = 0,
        walltime=None,
    ) -> Future:
        """
        EXPERIMENTAL Feature.

        Integrated different plots in a single figure.

        :param tag:
        :param content:
        :param global_step:
        :param walltime:
        :return:
        """
        """
        Expect the tag_value_dict in the form of
        {str, float} where the strs are taken as tags,
        the floats are taken as values. Or in the form
        of {str, (float, int)}, where ints are taken as
        steps.
        """
        walltime = walltime or time.time()
        doc = {
            "id": self._config["nid"],
            "type": DocType.Metric.value,
            "aggregate": True,
            "name": tag,
            "content": [
                {"name": sub_tag, "content": field_value}
                for sub_tag, field_value in content.items()
            ],
            "step": global_step,
            "time": walltime,
        }
        return self._executor.submit(self._expr.insert_one, doc)

    def send_arbitrary_object(self, f, filename: str) -> Future:
        """
        Store a readable object into gridfs.

        :param f: Readable object/content
        :param sre filename:
        :return: Future
        """
        file_id = self._fs.put(f, filename=filename)
        doc = {
            "id": self._config["nid"],
            "type": DocType.Resource.value,
            "content": file_id,
            "time": time.time(),
        }
        return self._executor.submit(self._expr.insert_one, doc)

    def get(self, tags: Tuple[str] = None) -> Future:
        _call_params = {"collection": self._expr, "fields": tags}
        return self._executor.submit(self.pull, **_call_params)

    # TODO(jing) elegently print payoff matrix
    def send_obj(
        self, tag, obj, global_step: int = 0, walltime: Union[float, int] = None
    ) -> Future:
        """
        worked as sending payoff matrix
        """
        if tag == "__Payoff__":
            columns = defaultdict(lambda: [])
            for (aid, pid), (raid, reward) in zip(
                obj["Population"].items(), obj["Agents-Reward"]
            ):
                columns["aid"].append(aid)
                columns["pid"].append(pid)
                assert aid == raid
                columns["reward"].append(reward)
            doc = {
                "id": self._config["nid"],
                "type": DocType.Payoff.value,
                "columns": columns,
                "step": global_step,
                "time": walltime or time.time(),
            }
        else:
            raise NotImplementedError
        self._executor.submit(self._expr.insert_one, doc)

    def send_image(self) -> Future:
        raise NotImplementedError

"""PP-OCRv6 CUDA adapter using an independent worker process.

The host-side adapter deliberately contains no Paddle, OpenCV, CUDA, or
cuDNN imports.  The proprietary NVIDIA/Paddle stack is loaded only by
``worker.py`` in a child process and communicates with this adapter through a
small JSON-lines protocol.  This is an actual process boundary, not a
LoadLibrary/dlopen trick: requests contain paths/base64 bytes and responses
contain JSON OCR records only.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from copy import deepcopy
from pathlib import Path

from .config import PRESETS


# Keep the logical package path (rather than resolving a staging junction) so
# a test/package host can find its sibling ``UmiOCR-data/runtime``.  The worker
# itself may resolve the path when locating its vendor files.
PLUGIN_DIR = Path(__file__).parent
WORKER = PLUGIN_DIR / "worker.py"
WORKER_LAUNCHER = PLUGIN_DIR / "UmiOCR-CudaWorker.exe"
DEVICE_RE = __import__("re").compile(r"^gpu:(\d+)$")


class _WorkerClient:
    def __init__(self):
        self.process = None
        self.responses = queue.Queue()
        self.reader = None
        self.stderr_file = None
        self.lock = threading.RLock()

    def start(self, config, timeout):
        with self.lock:
            self.stop()
            # A clean queue is part of the process restart contract.  The old
            # worker may have emitted its final ``stop`` acknowledgement just
            # as its reader thread was being torn down; never let that stale
            # line satisfy the new worker's ``start`` request.
            self.responses = queue.Queue()
            if not WORKER.is_file() or not WORKER_LAUNCHER.is_file():
                raise FileNotFoundError("CUDA worker 原生入口或脚本缺失，请完整安装匹配的插件。")
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            self.stderr_file = tempfile.NamedTemporaryFile(
                prefix="umi-ocr-cuda-worker-", suffix=".log", delete=False
            )
            environment = os.environ.copy()
            environment["PYTHONUTF8"] = "1"
            environment["PYTHONIOENCODING"] = "utf-8"
            self.process = subprocess.Popen(
                [str(WORKER_LAUNCHER)],
                cwd=str(PLUGIN_DIR),
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr_file,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                creationflags=creationflags,
            )
            self.reader = threading.Thread(
                target=self._read, args=(self.process.stdout, self.responses),
                daemon=True, name="UmiOCR-CUDA-worker-reader",
            )
            self.reader.start()
            response = self.request({"op": "start", "config": config}, timeout)
            if not response.get("ok"):
                raise RuntimeError(response.get("error") or "GPU worker failed to initialize")
            return response

    def _read(self, stream, responses):
        # 读线程只持有当前代次的队列，旧进程的EOF不能污染重启后的响应。
        error = "GPU worker response stream closed; inspect the worker diagnostic log"
        try:
            if stream is not None:
                for line in stream:
                    try:
                        responses.put(json.loads(line))
                    except (ValueError, TypeError) as exc:
                        responses.put({"ok": False, "error": f"invalid worker response: {exc}"})
        except (OSError, ValueError) as exc:
            error = f"GPU worker response stream failed: {exc}"
        finally:
            responses.put({"ok": False, "error": error})

    def request(self, payload, timeout):
        with self.lock:
            if not self.process or not self.process.stdin:
                raise RuntimeError("GPU worker process is not running")
            wait = max(1.0, float(timeout))
            if self.process.poll() is not None:
                wait = min(wait, 1.0)
            else:
                try:
                    self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    self.process.stdin.flush()
                except OSError:
                    wait = min(wait, 1.0)
            try:
                response = self.responses.get(timeout=wait)
            except queue.Empty as exc:
                self.stop()
                raise TimeoutError("GPU worker did not answer before the configured timeout") from exc
            if not isinstance(response, dict):
                raise RuntimeError("GPU worker returned a non-object response")
            return response

    def stop(self):
        process = self.process
        self.process = None
        try:
            if process is not None and process.poll() is None and process.stdin:
                try:
                    process.stdin.write(json.dumps({"op": "stop"}) + "\n")
                    process.stdin.flush()
                except OSError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            if process is not None:
                for stream in (process.stdin, process.stdout):
                    try:
                        if stream:
                            stream.close()
                    except OSError:
                        pass
            if self.stderr_file:
                try:
                    self.stderr_file.close()
                except OSError:
                    pass
                self.stderr_file = None


class Api:
    def __init__(self, globalArgd):
        self.device = str((globalArgd or {}).get("device", "gpu:0")).strip().lower()
        match = DEVICE_RE.fullmatch(self.device)
        if not match:
            raise ValueError("CUDA 设备必须使用 gpu:N 格式，例如 gpu:0。")
        self.device_index = int(match.group(1))
        self._config = None
        self._worker = _WorkerClient()
        self._lock = threading.RLock()
        self._warmup_thread = None
        self._warmup_token = None
        self._last_used_at = None
        self._last_error = ""
        self._last_timing = {}
        self._startup = {}

    @staticmethod
    def _normalize_config(argd):
        source = dict(PRESETS["accuracy"])
        source.update(argd or {})
        task_type = str(source.get("taskType", "screenshot"))
        if task_type not in {"screenshot", "batch", "document"}:
            task_type = "screenshot"
        undistort = str(source["undistort"])
        textline_orientation = str(source["textlineOrientation"])
        return {
            "device": str(source.get("device", "gpu:0")).lower(),
            "detModel": str(source["detModel"]),
            "recModel": str(source["recModel"]),
            "docOrientation": bool(source["docOrientation"]),
            "undistort": undistort,
            "textlineOrientation": textline_orientation,
            "useDocUnwarping": undistort == "on" or (undistort == "auto" and task_type in {"batch", "document"}),
            "useTextlineOrientation": textline_orientation == "on" or (textline_orientation == "auto" and task_type != "screenshot"),
            "longSide": int(source["longSide"]),
            "recBatchSize": int(source["recBatchSize"]),
            "scoreThreshold": float(source["scoreThreshold"]),
        }

    @staticmethod
    def _validate_config(config):
        if config["detModel"] != "PP-OCRv6_medium" or config["recModel"] != "PP-OCRv6_medium":
            raise ValueError("未知 PP-OCRv6 模型；本发布只包含 medium det/rec。")
        if config["undistort"] not in {"off", "auto", "on"}:
            raise ValueError("文档去扭曲选项无效。")
        if config["textlineOrientation"] not in {"off", "auto", "on"}:
            raise ValueError("文本行方向选项无效。")
        if not DEVICE_RE.fullmatch(config["device"]):
            raise ValueError("CUDA 设备必须使用 gpu:N 格式。")

    def start(self, argd):
        # The constructor device is the host-selected default.  Callers often
        # pass only task-specific overrides to start(); silently falling back
        # to the preset gpu:0 would make an invalid-device request appear to
        # succeed and would ignore the plugin's configured device.
        start_argd = dict(argd or {})
        start_argd.setdefault("device", self.device)
        config = self._normalize_config(start_argd)
        with self._lock:
            if config == self._config and self._worker.process and self._worker.process.poll() is None:
                return ""
            self._startup = {}
            try:
                self._validate_config(config)
                response = self._worker.start(config, timeout=180)
                self._startup = deepcopy(response.get("startup", {}))
                self._config = config
                self._last_used_at = time.monotonic()
                self._last_error = ""
                return ""
            except Exception as exc:
                self._worker.stop()
                self._config = None
                self._startup = {}
                self._last_error = f"[Error] PP-OCRv6 CUDA worker 启动失败：{exc}"
                return self._last_error

    def warmup(self, argd):
        config = self._normalize_config(argd)
        with self._lock:
            if config == self._config and self._worker.process and self._worker.process.poll() is None:
                return "[Success] PP-OCRv6 CUDA 已驻留显存。"
            if self._warmup_thread and self._warmup_thread.is_alive():
                return "[Success] PP-OCRv6 CUDA 正在后台预热。"
            token = object()
            self._warmup_token = token

            def run():
                message = self.start(config)
                with self._lock:
                    if self._warmup_token is token:
                        self._warmup_token = None
                if message:
                    try:
                        from umi_log import logger
                        logger.error(message)
                    except Exception:
                        pass

            self._warmup_thread = threading.Thread(target=run, name="UmiOCR-CUDA-Warmup", daemon=True)
            self._warmup_thread.start()
        return "[Success] PP-OCRv6 CUDA 已开始后台预热。"

    def get_execution_info(self):
        config = self._config or {}
        models = dict.fromkeys(config.get(key, "") for key in ("detModel", "recModel"))
        return {
            "device": config.get("device", "设备未确认").upper(),
            "model": " / ".join(model.replace("_", " ") for model in models if model),
            "engine": "paddle",
        }

    def get_status(self):
        thread = self._warmup_thread
        loaded = bool(self._worker.process and self._worker.process.poll() is None and self._config)
        return {
            "state": "loaded" if loaded else ("warming" if thread and thread.is_alive() else "unloaded"),
            "idleSeconds": round(max(0.0, time.monotonic() - self._last_used_at), 3) if loaded and self._last_used_at else 0.0,
            "error": self._last_error,
            "lastTiming": dict(self._last_timing),
            "architecture": "separate-worker-stdio-ipc",
            "startup": deepcopy(self._startup) if loaded else {},
        }

    def release(self):
        with self._lock:
            was_active = bool(self._config or self._worker.process)
            self._worker.stop()
            self._config = None
            self._startup = {}
            self._last_timing = {}
            self._last_used_at = None
            self._last_error = ""
        return "[Success] PP-OCRv6 CUDA worker 已清理。" if was_active else "[Warning] PP-OCRv6 CUDA 尚未加载。"

    def stop(self):
        self.release()

    def runPath(self, imgPath):
        return self._request({"op": "run_path", "path": str(imgPath)})

    def runBytes(self, imageBytes):
        return self._request({"op": "run_bytes", "data": base64.b64encode(imageBytes).decode("ascii")})

    def runBase64(self, imageBase64):
        try:
            return self.runBytes(base64.b64decode(imageBase64, validate=True))
        except Exception as exc:
            return {"code": 202, "data": f"[Error] Base64 图片无效：{exc}"}

    def _request(self, payload):
        with self._lock:
            if not self._config:
                return {"code": 203, "data": "[Error] PP-OCRv6 CUDA worker 尚未启动。"}
            try:
                response = self._worker.request(payload, timeout=180)
                self._last_used_at = time.monotonic()
                if response.get("timing"):
                    self._last_timing = dict(response["timing"])
                if not response.get("ok", False):
                    return {"code": 204, "data": f"[Error] PP-OCRv6 CUDA 推理失败：{response.get('error', 'unknown worker error')}"}
                return response.get("result", {"code": 101, "data": ""})
            except Exception as exc:
                self._last_error = f"[Error] PP-OCRv6 CUDA worker 通信失败：{exc}"
                return {"code": 204, "data": self._last_error}


def health(argd=None):
    ready = WORKER.is_file() and WORKER_LAUNCHER.is_file()
    return {
        "ok": ready,
        "message": "GPU OCR 使用原生 UTF-8 独立 worker；CUDA/Paddle 仅在 worker 进程中加载。" if ready else "缺少 GPU worker.py 或原生启动入口，请完整安装插件。",
        "details": {"worker": str(WORKER), "launcher": str(WORKER_LAUNCHER), "architecture": "separate-worker-stdio-ipc"},
    }

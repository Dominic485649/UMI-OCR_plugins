"""插件级单例：OCR/document 串行租用固定内置 llama.cpp，不接管外部进程。"""
from __future__ import annotations

import atexit
from collections import deque
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from . import checks
from .checks import CheckError, tr

LOG_PREFIX = "umi-paddlevl-server-"
LOG_TAIL_BYTES = 4000


def tempRoot():
    root = Path(os.getcwd()) / "temp_doc"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _int(argd, key, default, low, high):
    return checks.timeoutSeconds(argd, key, default, low, high)


class RuntimeService:
    """只持有自己 Popen 返回的进程句柄；状态/清理受同一租约锁保护。"""

    def __init__(self):
        self._lock = threading.RLock()
        self._leaseCondition = threading.Condition(self._lock)
        self._process = None
        self._logFile = self._logPath = None
        self._command = []
        self._config = {}
        self._fingerprint = None
        self._runtimeIdentity = None
        self._flags = {}
        self._state, self._error = "stopped", ""
        self._leases = set()
        self._leaseQueue = deque()
        self._lastUsedAt = None
        self._wake = threading.Event()

    def configure(self, argd):
        merged = checks.mergeArgd(argd)
        with self._lock:
            if self._busyChanged(merged):
                return self._busy()
            self._config = merged
            return checks.ok(tr("模型服务设置已应用。"), **self.get_status())

    def _busyChanged(self, merged):
        return bool((self._leases or self._leaseQueue) and self._fingerprint and self._fingerprintFor(merged) != self._fingerprint)

    def _busy(self):
        return checks.fail(tr("文字识别或文档解析正在使用模型服务，不能在任务运行时修改或释放服务。"), errorType="SERVICE_BUSY", activeLeases=len(self._leases), waitingRequests=len(self._leaseQueue))

    def get_status(self):
        with self._lock:
            process = self._process
            return {
                "state": self._state, "pid": process.pid if process and process.poll() is None else 0,
                "external": False, "activeLeases": len(self._leases), "waitingRequests": len(self._leaseQueue),
                "idleSeconds": round(0 if self._lastUsedAt is None else max(0, time.monotonic() - self._lastUsedAt), 3),
                "error": self._error, "logPath": str(self._logPath or ""), "command": list(self._command),
                "runtimeIdentity": self._runtimeIdentity,
            }

    def get_execution_info(self):
        with self._lock:
            info = {"device": tr('设备未确认'), "model": "PaddleOCR-VL 1.6", "engine": "llama.cpp"}
            if not self._isOwnedRunning() or not self._logPath:
                return info
            try:
                with open(self._logPath, "r", encoding="utf-8", errors="replace") as stream:
                    startup = stream.read(262144)
                layers = re.search(r"offloaded (\d+)/(\d+) layers to GPU", startup)
                vision = re.search(r"using (\S+) backend", startup)
                if layers and vision:
                    count, total = map(int, layers.groups())
                    vision_gpu = "CPU" not in vision.group(1).upper()
                    info["device"] = "GPU" if count == total and vision_gpu else "CPU" if count == 0 and not vision_gpu else "GPU+CPU"
            except OSError:
                pass
            return info

    def status(self):
        return self.get_status()

    def isRunning(self):
        with self._lock:
            return self._isOwnedRunning()

    def buildCommand(self, argd):
        argd = checks.mergeArgd(argd)
        runtime = checks.bundledRuntime()
        executable = runtime["path"]
        main_model = checks.requireFile(argd, "service.mainModel", tr('主 GGUF'), {".gguf"})
        mmproj = checks.requireFile(argd, "service.mmprojModel", tr('视觉投影 GGUF'), {".gguf"})
        checks.checkGguf(main_model, tr('主 GGUF'), expectProjector=False)
        checks.checkGguf(mmproj, tr('视觉投影 GGUF'), expectProjector=True)
        flags = checks.checkServerBinary(executable).get("flags") or {}
        if not flags.get("alias") or not flags.get("special"):
            raise CheckError(tr("内置 llama-server 不支持模型别名或文字定位标记，请重新安装配套插件。"), 902)
        budget = checks.resourceBudget(argd, local=False)
        host, port, _ = checks.endpoint(argd)
        batch = _int(argd, "resources.batchSize", 2048, 32, 8192)
        ubatch = _int(argd, "resources.ubatchSize", 512, 32, 4096)
        if ubatch > batch:
            raise CheckError(tr("Ubatch 不能大于 Batch。"), option="resources.ubatchSize")
        command = [
            executable, "-m", main_model, "--mmproj", mmproj, "--alias", "paddlevl", "--host", host, "--port", str(port),
            "--temp", "0", "-ngl", "all", "-c", str(budget["contextLength"]), "-np", str(budget["slots"]),
            "-n", str(budget["maxOutputTokens"]), "--special",
        ]
        # Paddle 分支实际像素预算由 worker 分块保证，不把通用 image-max-tokens 当作保证。
        warnings = []
        def add(flag, value=None, supported=True, label=""):
            if supported:
                command.append(flag)
                if value is not None:
                    command.append(str(value))
            elif label:
                warnings.append(tr("内置运行时不支持可选参数 {name}，已省略。", name=label))
        add("-b", batch, flags.get("batch"), "Batch")
        add("-ub", ubatch, flags.get("ubatch"), "Ubatch")
        flash = str(argd.get("resources.flashAttention", "auto") or "auto").lower()
        if flash not in ("auto", "on", "off"):
            raise CheckError(tr("Flash Attention 只能是 auto、on 或 off。"), option="resources.flashAttention")
        add("-fa", flash, flags.get("flashAttention"), "Flash Attention")
        for key, flag, yes, no in (("mmprojOffload", "mmprojOffload", "--mmproj-offload", "--no-mmproj-offload"), ("kvOffload", "kvOffload", "--kv-offload", "--no-kv-offload")):
            enabled = argd.get("resources." + key, True)
            if type(enabled) is not bool:
                raise CheckError(tr("{name} 必须是布尔值。", name=key), option="resources." + key)
            add(yes if enabled else no, supported=flags.get(flag if enabled else "no" + flag[0].upper() + flag[1:]), label=key)
        for key, flag, cli in (("cacheKeyType", "cacheKey", "-ctk"), ("cacheValueType", "cacheValue", "-ctv")):
            value = argd.get("resources." + key, "f16")
            if value not in ("f16", "q8_0"):
                raise CheckError(tr("{name} 只能是 f16 或 q8_0。", name=key), option="resources." + key)
            add(cli, value, flags.get(flag), key)
        add("--jinja", supported=flags.get("jinja"), label="Jinja")
        add("--no-ui", supported=flags.get("noUi"), label="--no-ui")
        add("--verbosity", 4, flags.get("verbosity"))
        self._runtimeIdentity = {**runtime["identity"], "manifestSha256": runtime["manifestSha256"]}
        return command, host, port, flags, warnings

    def start(self, argd=None):
        return self.ensure_ready(argd)

    def prestart(self, argd=None):
        return self.ensure_ready(argd)

    def ensure_ready(self, argd=None, cancel=None):
        with self._lock:
            merged = checks.mergeArgd(self._config, argd)
            if self._busyChanged(merged):
                return self._busy()
            try:
                fingerprint = self._fingerprintFor(merged)
                changed = bool(self._fingerprint and fingerprint != self._fingerprint)
                timeout = checks.timeoutSeconds(merged, "resources.startupTimeoutSeconds", 180, 10, 3600)
                host, port, base_url = checks.endpoint(merged)
                if self._isOwnedRunning() and changed:
                    self._stopOwnedLocked()
                self._config = merged
                # 自有进程运行中仍检查文件完整性；改变 stat 时不会复用旧 hash。
                checks.bundledRuntime()
                if self._isOwnedRunning():
                    return self._waitReady(base_url, timeout, cancel)
                # 不探测/接管外部服务，即使 /health 与 alias 完全一致也只是端口冲突。
                if checks.isPortBusy(host, port):
                    self._state, self._error = "failed", tr("端口已被其他进程占用。")
                    return checks.fail(tr("端口 {host}:{port} 已被其他进程占用。请改用空闲端口或自行退出占用程序；插件不会接管或终止它。", host=host, port=port), errorType="PORT_CONFLICT")
                if cancel and cancel.is_set():
                    return checks.fail(tr("任务已取消。"), errorType="CANCELLED")
                command, _, _, self._flags, warnings = self.buildCommand(merged)
                self._command, self._fingerprint = list(command), fingerprint
                self._closeLog()
                self._logFile = tempfile.NamedTemporaryFile(prefix=LOG_PREFIX, suffix=".log", dir=tempRoot(), delete=False)
                self._logPath = Path(self._logFile.name)
                self._process = self._spawn(command)
                self._state, self._error = "starting", ""
                result = self._waitReady(base_url, timeout, cancel)
                if warnings:
                    result.setdefault("details", {})["warnings"] = warnings
                return result
            except CheckError as exc:
                self._state, self._error = "failed", exc.message
                return checks.fail(exc.message, **exc.details)
            except OSError as exc:
                self._closeLog()
                self._state, self._error = "failed", str(exc)
                return checks.fail(tr("启动内置 llama-server 失败：{error}。请检查插件完整性与 NVIDIA 驱动。", error=str(exc)), errorType="SERVER_START_FAILED")

    def acquire(self, purpose, argd=None, cancel=None):
        # 就绪和入队在同一锁内，禁止最后一次就绪检查与排队之间被清理。
        with self._leaseCondition:
            ready = self.ensure_ready(argd, cancel=cancel)
            if not ready.get("ok"):
                details = dict(ready.get("details") or {})
                details.setdefault("errorType", ready.get("errorType"))
                raise CheckError(ready.get("message") or tr('模型服务不可用。'), 902, **details)
            merged = checks.mergeArgd(self._config, argd)
            timeout = checks.timeoutSeconds(merged)
            lease = f"{purpose}-{time.monotonic_ns()}"
            self._leaseQueue.append(lease)
            deadline = time.monotonic() + timeout
            try:
                while self._leaseQueue[0] != lease or self._leases:
                    remaining = deadline - time.monotonic()
                    if cancel and cancel.is_set():
                        raise CheckError(tr("任务已取消。"), 904, errorType="CANCELLED")
                    if remaining <= 0:
                        raise CheckError(tr("等待共享模型服务超过 {seconds} 秒，请等待当前任务结束后重试。", seconds=timeout), 908)
                    self._leaseCondition.wait(min(remaining, 0.1))
                if cancel and cancel.is_set():
                    raise CheckError(tr("任务已取消。"), 904, errorType="CANCELLED")
                self._leaseQueue.popleft()
                self._leases.add(lease)
                self._state = "busy"
                return lease
            except BaseException:
                if lease in self._leaseQueue:
                    self._leaseQueue.remove(lease)
                self._leaseCondition.notify_all()
                raise

    def release(self, lease=None):
        if lease is None or isinstance(lease, dict):
            return self.stop_owned()
        with self._leaseCondition:
            self._leases.discard(lease)
            self._lastUsedAt = time.monotonic()
            if not self._leases:
                self._state = "ready" if self._isOwnedRunning() else "stopped"
            self._leaseCondition.notify_all()

    def health(self, argd=None):
        with self._lock:
            merged = checks.mergeArgd(self._config, argd)
            try:
                runtime = checks.bundledRuntime()
                for key, title, projector in (("service.mainModel", tr('主 GGUF'), False), ("service.mmprojModel", tr('视觉投影 GGUF'), True)):
                    checks.checkGguf(checks.requireFile(merged, key, title, {".gguf"}), title, expectProjector=projector)
                checks.resourceBudget(merged)
                _, _, base_url = checks.endpoint(merged)
                if not self._isOwnedRunning():
                    return checks.fail(tr("内置运行时与模型文件有效，模型将在首次使用时启动；CUDA 实际加载尚未验证。"), errorType="MODEL_NOT_READY", runtime=runtime, **self.get_status())
                result = checks.probeService(base_url, min(checks.timeoutSeconds(merged), 15))
                return {**result, "details": {**result.get("details", {}), "runtime": runtime, **self.get_status()}}
            except CheckError as exc:
                return checks.fail(exc.message, **exc.details)

    def maintain(self, argd=None):
        # 预热延迟和 never/memory/idle 的决定由宿主唯一策略器作出。
        with self._lock:
            if self._leases or self._leaseQueue:
                return self._busy()
            reason = (argd or {}).get("reason", "")
            if reason in ("memory", "idle", "release"):
                return self._stopOwnedLocked()
            return checks.ok(tr("模型服务无需处理。"), **self.get_status())

    def stop_owned(self, *_ignored):
        with self._lock:
            if self._leases or self._leaseQueue:
                return self._busy()
            return self._stopOwnedLocked()

    def stop(self):
        return self.stop_owned()

    def shutdown_owned(self):
        return self.stop_owned()

    def _waitReady(self, baseUrl, timeout, cancel=None):
        deadline, last = time.monotonic() + timeout, ""
        while time.monotonic() < deadline:
            if cancel and cancel.is_set():
                self._stopOwnedLocked()
                return checks.fail(tr("任务已取消。"), errorType="CANCELLED")
            if not self._isOwnedRunning():
                code = getattr(self._process, "returncode", None)
                self._process = None
                self._state, self._error = "failed", self._logTail()
                return checks.fail(tr("内置 llama-server 启动后退出（代码 {code}）：{error}", code=code, error=self._error), errorType="SERVER_EXITED", exitCode=code)
            probe = checks.probeService(baseUrl, max(0.1, min(10, deadline - time.monotonic())))
            # 健康探测期间自有进程退出时，不接受抢占端口的其他服务。
            if probe["ok"] and self._isOwnedRunning():
                execution = self.get_execution_info()
                if execution.get("device") not in ("GPU", "GPU+CPU"):
                    self._stopOwnedLocked()
                    self._state = "failed"
                    return checks.fail(tr("无法确认内置运行时已将模型与视觉计算加载到 CUDA，本插件不提供 CPU 回退。请检查 NVIDIA 驱动与启动日志。"), errorType="GPU_INITIALIZATION_FAILED")
                self._state, self._error = "busy" if self._leases else "ready", ""
                self._lastUsedAt = time.monotonic()
                return checks.ok(tr("内置 llama-server 已就绪。"), **probe["details"], execution=execution, cudaLoadVerified=execution.get("device") in ("GPU", "GPU+CPU"), **self.get_status())
            last = probe["message"]
            self._wake.wait(min(0.25, max(0, deadline - time.monotonic())))
        self._stopOwnedLocked()
        self._state, self._error = "failed", last
        return checks.fail(tr("等待内置 llama-server 就绪超时：{error}", error=last), errorType="SERVER_START_TIMEOUT")

    def _spawn(self, command):
        # 直接执行固定 EXE；不再引入 PATH 中的 shell 或外部程序作为回退。
        return subprocess.Popen(command, cwd=os.path.dirname(command[0]), stdin=subprocess.DEVNULL,
                                stdout=self._logFile, stderr=subprocess.STDOUT, shell=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)

    def _stopOwnedLocked(self):
        if self._leases or self._leaseQueue:
            return self._busy()
        process = self._process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        self._process, self._command, self._fingerprint = None, [], None
        self._state = "stopped"
        self._closeLog()
        return checks.ok(tr("已停止插件自有的 llama-server。"), **self.get_status())

    def _isOwnedRunning(self):
        return bool(self._process and self._process.poll() is None)

    def _fingerprintFor(self, argd):
        keys = ("service.port", "service.mainModel", "service.mmprojModel", "resources.contextLength",
                "resources.maxOutputTokens", "resources.slots", "resources.batchSize",
                "resources.ubatchSize", "resources.flashAttention", "resources.mmprojOffload", "resources.kvOffload",
                "resources.cacheKeyType", "resources.cacheValueType")
        return (checks.bundledRuntimeStamp(),) + tuple((key, str(argd.get(key, ""))) for key in keys)

    def _closeLog(self):
        if self._logFile:
            self._logFile.close()
            self._logFile = None

    def _logTail(self):
        self._closeLog()
        if not self._logPath or not self._logPath.is_file():
            return tr('(没有捕获到服务端日志)')
        try:
            with self._logPath.open("rb") as stream:
                stream.seek(max(0, self._logPath.stat().st_size - LOG_TAIL_BYTES))
                return self._redactLog(stream.read().decode("utf-8", errors="replace"))
        except OSError as exc:
            return str(exc)

    @staticmethod
    def _redactLog(text):
        value = str(text or "")
        value = re.sub(r"(?i)([A-Z]:\\Users\\)[^\\\r\n]+", tr('\\1<用户>'), value)
        value = re.sub(r"(?i)(/home/)[^/\r\n]+", tr('\\1<用户>'), value)
        return re.sub(r"(?i)\b(api[_-]?key|authorization|password|access[_-]?token)\b\s*([=:])\s*[^\s,;]+", tr('\\1\\2<已隐藏>'), value)


_SERVICE = RuntimeService()
atexit.register(_SERVICE.shutdown_owned)


def getService():
    return _SERVICE

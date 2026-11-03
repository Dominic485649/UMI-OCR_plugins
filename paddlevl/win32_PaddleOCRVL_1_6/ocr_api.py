"""PaddleOCR-VL 1.6 Spotting OCR capability for MissionOCR."""

from __future__ import annotations

import base64
import os
import tempfile
import threading
import time
from pathlib import Path

from . import checks
from .checks import tr
from .api import _WorkerClient
from .checks import CheckError
from .runtime import getService, tempRoot


def _sharedService():
    try:
        from py_src.plugins_controller.services import PluginServices

        return PluginServices.get(checks.PLUGIN_ID, "paddlevl")
    except Exception:
        return getService()


class Api:
    """传统 OCR 形状的适配器，底层始终是 PaddleOCR-VL Spotting。"""

    def __init__(self, argd):
        self._globalArgd = dict(argd or {})
        self._argd = checks.mergeArgd(self._globalArgd)
        self._runtime = _sharedService()
        self._worker = _WorkerClient()
        self._workerKey = None
        self._lock = threading.RLock()
        self._state = "unloaded"
        self._error = ""
        self._lastUsedAt = None
        self._cancelled = threading.Event()

    def start(self, argd):
        self._cancelled.clear()
        merged = checks.mergeArgd(self._globalArgd, argd)
        with self._lock:
            self._argd = merged
            try:
                checks.resourceBudget(merged)
                ready = self._runtime.ensure_ready(merged, cancel=self._cancelled)
                if not ready.get("ok"):
                    details = dict(ready.get("details") or {})
                    details.setdefault("errorType", ready.get("errorType"))
                    raise CheckError(ready["message"], 902, **details)
                self._ensureWorker(merged)
            except CheckError as exc:
                self._state, self._error = "unavailable", f"[Error] {exc.message}"
                return self._error
            except Exception as exc:
                self._state, self._error = "unavailable", tr('[Error] 文字识别后端启动失败：{error}', error=exc)
                return self._error
            self._state, self._error = "loaded", ""
            self._lastUsedAt = time.monotonic()
            return ""

    def warmup(self, argd):
        msg = self.start(argd)
        return msg or tr('[Success] PaddleOCR-VL 文字识别已预热。')

    def runPath(self, path):
        source = Path(str(path or "")).expanduser()
        if not source.is_file():
            return {
                "code": 901,
                "data": tr('[Error] 图片不存在：{path} 。', path=source),
                "errorType": checks.errorTypeFor(901),
            }
        return self._run(source)

    def runBytes(self, data):
        if not isinstance(data, (bytes, bytearray)) or not data or len(data) > 64 * 1024 * 1024:
            return {
                "code": 901,
                "data": checks.tr("[Error] 图片字节为空、格式无效或超过 64 MiB，请拆分图片后重试。"),
                "errorType": checks.errorTypeFor(901),
            }
        handle = tempfile.NamedTemporaryFile(prefix="umi-vl-spotting-", suffix=".png", dir=tempRoot(), delete=False)
        try:
            handle.write(data)
            handle.close()
            return self._run(Path(handle.name))
        finally:
            try:
                os.unlink(handle.name)
            except OSError:
                pass

    def runBase64(self, data):
        try:
            payload = str(data or "")
            if "," in payload and payload.lstrip().startswith("data:"):
                payload = payload.split(",", 1)[1]
            if len(payload) > 90 * 1024 * 1024:
                raise ValueError(checks.tr("Base64 图片超过文件预算，请拆分图片后重试。"))
            return self.runBytes(base64.b64decode(payload, validate=True))
        except Exception as exc:
            return {
                "code": 901,
                "data": tr('[Error] Base64 图片无效：{error}', error=exc),
                "errorType": checks.errorTypeFor(901),
            }

    def cancel(self):
        self._cancelled.set()
        process = self._worker.process
        if process is not None and process.poll() is None:
            process.terminate()
            self._worker.responses.put({"ok": False, "error": checks.tr("任务已取消。"), "errorType": "CANCELLED"})

    def get_execution_info(self):
        return self._runtime.get_execution_info()

    def get_status(self):
        runtime = self._runtime.get_status()
        idle = 0 if self._lastUsedAt is None else max(0, time.monotonic() - self._lastUsedAt)
        return {
            "state": self._state if self._state != "unloaded" else runtime.get("state", "unloaded"),
            "idleSeconds": round(idle, 3), "error": self._error or runtime.get("error", ""),
            "service": runtime,
        }

    def release(self):
        with self._lock:
            active = self._worker.process is not None
            self._worker.stop()
            self._workerKey = None
            self._state, self._error = "unloaded", ""
        # Keep the plugin-wide server alive.  Stopping it here would make
        # switching between screenshot OCR and document parsing needlessly
        # reload the same GGUF, and would bypass the global idle policy.
        return tr('[Success] 已释放文字识别 worker。') if active else tr('[Success] 文字识别 worker 尚未加载。')

    def stop(self):
        with self._lock:
            self._worker.stop()
            self._workerKey = None
            self._state = "unloaded"

    def health(self, argd):
        merged = checks.mergeArgd(self._globalArgd, argd)
        details = {"worker": str(Path(__file__).with_name("worker.py"))}
        try:
            if not Path(details["worker"]).is_file() or not Path(details["worker"]).parent.joinpath("vendor").is_dir():
                raise CheckError(tr('PaddleOCR-VL worker 或私有 vendor 闭包缺失，请重新完整解压插件归档。'), 905)
            checks.resourceBudget(merged)
            _, _, _, flags, warnings = self._runtime.buildCommand(merged)
            details["dependencies"] = self._worker.probe(
                {"requireCuda": False},
                min(30, checks.timeoutSeconds(merged)),
            )
        except CheckError as exc:
            return checks.fail(exc.message, **details, **exc.details)
        except TimeoutError:
            return checks.fail(
                tr('PaddleOCR-VL 依赖检查超时。请确认插件私有运行时没有被安全软件拦截。'),
                errorType="REQUEST_TIMEOUT",
                **details,
            )
        except RuntimeError as exc:
            return checks.fail(tr('PaddleOCR-VL 私有依赖检查失败：{error}', error=exc), errorType="PIPELINE_DEPENDENCY_MISSING", **details)
        result = self._runtime.health(merged)
        result.setdefault("details", {}).update({**details, "llamaFlags": flags, "warnings": warnings})
        return result

    def _ensureWorker(self, argd):
        _, _, baseUrl = checks.endpoint(argd)
        key = (baseUrl, int(argd.get("resources.timeoutSeconds", 120)), self._worker._environment().get("UMI_PLUGIN_LANGUAGE", ""))
        if self._worker.process and self._worker.process.poll() is None and key == self._workerKey:
            return
        self._worker.stop()
        self._state = "warming"
        self._worker.start(
            {
                "mode": "spotting", "baseUrl": baseUrl, "model": "paddlevl",
                "timeout": checks.timeoutSeconds(argd), "concurrency": 1, "tempRoot": str(tempRoot()),
            },
            max(15, checks.timeoutSeconds(argd)),
        )
        self._workerKey = key

    def _run(self, source):
        with self._lock:
            if not self._worker.process or self._worker.process.poll() is not None:
                return {
                    "code": 903,
                    "data": tr('[Error] 文字识别后端尚未启动。请先应用全局设置后重试。'),
                    "errorType": checks.errorTypeFor(903),
                }
            lease = None
            try:
                budget = checks.resourceBudget(self._argd)
                task_seconds = checks.taskSeconds(self._argd, 300)
                lease = self._runtime.acquire("ocr", self._argd, cancel=self._cancelled)
                deadline = time.monotonic() + task_seconds
                from py_src.document.image_input import prepared_document_input
                with prepared_document_input(source, tempRoot(), deadline, self._cancelled.is_set, single_frame=True) as prepared:
                    response = self._worker.request(
                        {
                            "op": "spotting", "path": prepared["path"],
                            "originalPath": str(source.resolve()), "preparedInput": prepared["prepared"],
                            "taskTimeoutSeconds": task_seconds, "deadlineMonotonic": deadline,
                            "options": {"maxOutputTokens": budget["maxOutputTokens"]},
                        },
                        max(0.01, deadline - time.monotonic()) + 5,
                    )
                if not response.get("ok"):
                    raise CheckError(
                        str(response.get("error") or tr('文字识别 Pipeline 返回错误')),
                        904,
                        errorType=response.get("errorType") or "PIPELINE_ERROR",
                    )
                result = response.get("result")
                if not isinstance(result, dict) or result.get("code") not in (100, 101):
                    raise CheckError(
                        tr('INVALID_MODEL_OUTPUT：文字识别 Pipeline 返回了无效结果。'),
                        904,
                        errorType="INVALID_MODEL_OUTPUT",
                    )
                return result
            except CheckError as exc:
                self._error = f"[Error] {exc.message}"
                return {"code": exc.code, "data": self._error, "errorType": exc.errorType}
            except TimeoutError as exc:
                self._error = tr('[Error] 请求超时：{error}', error=exc)
                return {"code": 908, "data": self._error, "errorType": checks.errorTypeFor(908)}
            except Exception as exc:
                self._error = tr('[Error] 文字识别失败：{error}', error=exc)
                return {"code": 904, "data": self._error, "errorType": checks.errorTypeFor(904)}
            finally:
                if self._cancelled.is_set():
                    self._worker.stop()  # 等自有 worker 完全退出后才释放共享租约
                    self._workerKey = None
                if lease:
                    self._runtime.release(lease)
                self._lastUsedAt = time.monotonic()

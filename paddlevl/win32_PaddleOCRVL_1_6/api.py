"""PaddleOCR-VL 1.6 的完整文档解析 capability。"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import checks, asset_paths
from .checks import tr
from .checks import CheckError
from .runtime import RuntimeService, getService, tempRoot

WORKER = Path(__file__).with_name("worker.py")


class _WorkerClient:
    """独立 Paddle worker 的小型 JSON-lines 客户端，按 capability 惰性启动。"""

    def __init__(self):
        self.process = None
        self.responses = queue.Queue()
        self.reader = None
        self.stderr_file = None
        self.lock = threading.RLock()

    @staticmethod
    def _workerPython():
        worker_python = Path(sys.executable)
        bundled = WORKER.parents[2] / "runtime" / ("python.exe" if os.name == "nt" else "python")
        if bundled.is_file():
            worker_python = bundled
        return worker_python

    @staticmethod
    def _environment():
        environment = os.environ.copy()
        try:
            try:
                from plugin_i18n import LangCode
            except ImportError:
                from py_src.imports.plugin_i18n import LangCode
            environment["UMI_PLUGIN_LANGUAGE"] = LangCode
        except ImportError:
            pass
        return environment

    @staticmethod
    def _tail(text, limit=2000):
        # Worker stderr can include native-library diagnostics.  Keep the
        # useful tail but apply the same local-path/secret redaction used for
        # llama-server failures before it crosses the process boundary.
        return RuntimeService._redactLog(str(text or "")[-limit:])

    @classmethod
    def probe(cls, config, timeout):
        """Check the isolated Paddle environment without creating a pipeline.

        The worker's interpreter is deliberately used here: importing Paddle
        in the host process would inspect a different Python environment and
        could trigger CUDA initialization in Umi-OCR itself.
        """
        worker_python = cls._workerPython()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        payload = json.dumps({"op": "probe", "config": dict(config or {})}, ensure_ascii=False) + "\n"
        try:
            completed = subprocess.run(
                [str(worker_python), "-u", str(WORKER)],
                cwd=str(WORKER.parent), input=payload, text=True, encoding="utf-8",
                errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=max(5, float(timeout)), shell=False, creationflags=flags, env=cls._environment(),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(tr('PaddleOCR-VL 依赖检查在超时前没有返回。')) from exc
        except OSError as exc:
            raise RuntimeError(tr('无法启动 PaddleOCR-VL 依赖检查进程：{error}', error=exc)) from exc

        response = None
        for line in reversed(completed.stdout.splitlines()):
            try:
                parsed = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                response = parsed
                break
        if isinstance(response, dict):
            if response.get("ok"):
                return response.get("result") or {}
            raise RuntimeError(str(response.get("error") or tr('PaddleOCR-VL 依赖检查失败')))
        tail = cls._tail(completed.stderr)
        suffix = f"\n{tail}" if tail else ""
        raise RuntimeError(
            tr('PaddleOCR-VL 依赖检查进程异常退出（代码 {returncode}）。{suffix}', returncode=completed.returncode, suffix=suffix)
        )

    def start(self, config, timeout):
        with self.lock:
            self.stop()
            self.responses = queue.Queue()
            worker_python = self._workerPython()
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            self.stderr_file = tempfile.NamedTemporaryFile(prefix="umi-paddlevl-worker-", suffix=".log", dir=tempRoot(), delete=False)
            self.process = subprocess.Popen(
                [str(worker_python), "-u", str(WORKER)],
                cwd=str(WORKER.parent), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr_file, text=True, encoding="utf-8", errors="replace",
                shell=False, creationflags=flags, env=self._environment(),
            )
            self.reader = threading.Thread(target=self._read, args=(self.process.stdout, self.responses), daemon=True, name="UmiOCR-PaddleVL-worker")
            self.reader.start()
            response = self.request({"op": "start", "config": config}, timeout)
            if not response.get("ok"):
                raise RuntimeError(response.get("error") or tr('PaddleOCR-VL worker 初始化失败'))
            return response

    on_progress = None
    on_partial = None

    def _read(self, stream=None, responses=None):
        stream = stream or (self.process.stdout if self.process else None)
        responses = responses if responses is not None else self.responses
        if stream is None:
            return
        for line in stream:
            try:
                payload = json.loads(line)
                if payload.get("event") == "progress":
                    # 展示回调出错不能污染结果队列，也不能让一次 progress 延长请求时限。
                    try:
                        if callable(self.on_progress):
                            self.on_progress(str(payload.get("stage", "")))
                        if "page" in payload and callable(self.on_partial):
                            self.on_partial(payload)
                    except Exception:
                        pass
                else:
                    responses.put(payload)
            except Exception as exc:
                responses.put({"ok": False, "error": tr('worker 返回无效 JSON：{error}', error=exc)})

        responses.put({"ok": False, "error": tr('PaddleOCR-VL worker 已退出，请重试并检查工作进程日志。')})

    def request(self, payload, timeout):
        with self.lock:
            if not self.process or self.process.poll() is not None or not self.process.stdin:
                raise RuntimeError(tr('PaddleOCR-VL worker 未运行'))
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            try:
                response = self.responses.get(timeout=max(0.01, float(timeout)))
            except queue.Empty as exc:
                self.stop()
                raise TimeoutError(tr('PaddleOCR-VL worker 在请求超时前没有返回')) from exc
            if not isinstance(response, dict):
                raise RuntimeError(tr('PaddleOCR-VL worker 返回了非对象'))
            return response

    def stop(self):
        with self.lock:
            process, self.process = self.process, None
            if process is None:
                return
            try:
                if process.poll() is None and process.stdin:
                    try:
                        process.stdin.write('{"op":"stop"}\n')
                        process.stdin.flush()
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            finally:
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


def _sharedService():
    try:
        from py_src.plugins_controller.services import PluginServices

        return PluginServices.get(checks.PLUGIN_ID, "paddlevl")
    except Exception:
        return getService()


class Api:
    """完整 Pipeline：版面检测在 worker，VLM 请求复用插件级 llama-server。"""

    def __init__(self, argd):
        self._globalArgd = dict(argd or {})
        self._argd = checks.mergeArgd(self._globalArgd)
        self._worker = _WorkerClient()
        self._workerKey = None
        self._cancelled = threading.Event()
        self._runtime = _sharedService()
        self._lock = threading.RLock()
        self._state = "unloaded"
        self._error = ""
        self._lastUsedAt = None
        self._partialResults = {}

    def get_status(self):
        runtime = self._runtime.get_status()
        idle = 0 if self._lastUsedAt is None else max(0, time.monotonic() - self._lastUsedAt)
        return {
            "state": self._state if self._state != "unloaded" else runtime.get("state", "unloaded"),
            "idleSeconds": round(idle, 3), "error": self._error or runtime.get("error", ""),
            "service": runtime,
        }

    def start(self, argd):
        self._cancelled.clear()
        merged = checks.mergeArgd(self._globalArgd, argd)
        with self._lock:
            self._argd = merged
            try:
                self._validateDocumentConfig(merged)
                if callable(getattr(self, "on_progress", None)): self.on_progress(tr('启动模型服务'))
                self._worker.on_progress = getattr(self, "on_progress", None)
                self._worker.on_partial = self._onPartial
                ready = self._runtime.ensure_ready(merged, cancel=self._cancelled)
                if not ready.get("ok"):
                    details = dict(ready.get("details") or {})
                    details.setdefault("errorType", ready.get("errorType"))
                    raise CheckError(ready["message"], 902, **details)
                if self._cancelled.is_set(): raise RuntimeError(tr('文档解析已取消'))
                self._ensureWorker(merged)
            except CheckError as exc:
                self._state, self._error = "unavailable", f"[Error] {exc.message}"
                return self._error
            except Exception as exc:
                self._state, self._error = "unavailable", tr('[Error] 文档解析后端启动失败：{error}', error=exc)
                return self._error
            self._state, self._error = "loaded", ""
            self._lastUsedAt = time.monotonic()
            return ""

    def warmup(self, argd):
        msg = self.start(argd)
        return msg or tr('[Success] 文档解析后端已预热。')

    def cancel(self):
        self._cancelled.set()
        process = self._worker.process
        if process is not None and process.poll() is None:
            process.terminate()
            self._worker.responses.put({"ok": False, "error": tr('文档解析已取消'), "errorType": "CANCELLED"})

    def parseDoc(self, msn):
        from py_src.document.image_input import prepared_document_input
        source = Path(str((msn or {}).get("path", "") or "")).expanduser()
        if not source.is_file():
            return {
                "code": 901,
                "data": tr('[Error] 待解析文档不存在：{path} 。', path=source),
                "errorType": checks.errorTypeFor(901),
            }
        with self._lock:
            if self._worker.process is None or self._worker.process.poll() is not None:
                return {
                    "code": 903,
                    "data": tr('[Error] 文档解析后端尚未启动。请先应用全局设置后重试。'),
                    "errorType": checks.errorTypeFor(903),
                }
            lease = None
            self._partialResults = {}
            try:
                budget = checks.resourceBudget(self._argd)
                asset_root = asset_paths.output_root(msn["assetOutputRoot"]) if (msn or {}).get("assetOutputRoot") else None
                task_seconds = checks.taskSeconds(msn or {}, 1800)
                lease = self._runtime.acquire("document", self._argd, cancel=self._cancelled)
                deadline = time.monotonic() + task_seconds
                self._documentSource = source.resolve()
                with prepared_document_input(source, tempRoot(), deadline, self._cancelled.is_set) as prepared:
                    response = self._worker.request(
                        {
                            "op": "document",
                            "path": prepared["path"],
                            "originalPath": str(source.resolve()),
                            "preparedInput": prepared["prepared"],
                            "taskTimeoutSeconds": task_seconds,
                            "deadlineMonotonic": deadline,
                            "assetOutputRoot": str(asset_root) if asset_root is not None else None,
                            "options": {
                                "maxOutputTokens": budget["maxOutputTokens"],
                                "pageRange": (msn or {}).get("pageRange"),
                                "mergeTables": bool(self._argd.get("mergeTables", True)),
                                "relevelTitles": bool(self._argd.get("relevelTitles", True)),
                                "concatenatePages": bool(self._argd.get("concatenatePages", True)),
                                "exportAssets": bool(self._argd.get("exportAssets", True)),
                            },
                        },
                        max(0.01, deadline - time.monotonic()) + 5,
                    )
                if not response.get("ok"):
                    raise CheckError(
                        str(response.get("error") or tr('文档 Pipeline 返回错误')),
                        904,
                        errorType=response.get("errorType") or "PIPELINE_ERROR",
                    )
                result = response.get("result")
                if not isinstance(result, dict) or result.get("code") not in (100, 101, 904, 908):
                    raise CheckError(
                        tr('INVALID_MODEL_OUTPUT：文档 Pipeline 返回了无效结果。'),
                        904,
                        errorType="INVALID_MODEL_OUTPUT",
                    )
                return result
            except CheckError as exc:
                self._error = f"[Error] {exc.message}"
                if self._partialResults:
                    return self._partialFailure(source, self._error, exc.errorType)
                return {"code": exc.code, "data": self._error, "errorType": exc.errorType}
            except TimeoutError as exc:
                self._error = tr('[Error] 请求超时：{error}', error=exc)
                if self._partialResults:
                    return self._partialFailure(source, self._error, "REQUEST_TIMEOUT")
                return {"code": 908, "data": self._error, "errorType": checks.errorTypeFor(908)}
            except Exception as exc:
                self._error = tr('[Error] 文档解析失败：{error}', error=exc)
                return {"code": 904, "data": self._error, "errorType": checks.errorTypeFor(904)}
            finally:
                if self._cancelled.is_set():
                    self._releaseWorker()  # 取消后的 CUDA/HTTP 工作结束前不能释放租约
                if lease:
                    self._runtime.release(lease)
                self._lastUsedAt = time.monotonic()

    def health(self, argd):
        merged = checks.mergeArgd(self._globalArgd, argd)
        details = {"worker": str(WORKER), "workerExists": WORKER.is_file()}
        if not WORKER.is_file() or not (WORKER.parent / "vendor").is_dir():
            return checks.fail(tr('PaddleOCR-VL worker 或私有 vendor 闭包缺失，请重新完整解压插件归档。'), errorType="PIPELINE_DEPENDENCY_MISSING", **details)
        try:
            details["layoutModelDir"] = checks.requireModelDir(merged, "pipeline.layoutModelDir", tr('版面检测模型目录'))
            device = str(merged.get("pipeline.cudaDevice", "gpu:0"))
            if not device.startswith("gpu:"):
                raise CheckError(tr('版面检测设备必须是 gpu:N；CPU fallback 被拒绝。'), option="pipeline.cudaDevice")
            details["device"] = device
            _, _, _, flags, warnings = self._runtime.buildCommand(merged)
            details["llamaFlags"] = flags
            details["warnings"] = warnings
            details["dependencies"] = self._worker.probe(
                {
                    "requireCuda": True,
                    "device": device,
                },
                min(30, checks.timeoutSeconds(merged)),
            )
        except CheckError as exc:
            return checks.fail(exc.message, **details, **exc.details)
        except TimeoutError:
            return checks.fail(
                tr('PaddleOCR-VL 依赖与 GPU 检查超时。请确认 NVIDIA 驱动、Paddle CUDA 运行时和版面检测设备可用。'),
                errorType="REQUEST_TIMEOUT",
                **details,
            )
        except RuntimeError as exc:
            return checks.fail(
                tr('PaddleOCR-VL 私有依赖或 GPU 初始化失败：{error}', error=exc),
                errorType="GPU_INITIALIZATION_FAILED",
                **details,
            )
        service = self._runtime.health(merged)
        return {**service, "details": {**details, **service.get("details", {})}}

    def release(self):
        with self._lock:
            active = self._worker.process is not None
            self._releaseWorker()
            self._state, self._error = "unloaded", ""
        # The llama-server belongs to the plugin-level service, not to this
        # short-lived document capability.  Its lifecycle is controlled from
        # global settings / application shutdown so an OCR backend can reuse
        # the already loaded GGUF immediately.
        return tr('[Success] 已释放文档 Pipeline worker。') if active else tr('[Success] 文档 Pipeline worker 尚未加载。')

    def stop(self):
        with self._lock:
            self._releaseWorker()
            self._state = "unloaded"

    def serviceController(self):
        return self._runtime

    def _onPartial(self, event):
        result = event.get("result")
        if isinstance(result, dict) and isinstance(result.get("data"), dict):
            self._partialResults[event["page"]] = result
        callback = getattr(self, "on_partial", None)
        if callable(callback):
            callback(event)

    def _partialFailure(self, source, message, error_type):
        # worker 异常退出仍保留已回传的页，不能降格成完整成功或丢掉成果。
        pages, blocks, assets, warnings = [], [], [], [message]
        for _, result in sorted(self._partialResults.items()):
            data = result["data"]
            pages.extend(data.get("pages", []))
            blocks.extend(data.get("blocks", []))
            assets.extend(data.get("assets", []))
            warnings.extend(result.get("warnings", []))
        return {"code": 100, "partial": True, "errorType": error_type, "warnings": warnings,
            "text": "\n\n".join(page.get("text", "") for page in pages),
            "markdown": "\n\n".join(page.get("markdown", "") for page in pages),
            "data": {"schemaVersion": 1, "pages": pages, "blocks": blocks, "assets": assets},
            "metadata": {"source": str(source), "sourceStem": source.stem, "pageCount": len(pages), "incomplete": True}}

    def _validateDocumentConfig(self, argd):
        checks.resourceBudget(argd)
        checks.requireModelDir(argd, "pipeline.layoutModelDir", tr('版面检测模型目录'))
        device = str(argd.get("pipeline.cudaDevice", "gpu:0"))
        if not device.startswith("gpu:") or not device[4:].isdigit():
            raise CheckError(tr('版面检测设备必须写成 gpu:N；不会自动回退 CPU。'), option="pipeline.cudaDevice")

    def _ensureWorker(self, argd):
        layout = checks.requireModelDir(argd, "pipeline.layoutModelDir", tr('版面检测模型目录'))
        _, _, baseUrl = checks.endpoint(argd)
        key = (layout, str(argd.get("pipeline.cudaDevice", "gpu:0")), baseUrl, checks.timeoutSeconds(argd), self._worker._environment().get("UMI_PLUGIN_LANGUAGE", ""))
        if self._worker.process and self._worker.process.poll() is None and self._workerKey == key:
            return
        self._releaseWorker()
        config = {
            "mode": "document", "layoutModelDir": layout,
            "device": str(argd.get("pipeline.cudaDevice", "gpu:0")),
            "timeout": checks.timeoutSeconds(argd), "concurrency": 1,
            "baseUrl": baseUrl, "model": "paddlevl", "tempRoot": str(tempRoot()),
        }
        self._state = "warming"
        self._worker.start(config, max(15, checks.timeoutSeconds(argd)))
        self._workerKey = key

    def _releaseWorker(self):
        self._worker.stop()
        self._workerKey = None

"""Focused contract tests for the PaddleOCR-VL 1.6 dual-mode plugin.

These tests deliberately use a tiny HTTP server and a tiny JSON-lines worker;
they do not import Paddle or load a GGUF.  Real GPU/Pipeline validation is a
separate acceptance gate and must run from the packaged runtime.
"""

import argparse
import base64
import json
import os
import sys
import tempfile
import threading
import textwrap
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch


class MockHandler(BaseHTTPRequestHandler):
    health_status = 200
    model_id = "paddlevl"

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path == "/health":
            body = {"status": "ok"}
            status = self.health_status
        elif self.path == "/v1/models":
            body = {"data": [{"id": self.model_id}]}
            status = 200
        else:
            body = {"error": "not found"}
            status = 404
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


ALL_FLAGS = {
    "model": True, "imageMaxTokens": True, "alias": True, "batch": True,
    "ubatch": True, "flashAttention": True, "mmprojOffload": True,
    "noMmprojOffload": True, "kvOffload": True, "noKvOffload": True,
    "cacheKey": True, "cacheValue": True, "sleepIdle": True, "jinja": True,
    "cors": True, "noUi": True, "temperature": True, "gpuLayers": True,
    "special": True, "context": True, "parallel": True, "predict": True, "port": True,
}


def _binary_info(flags=None):
    return {"path": "mock-llama-server", "flags": dict(ALL_FLAGS if flags is None else flags)}


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def test_probe_service(checks, host, port):
    base_url = f"http://{host}:{port}/v1"
    _assert(checks.probeService(base_url, 5)["ok"], "healthy paddlevl server must be accepted")
    MockHandler.health_status = 503
    loading = checks.probeService(base_url, 5)
    _assert(not loading["ok"] and loading["details"].get("loading") and loading["errorType"] == "MODEL_NOT_READY", "503 must mean model loading")
    MockHandler.health_status = 200
    MockHandler.model_id = "another-model"
    mismatch = checks.probeService(base_url, 5)
    _assert(not mismatch["ok"] and mismatch["errorType"] == "MODEL_ALIAS_MISMATCH" and "模型" in mismatch["message"], "wrong model alias must be rejected without exposing implementation alias")
    MockHandler.model_id = "paddlevl"


def test_endpoint_defaults(checks):
    manifest = json.loads(Path(checks.MANIFEST_PATH).read_text(encoding="utf-8"))
    port_option = manifest["services"][0]["config"]["service"]["port"]
    _assert(port_option["default"] == 9931, "设置默认端口与 llama.cpp 保持一致")
    for config in ({"service.host": "127.0.0.1"}, checks.mergeArgd({})):
        _assert(checks.endpoint(config) == ("127.0.0.1", 9931, "http://127.0.0.1:9931/v1"), "省略端口时统一使用新默认值")
    _assert(checks.endpoint({"service.host": "http://[::1]"}) == ("127.0.0.1", 9931, "http://127.0.0.1:9931/v1"), "旧 host 不参与执行；内置运行时固定 IPv4 回环")
    for explicit in (8080, 18872, 65535):
        config = checks.mergeArgd({"service.port": explicit})
        _assert(checks.endpoint(config)[1] == explicit, "保留用户明确设置的端口，不强制迁移")


def test_error_types(checks):
    try:
        checks.requireFile({}, "service.mainModel", "主 GGUF", {".gguf"})
    except checks.CheckError as exc:
        result = checks.fail(exc.message, **exc.details)
        _assert(result["errorType"] == "MODEL_FILE_MISSING", "CheckError type survives conversion to a service result")
    else:
        raise AssertionError("missing model file was accepted")


def test_runtime(checks, RuntimeService, config, temp):
    executable = temp / "llama & server 中文.exe"
    main_model = temp / "主 & 模型.gguf"
    mmproj = temp / "视觉 投影 & mmproj.gguf"
    for path in (executable, main_model, mmproj):
        path.write_bytes(b"test")
    runtime = {
        **config,
        "service.mode": "managed",
        "service.serverExecutable": str(executable),
        "service.mainModel": str(main_model),
        "service.mmprojModel": str(mmproj),
        "resources.gpuLayers": "all",
        "resources.contextLength": 8192,
        "resources.maxOutputTokens": 4096,
        "resources.slots": 1,
        "resources.batchSize": 2048,
        "resources.ubatchSize": 512,
        "resources.imageMaxTokens": 2048,
        "resources.flashAttention": "auto",
        "resources.mmprojOffload": True,
        "resources.kvOffload": True,
        "resources.cacheKeyType": "f16",
        "resources.cacheValueType": "f16",
        "lifecycle.autoStopMode": "never",
    }
    service = RuntimeService()
    with patch.object(checks, "checkGguf"), patch.object(checks, "checkServerBinary", return_value=_binary_info()):
        command, command_host, command_port, flags, warnings = service.buildCommand(runtime)
    _assert((command_host, command_port) == (config["service.host"], config["service.port"]), "host and port are preserved")
    without_port = {key: value for key, value in runtime.items() if key != "service.port"}
    with patch.object(checks, "checkGguf"), patch.object(checks, "checkServerBinary", return_value=_binary_info()):
        default_command, _, default_port, _, _ = service.buildCommand(without_port)
    _assert(default_port == 9931 and default_command[default_command.index("--port") + 1] == "9931", "托管启动显式传入新默认端口")
    _assert(command[0] == str(executable) and str(main_model) in command and str(mmproj) in command, "paths with spaces and Chinese stay argv items")
    for value in ("--alias", "paddlevl", "--image-max-tokens", "2048", "--special", "--temp", "0", "-np", "1", "-fa", "auto"):
        _assert(value in command, f"default argv contains {value}")
    _assert(not warnings and flags["imageMaxTokens"], "fully supported server gets no optional warning")

    optional_flags = dict(ALL_FLAGS)
    optional_flags.update({"flashAttention": False, "mmprojOffload": False, "kvOffload": False, "jinja": False, "cors": False, "noUi": False})
    with patch.object(checks, "checkGguf"), patch.object(checks, "checkServerBinary", return_value=_binary_info(optional_flags)):
        _, _, _, _, omitted = service.buildCommand(runtime)
    _assert(len(omitted) >= 6, "unsupported optional flags are reported and omitted")

    missing_tokens = dict(ALL_FLAGS)
    missing_tokens["imageMaxTokens"] = False
    with patch.object(checks, "checkGguf"), patch.object(checks, "checkServerBinary", return_value=_binary_info(missing_tokens)):
        try:
            service.buildCommand(runtime)
        except checks.CheckError as exc:
            _assert(exc.code == 902 and "image-max-tokens" in exc.message, "spotting capability gate is explicit")
        else:
            raise AssertionError("unsupported image-max-tokens did not fail closed")

    missing_special = dict(ALL_FLAGS)
    missing_special["special"] = False
    with patch.object(checks, "checkGguf"), patch.object(checks, "checkServerBinary", return_value=_binary_info(missing_special)):
        try:
            service.buildCommand(runtime)
        except checks.CheckError as exc:
            _assert(exc.code == 902 and "--special" in exc.message, "coordinate-token support fails closed")
        else:
            raise AssertionError("unsupported --special did not fail closed")

    # A matching externally-owned process is reusable but never terminated.
    external = RuntimeService()
    reuse = external.ensure_ready({**config, "service.mode": "existing"})
    _assert(reuse["ok"] and external.get_status()["external"], "matching existing server is reused")
    _assert(external.stop_owned()["ok"] and external.get_status()["external"], "external process is never adopted or stopped")
    external._leases.add("active")
    _assert(not external.configure({**config, "service.port": config["service.port"] + 1})["ok"], "external-service lease blocks a configuration change")
    external._leases.clear()

    conflict = RuntimeService()
    with patch.object(checks, "probeService", return_value=checks.fail("not llama")), \
         patch.object(checks, "isPortBusy", return_value=True):
        result = conflict.ensure_ready({**config, "service.mode": "managed"})
    _assert(result["errorType"] == "PORT_CONFLICT", "occupied non-llama port has a stable type")

    # /health=503 is a loading state, not an invitation to kill the process
    # or misreport a port conflict.  Once its alias appears, reuse it.
    MockHandler.health_status = 503
    threading.Timer(0.15, lambda: setattr(MockHandler, "health_status", 200)).start()
    loading_external = RuntimeService()
    loading_reuse = loading_external.ensure_ready({**config, "service.mode": "existing"})
    _assert(loading_reuse["ok"] and loading_external.get_status()["external"], "503-to-ready external service is reused")

    # Startup does not use a shell after an executable was successfully created.
    process = Mock()
    process.poll.return_value = None
    owned = RuntimeService()
    with patch.object(owned, "buildCommand", return_value=(command, command_host, command_port, ALL_FLAGS, [])), \
         patch.object(checks, "probeService", side_effect=[checks.fail("not ready"), checks.ok("ready", model="paddlevl")]), \
         patch.object(checks, "isPortBusy", return_value=False), \
         patch("subprocess.Popen", return_value=process) as popen:
        started = owned.ensure_ready(runtime)
        _assert(started["ok"], "managed server can transition to ready")
        _assert(popen.call_args.kwargs["shell"] is False, "direct argv launch is shell-free")
        _assert(popen.call_count == 1, "a started process never triggers a shell retry")
        _assert(owned.stop_owned()["ok"], "owned process can be stopped")
        process.terminate.assert_called_once()

    exited = RuntimeService()
    dead_process = Mock()
    dead_process.poll.return_value = 17
    dead_process.returncode = 17
    exited._process = dead_process
    with patch.object(exited, "_logTail", return_value="GPU initialization failed"):
        early_exit = exited._waitReady("http://127.0.0.1:1/v1", 1)
    _assert(early_exit["errorType"] == "SERVER_EXITED", "early server exit has a stable type")
    _assert(RuntimeService()._waitReady("http://127.0.0.1:1/v1", 0)["errorType"] == "SERVER_START_TIMEOUT", "startup timeout has a stable type")

    # Shell fallbacks exist solely for an OS-level launch failure.  The first
    # retry is pwsh 7 on Windows and every invocation still keeps shell=False.
    fallback = RuntimeService()
    fallback._logFile = tempfile.NamedTemporaryFile(delete=True)
    shell_process = Mock()
    with patch("subprocess.Popen", side_effect=[OSError("direct launcher failure"), shell_process]) as popen, \
         patch("shutil.which", side_effect=lambda name: "pwsh.exe" if name == "pwsh" else None):
        _assert(fallback._spawn(command) is shell_process, "pwsh fallback starts after direct launcher failure")
        _assert(popen.call_args_list[0].args[0] == command, "direct argv is always attempted first")
        pwsh_command = popen.call_args_list[1].args[0]
        _assert(pwsh_command[0] == "pwsh.exe", "pwsh is the first Windows fallback")
        _assert("-EncodedCommand" in pwsh_command, "pwsh fallback keeps argv out of PowerShell parser")
        script = base64.b64decode(pwsh_command[-1]).decode("utf-16le")
        _assert(str(executable) in script and str(main_model) in script, "encoded pwsh fallback preserves special paths")
        _assert(all(call.kwargs["shell"] is False for call in popen.call_args_list), "fallbacks do not invoke a shell=True command")
    fallback._closeLog()

    redacted = RuntimeService._redactLog(
        "load C:\\Users\\ROG\\私人模型\\model.gguf api_key=secret-value"
    )
    _assert("ROG" not in redacted and "secret-value" not in redacted, "server failure log tail is redacted")

    # Service requests are FIFO even if an advanced user exposes more slots.
    queue_service = RuntimeService()
    with patch.object(queue_service, "ensure_ready", return_value=checks.ok("ready")):
        first = queue_service.acquire("ocr", {"resources.timeoutSeconds": 5})
        acquired = []
        worker = threading.Thread(target=lambda: acquired.append(queue_service.acquire("document", {"resources.timeoutSeconds": 5})))
        worker.start()
        time.sleep(0.1)
        _assert(queue_service.get_status()["waitingRequests"] == 1, "second request waits behind active lease")
        queue_service.release(first)
        worker.join(2)
        _assert(len(acquired) == 1, "waiting request gets lease after release")
        queue_service.release(acquired[0])

    queue_service._fingerprint = (("service.port", "8080"),)
    queue_service._config = {"service.port": 8080}
    queue_service._leases.add("active")
    changed = queue_service.configure({"service.port": 8090})
    _assert(not changed["ok"], "restart-causing configuration change is blocked by lease")

    waiting_service = RuntimeService()
    waiting_service._fingerprint = (("service.port", "8080"),)
    waiting_service._config = {"service.port": 8080}
    waiting_service._leaseQueue.append("document-waiting")
    with patch.object(checks, "probeService", side_effect=AssertionError("busy config must not probe")):
        waiting_changed = waiting_service.ensure_ready({"service.port": 8090})
    _assert(not waiting_changed["ok"] and waiting_changed["errorType"] == "SERVICE_BUSY", "waiting request blocks restart-causing change")
    _assert(waiting_service._config["service.port"] == 8080, "rejected busy config does not replace active config")
    invalid_busy = waiting_service.ensure_ready({"service.port": 8090, "resources.startupTimeoutSeconds": "invalid"})
    _assert(not invalid_busy["ok"] and invalid_busy["errorType"] == "SERVICE_BUSY", "busy restart change wins over new config validation")


def test_worker_helpers(plugin_root):
    from win32_PaddleOCRVL_1_6 import worker

    pipeline = worker._client_kwargs({}, 42)
    client = pipeline["SubModules"]["VLRecognition"]["genai_config"]["client_kwargs"]
    _assert(
        client == {"timeout": 42, "model_name": "paddlevl", "max_retries": 0},
        "document client has one explicit retry owner",
    )

    raw = {
        "spotting_res": {
            "rec_texts": ["小图文字"],
            "rec_polys": [[[20, 10], [80, 10], [80, 40], [20, 40]]],
        }
    }
    with patch.object(worker, "_source_size", return_value=(100, 50)):
        data, warnings, transform = worker._extract_spotting(raw, plugin_root / "fixture.png")
    _assert(not warnings and transform["upscaled"], "small image follows official upscale rule")
    _assert(transform["coordinateScale"] == 1.0, "upscaled inference already returns original-image coordinates")
    with patch.object(worker, "_source_size", return_value=(1600, 900)):
        large, _, large_transform = worker._extract_spotting(raw, plugin_root / "fixture.png")
    _assert(large[0]["polygon"] == raw["spotting_res"]["rec_polys"][0] and not large_transform["upscaled"], "large images keep original coordinates too")
    _assert(data[0]["polygon"] == raw["spotting_res"]["rec_polys"][0] and data[0]["score"] is None, "PaddleX original-image coordinates must not be halved; confidence remains null")
    _assert(
        worker._box_from_points([[0, 0], [4, 0], [6, 3], [3, 6], [0, 4]])
        == [[0, 0], [6, 0], [6, 6], [0, 6]],
        "non-quadrilateral polygon gets a compatible enclosing box",
    )
    try:
        worker._extract_spotting({"spotting_res": {"rec_texts": ["x"], "rec_polys": []}}, plugin_root / "fixture.png")
    except ValueError as exc:
        _assert("INVALID_MODEL_OUTPUT" in str(exc), "mismatched spotting output fails closed")
    else:
        raise AssertionError("mismatched spotting output was accepted")

    blocks, warnings = worker._document_blocks({
        "width": 100, "height": 50,
        "parsing_res_list": [
            {"block_label": "table", "block_content": "<table><tr><td>A</td></tr></table></s>", "block_bbox": [0, 0, 20, 10], "block_id": 7, "block_order": 2, "group_id": 4, "global_block_id": 19, "global_group_id": 18},
            {"block_label": "formula", "block_content": "x^2", "block_bbox": [0, 20, 20, 30], "block_id": 8, "block_order": 1, "group_id": 5},
        ]
    }, 1)
    _assert(not warnings and blocks[0]["category"] == "formula", "official block_order is preserved for reading order")
    table = blocks[1]
    _assert(table["id"] == 7 and table["globalId"] == 19 and table["table"]["html"].startswith("<table"), "table IDs and HTML survive normalization")
    _assert("</s>" not in table["content"] and "</s>" not in table["table"]["html"], "model stop markers are removed from exported content")
    mixed, warnings = worker._document_blocks({
        "width": 100, "height": 50,
        "parsing_res_list": [
            {"block_label": "title", "block_content": "T", "block_bbox": [0, 0, 20, 10], "block_id": 0, "block_order": 1},
            {"block_label": "table", "block_content": "<table></table>", "block_bbox": [0, 10, 20, 20], "block_id": 1, "block_order": None},
            {"block_label": "text", "block_content": "B", "block_bbox": [0, 20, 20, 30], "block_id": 2, "block_order": 2},
        ]
    }, 1)
    _assert(
        not warnings and [item["id"] for item in mixed] == [0, 1, 2],
        "null-order layout blocks keep their official parsing_res_list position",
    )

    class Image:
        def save(self, target, format=None):
            Path(target).write_bytes(b"png")

    with tempfile.TemporaryDirectory() as raw_temp, patch.object(worker, "PLUGIN_DIR", Path(raw_temp)):
        markdown, assets, warnings = worker._materialize_assets(
            "![image](image.jpg)", [{"path": "image.jpg", "img": Image()}], "list-assets"
        )
        _assert(
            not warnings and len(assets) == 1 and "file:" in markdown,
            "list-form Markdown assets are materialized",
        )
        asset = Path(assets[0]["path"])
        _assert(asset.suffix == ".png" and asset.read_bytes() == b"png", "PNG assets never retain a misleading source suffix")


def test_document_region_failure_isolation(plugin_root):
    from win32_PaddleOCRVL_1_6 import worker

    class Model:
        def __init__(self):
            self.calls = []

        def predict(self, payloads, **_kwargs):
            query = payloads[0]["query"]
            self.calls.append(query)
            if query == "bad":
                raise RuntimeError("HTTP 500")
            if query == "invalid":
                return [{}]
            return [{"result": query}]

    class Engine:
        def __init__(self):
            self.vl_rec_model = Model()

        def _paddleocr_vl_run_vl_recognition_batches(self, *_args, **_kwargs):
            raise AssertionError("official batch method should be replaced")

    engine = Engine()
    worker._partial_failures.clear()
    worker._install_document_failure_isolation(engine)
    batch = {
        (112896, 1003520): {
            "images": [object(), object(), object()],
            "queries": ["ok", "bad", "invalid"],
            "vlm_block_ids": [(0, 0), (0, 1), (0, 2)],
        }
    }
    engine._paddleocr_vl_run_vl_recognition_batches(batch, False, {"temperature": 0})
    results = batch[(112896, 1003520)]["vlm_results"]
    _assert(results[0]["result"] == "ok", "healthy document region survives isolation")
    _assert(results[1]["result"] == "" and results[2]["result"] == "", "failed regions become empty placeholders")
    _assert(len(worker._partial_failures) == 2, "partial failures are recorded per region")
    _assert(all(item["retries"] == 1 for item in worker._partial_failures), "region retry is bounded to one retry")
    _assert(engine.vl_rec_model.calls.count("bad") == 2, "HTTP failure is retried once")
    class Nested:
        def __init__(self):
            self.paddlex_pipeline = Engine()

        def __getattr__(self, name):
            return getattr(self.paddlex_pipeline, name)

    nested = Nested()
    worker._install_document_failure_isolation(nested)
    nested_batch = {
        (112896, 1003520): {
            "images": [object()], "queries": ["nested"], "vlm_block_ids": [(0, 0)]
        }
    }
    nested.paddlex_pipeline._paddleocr_vl_run_vl_recognition_batches(
        nested_batch, False, {}
    )
    _assert(
        nested_batch[(112896, 1003520)]["vlm_results"][0]["result"] == "nested",
        "nested PaddleOCRVL pipeline receives the isolation wrapper",
    )
    class AutoParallel:
        def __init__(self):
            self._pipeline = Engine()

        def __getattr__(self, name):
            return getattr(self._pipeline, name)

    auto = Nested()
    auto.paddlex_pipeline = AutoParallel()
    worker._install_document_failure_isolation(auto)
    auto_batch = {
        (112896, 1003520): {
            "images": [object()], "queries": ["auto"], "vlm_block_ids": [(0, 0)]
        }
    }
    auto.paddlex_pipeline._pipeline._paddleocr_vl_run_vl_recognition_batches(
        auto_batch, False, {}
    )
    _assert(
        auto_batch[(112896, 1003520)]["vlm_results"][0]["result"] == "auto",
        "AutoParallel inner execution pipeline receives the isolation wrapper",
    )
    worker._partial_failures.clear()


def test_spotting_call_uses_official_budget(plugin_root):
    """The no-layout OCR call must preserve Paddle's spotting contract."""
    from win32_PaddleOCRVL_1_6 import worker

    class Result:
        json = {"spotting_res": {"rec_texts": ["x"], "rec_polys": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}}

    class Engine:
        kwargs = None

        def predict(self, *_args, **kwargs):
            self.kwargs = kwargs
            return [Result()]

    with tempfile.TemporaryDirectory() as raw_temp:
        source = Path(raw_temp) / "spotting-fixture.png"
        from PIL import Image
        Image.new("RGB", (100, 50), "white").save(source)
        engine = Engine()
        old_engine, old_mode = worker._engine, worker._mode
        worker._engine, worker._mode = engine, "spotting"
        try:
            with patch.object(worker, "_extract_spotting", return_value=([
                {"text": "x", "box": [[0, 0]] * 4, "polygon": [[0, 0], [1, 0], [1, 1]], "score": None}
            ], [], {})):
                result = worker._spotting({"path": str(source), "options": {}})
        finally:
            worker._engine, worker._mode = old_engine, old_mode
    _assert(result["code"] == 100, "spotting mock returns a valid OCR result")
    _assert(engine.kwargs["use_layout_detection"] is False and engine.kwargs["prompt_label"] == "spotting", "text OCR uses the official no-layout spotting flow")
    _assert(engine.kwargs["min_pixels"] == worker.SPOTTING_MIN_PIXELS and engine.kwargs["max_pixels"] == worker.SPOTTING_MAX_PIXELS, "spotting call keeps the 1605632-pixel budget")
    _assert(engine.kwargs["temperature"] == 0 and "top_p" not in engine.kwargs, "spotting is deterministic without extra sampling")


def test_document_uses_restructured_pages(plugin_root):
    """DocumentResultV1 must take IDs/order from Paddle's restructured pages."""
    from win32_PaddleOCRVL_1_6 import worker

    class Result:
        def __init__(self, raw, markdown):
            self.json = raw
            self.markdown = markdown

    raw_page = Result(
        {"page_index": 0, "width": 100, "height": 50, "parsing_res_list": []},
        {"markdown_texts": ""},
    )
    restructured_page = Result(
        {
            "page_index": 0,
            "width": 100,
            "height": 50,
            "parsing_res_list": [{
                "block_label": "table",
                "block_content": "<table><tr><td>A</td></tr></table>",
                "block_bbox": [1, 2, 30, 20],
                "block_id": 3,
                "global_block_id": 41,
                "group_id": 5,
                "global_group_id": 40,
                "block_order": 0,
            }],
        },
        {"markdown_texts": "| A |"},
    )

    class Engine:
        def predict(self, *_args, **_kwargs):
            return [raw_page]

        def restructure_pages(self, pages, **kwargs):
            _assert(pages == [raw_page], "official predictor pages pass to restructure")
            _assert(kwargs["concatenate_pages"] is False, "page records stay page-scoped")
            return [restructured_page]

    with tempfile.TemporaryDirectory() as raw_temp:
        source = Path(raw_temp) / "document-fixture.png"
        from PIL import Image
        Image.new("RGB", (100, 50), "white").save(source)
        old_engine, old_mode = worker._engine, worker._mode
        worker._engine, worker._mode = Engine(), "document"
        try:
            result = worker._document({"path": str(source), "options": {"concatenatePages": True}})
        finally:
            worker._engine, worker._mode = old_engine, old_mode
    block = result["data"]["pages"][0]["blocks"][0]
    _assert(block["globalId"] == 41 and block["globalGroupId"] == 40, "global IDs come from official restructuring")
    _assert(block["table"]["html"].startswith("<table"), "table HTML is retained in DocumentResultV1")


def test_worker_client_and_page(plugin_root):
    from win32_PaddleOCRVL_1_6.api import _WorkerClient
    from win32_PaddleOCRVL_1_6.page import StructuredDocumentPage

    page = StructuredDocumentPage("immediate", None)
    class ImmediateMission:
        def addMissionList(self, info, items):
            info["onEnd"](info,"[Success]")
            return info["_reserved_msnID"]
    with patch.object(page,"_missionDocument",return_value=ImmediateMission()):
        first = page.parseDocuments(["sample.png"])
        _assert(first and not page._missionId, "immediate completion clears page state without an ID race")
        _assert(not page.parseDocuments(["sample.png"]).startswith("[Error]"), "completed page can immediately retry")

    worker_script = Path(tempfile.mkdtemp()) / "worker.py"
    worker_script.write_text(textwrap.dedent("""
        import json, sys, time
        def reply(value): print(json.dumps(value), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('op') == 'start': reply({'ok': True, 'status': 'loaded'})
            elif request.get('op') == 'probe': reply({'ok': True, 'result': {'paddleocr': '3.7.0', 'cudaCompiled': True}}); break
            elif request.get('op') in ('document', 'spotting'):
                reply({'ok': True, 'result': {'code': 100, 'text': 'test', 'markdown': '# test', 'data': {'pages': [], 'blocks': [], 'assets': []}}})
            elif request.get('op') == 'hang': time.sleep(5)
            elif request.get('op') == 'stop': reply({'ok': True, 'status': 'stopped'}); break
    """), encoding="utf-8")
    with patch("win32_PaddleOCRVL_1_6.api.WORKER", worker_script):
        client = _WorkerClient()
        probe = client.probe({'requireCuda': True, 'device': 'gpu:0'}, 3)
        _assert(probe['cudaCompiled'] is True, "worker dependency probe uses the isolated worker interpreter")
        _assert(client.start({"mode": "document"}, 3)["ok"], "worker starts with mode")
        structured = client.request({"op": "document", "path": "sample.pdf"}, 3)
        _assert(structured["ok"] and structured["result"]["markdown"] == "# test", "document worker result travels through IPC")
        try:
            client.request({"op": "hang"}, 0.1)
        except TimeoutError:
            pass
        else:
            raise AssertionError("worker timeout did not fail closed")
        _assert(client.process is None, "timed-out worker is stopped")

    with tempfile.TemporaryDirectory() as output:
        asset = Path(output) / "asset.png"
        asset.write_bytes(b"png")
        result = {
            "code": 100, "text": "plain", "markdown": "![img](" + asset.resolve().as_uri() + ")",
            "data": {
                "pages": [{
                    "page": 1,
                    "markdown": "![img](" + asset.resolve().as_uri() + ")",
                    "resources": [{"id": "asset-1", "name": "asset.png", "path": str(asset), "uri": asset.resolve().as_uri(), "source": "asset.png"}],
                }],
                "blocks": [],
                "assets": [{"id": "asset-1", "name": "asset.png", "path": str(asset), "uri": asset.resolve().as_uri(), "source": "asset.png"}],
            },
            "metadata": {"sourceStem": "测试 文档"},
        }
        page = StructuredDocumentPage("test", None)
        exported = page.exportResult(result, output)
        _assert(exported["ok"], "DocumentResultV1 exports atomically")
        for key in ("markdown", "json"):
            _assert(Path(exported["details"][key]).is_file(), f"{key} export exists")
        _assert("txt" not in exported["details"], "UI all export has no duplicate TXT")
        _assert("测试 文档_assets/asset.png" in Path(exported["details"]["markdown"]).read_text(encoding="utf-8"), "markdown asset URI becomes relative resource path")
        compatible = page.exportResult(result, output, "txt")
        _assert(compatible["ok"], "explicit TXT compatibility export succeeds")
        compatible_path = Path(compatible["details"]["txt"])
        _assert(compatible_path.name.endswith(".md.txt"), "TXT compatibility uses .md.txt")
        _assert(compatible_path.read_bytes() == Path(exported["details"]["markdown"]).read_bytes(), "TXT compatibility preserves exactly the same Markdown")
        exported_json = json.loads(Path(exported["details"]["json"]).read_text(encoding="utf-8"))
        page = exported_json["data"]["pages"][0]
        _assert("file:" not in page["markdown"] and page["resources"][0]["uri"].startswith("测试 文档_assets/"), "JSON page Markdown and resources remain portable after asset export")


def test_manifest_and_qml(plugin_root):
    manifest = json.loads((plugin_root / "umi-plugin.json").read_text(encoding="utf-8"))
    # Source runs have VERSION beside the plugin; Build-Plugin.ps1 runs this
    # same test against ``staging/UmiOCR-data/plugins/<id>`` where VERSION is
    # delivered under ``staging/source`` instead.  Resolve both layouts so the
    # packaged-tree regression checks the actual staged plugin rather than
    # depending on a file that is intentionally not installed next to it.
    version_candidates = [
        plugin_root.parent / "VERSION",
        plugin_root.parent.parent.parent / "source" / "VERSION",
        plugin_root.parent.parent.parent / "LICENSES" / "plugins" / plugin_root.name / "source-records" / "VERSION",
        plugin_root.parent.parent.parent / "VERSION",
    ]
    version_file = next((path for path in version_candidates if path.is_file()), None)
    _assert(
        version_file is not None and manifest["version"] == version_file.read_text(encoding="utf-8").strip(),
        "manifest and package version stay aligned",
    )
    caps = {item["type"]: item for item in manifest["capabilities"]}
    _assert(set(caps) == {"ocr", "document"}, "manifest registers isolated OCR and document capabilities")
    _assert(caps["ocr"]["meta"]["serviceRefs"] == ["paddlevl"] == caps["document"]["meta"]["serviceRefs"], "both capabilities share exactly one service")
    if (plugin_root / "StructuredDocument.qml").is_file():
        qml = (plugin_root / "StructuredDocument.qml").read_text(encoding="utf-8")
    else:
        from PySide6.QtCore import QResource, QFile
        assert QResource.registerResource(str(plugin_root / "resources.rcc"))
        resource = QFile(":/plugins/"+plugin_root.name+"/StructuredDocument.qml")
        assert resource.open(QFile.ReadOnly)
        qml = bytes(resource.readAll()).decode("utf-8")
    for text in ("启动服务", "停止服务", "testConnection", "startService", "stopService"):
        _assert(text not in qml, f"document page has no private service control: {text}")
    for text in ("打开", "粘贴", "开始解析", "取消任务", "复制", "导出", "DocPreviewConnector", "parseDocuments"):
        _assert(text in qml, f"document page exposes required UI action: {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plugin_root", type=Path)
    plugin_root = parser.parse_args().plugin_root.resolve()
    if plugin_root.name != "win32_PaddleOCRVL_1_6" or not plugin_root.is_dir():
        raise FileNotFoundError(plugin_root)
    sys.path.insert(0, str(plugin_root.parent))
    sys.path.insert(0, str(plugin_root.parent.parent / "site-packages"))
    host_packages = Path(sys.executable).parent.parent / "site-packages"
    if host_packages.is_dir():
        sys.path.append(str(host_packages))

    from win32_PaddleOCRVL_1_6 import checks
    from win32_PaddleOCRVL_1_6.runtime import RuntimeService

    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    config = {"service.host": host, "service.port": port, "resources.timeoutSeconds": 5, "resources.startupTimeoutSeconds": 10}
    try:
        test_probe_service(checks, host, port)
        test_endpoint_defaults(checks)
        test_error_types(checks)
        # 外部服务复用/可编辑 EXE 的旧断言已由内置运行时新契约取代。
        import importlib.util
        suite_path = Path(__file__).with_name("test_model_settings_vl.py")
        spec = importlib.util.spec_from_file_location("test_model_settings_vl", suite_path)
        suite = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(suite)
        with tempfile.TemporaryDirectory() as raw_temp:
            _assert(suite.run(plugin_root, Path(raw_temp)), "内置运行时与预算定向回归通过")
        from win32_PaddleOCRVL_1_6 import worker
        with tempfile.TemporaryDirectory() as raw_temp, patch.object(worker, "_config", {"tempRoot": raw_temp}), patch.object(worker, "_progress"):
            test_worker_helpers(plugin_root)
            test_document_region_failure_isolation(plugin_root)
            test_spotting_call_uses_official_budget(plugin_root)
            test_document_uses_restructured_pages(plugin_root)
            test_worker_client_and_page(plugin_root)
        test_manifest_and_qml(plugin_root)
    finally:
        server.shutdown()
        server.server_close()
    print("PaddleOCR-VL 1.6 dual-mode plugin mock regression passed.")


if __name__ == "__main__":
    main()

"""VL 定向 mock 回归：不加载 Paddle/GPU，不读取真实桌面数据。"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class VLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from win32_PaddleOCRVL_1_6 import checks, runtime, worker, budgets
        cls.checks, cls.runtime, cls.worker, cls.budgets = checks, runtime, worker, budgets

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=WORK_ROOT)
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(setattr, self.worker, "_task_deadline", float("inf"))
        self.addCleanup(setattr, self.worker, "_request_timeout", 120)
        self.addCleanup(self.worker._generation_signals.clear)
        self.addCleanup(setattr, self.worker, "_asset_output_root", None)
        self.worker._task_deadline = float("inf")
        self.worker._asset_output_bytes = self.worker._asset_output_count = 0
        self.worker._asset_budget_exhausted = False

    def bundle(self):
        plugin = self.root / "plugin"
        runtime = plugin / "resources" / "llama.cpp"
        runtime.mkdir(parents=True)
        files = {"llama-server.exe": b"exe", "ggml-cuda.dll": b"dll", "licenses/LICENSE": b"license"}
        for name, content in files.items():
            path = runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        manifest = {"schemaVersion": 1, "identity": {"commit": "a" * 40, "cudaToolkit": "13.4.92", "architectures": ["75", "89"], "buildOptions": {"GGML_CUDA": "ON"}},
            "files": [{"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()} for name, content in files.items()]}
        (runtime / "runtime-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return plugin, runtime, manifest

    def image(self, size=(80, 40), name="input.png"):
        from PIL import Image
        target = self.root / name
        Image.new("RGB", size, "white").save(target)
        return target

    def test_all_user_chinese_messages_are_stable_templates(self):
        import ast
        import re
        han = re.compile(r"[㐀-鿿]")
        residual = []
        for path in Path(self.checks.PLUGIN_DIR).glob("*.py"):
            if path.name == "runtime_archive_bootstrap.py":
                continue  # 容器完整性诊断先于翻译器加载；由独立归档完整性回归覆盖。
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                    docstrings.add(id(node.body[0].value))
            def visit(node):
                if id(node) in docstrings:
                    return
                if isinstance(node, ast.FunctionDef) and node.name in ("tr", "_error_type"):
                    return  # 翻译器自身及错误分类匹配词不是用户文案
                if isinstance(node, ast.Call) and (isinstance(node.func, ast.Name) and node.func.id == "tr" or isinstance(node.func, ast.Attribute) and node.func.attr == "tr"):
                    if node.args and isinstance(node.args[0], ast.JoinedStr):
                        residual.append((path.name,node.lineno,"dynamic tr template"))
                    for keyword in node.keywords:
                        visit(keyword.value)
                    return
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and han.search(node.value):
                    residual.append((path.name,node.lineno,node.value[:100]))
                for child in ast.iter_child_nodes(node):
                    visit(child)
            visit(tree)
        self.assertEqual(residual, [])

    def test_named_templates_translate_before_interpolation(self):
        from win32_PaddleOCRVL_1_6.page import StructuredDocumentPage
        templates = {"[Error] 没有选择任何文档。":"[Error] No document selected.", "{key} 必须是整数。":"{key} must be an integer."}
        fake = SimpleNamespace(LangCode="en", Translator=lambda *args:lambda message:templates.get(message,message))
        with patch.dict(sys.modules,{"plugin_i18n":fake,"py_src.imports.plugin_i18n":fake}), patch.dict(self.checks._translationCache,{},clear=True):
            self.assertEqual(StructuredDocumentPage("test",None).parseDocuments([]),"[Error] No document selected.")
            with self.assertRaises(self.checks.CheckError) as caught:
                self.checks.timeoutSeconds({"resources.timeoutSeconds":"invalid"})
            self.assertEqual(caught.exception.message,"resources.timeoutSeconds must be an integer.")
            self.assertEqual(caught.exception.errorType,"INVALID_CONFIGURATION")
            self.assertEqual(self.worker._error_type("INVALID_MODEL_OUTPUT: detail"),"INVALID_MODEL_OUTPUT")

    def test_plain_host_language_wins_over_duplicate_canonical_module(self):
        from win32_PaddleOCRVL_1_6.api import _WorkerClient
        plain = SimpleNamespace(LangCode="ta", Translator=lambda *args:lambda text:"TA:" + text)
        canonical = SimpleNamespace(LangCode="", Translator=lambda *args:lambda text:"WRONG:" + text)
        with patch.dict(sys.modules,{"plugin_i18n":plain,"py_src.imports.plugin_i18n":canonical}), patch.dict(self.checks._translationCache,{},clear=True):
            self.assertEqual(self.checks.tr("message"),"TA:message")
            self.assertEqual(_WorkerClient._environment()["UMI_PLUGIN_LANGUAGE"],"ta")
            plain.LangCode = "ja_JP"
            self.assertEqual(_WorkerClient._environment()["UMI_PLUGIN_LANGUAGE"],"ja_JP")
            self.assertEqual(canonical.LangCode,"")

    def test_manifest_keys_and_legacy(self):
        manifest = self.checks.loadManifest()
        defaults = self.checks.manifestDefaults()
        self.assertEqual(defaults["resources.contextLength"], 16384)
        service = manifest["services"][0]["config"]
        self.assertNotIn("gpuLayers", service["resources"])
        self.assertIn("PaddleOCR-VL-1.6-GGUF.gguf", service["service"]["mainModel"]["toolTip"])
        self.assertIn("PaddleOCR-VL-1.6-GGUF-mmproj.gguf", service["service"]["mmprojModel"]["toolTip"])
        self.assertEqual(service["resources"]["maxOutputTokens"]["default"], 4096)
        self.assertIn("服务端最大输出 token", service["resources"]["maxOutputTokens"]["toolTip"])
        for key in self.checks.RETIRED_KEYS:
            self.assertNotIn(key, defaults)
            self.assertNotIn(key, self.checks.mergeArgd({key: "malicious"}))
        for value in ("all", "auto", 12, 0):
            self.assertNotIn("resources.gpuLayers", self.checks.mergeArgd({"resources.gpuLayers": value}))
        capabilities = {item["type"]: item for item in manifest["capabilities"]}
        ocr = capabilities["ocr"]
        self.assertEqual(ocr["meta"]["settingsOrder"], 20)
        self.assertIn("serviceRefs", ocr["meta"])
        self.assertIn("服务端最大输出 token", ocr["config"]["local_options"]["maxOutputTokens"]["toolTip"])
        document = capabilities["document"]
        device = document["config"]["global_options"]["pipeline"]["cudaDevice"]
        self.assertEqual(device["default"], "gpu:0")
        self.assertIs(type(device["default"]), str)
        self.assertEqual(device["deviceProvider"], "nvidia-cuda")
        self.assertIn("服务端最大输出 token", document["config"]["local_options"]["maxOutputTokens"]["toolTip"])
        self.assertEqual(self.checks.endpoint({"service.host": "0.0.0.0", "service.port": 18888}), ("127.0.0.1", 18888, "http://127.0.0.1:18888/v1"))
        self.assertEqual(self.checks.endpoint({})[1], 9931)

    def test_bundle_integrity_cache(self):
        plugin, runtime, _ = self.bundle()
        with patch.object(self.checks, "PLUGIN_DIR", str(plugin)), patch.object(self.checks.hashlib, "sha256", wraps=hashlib.sha256) as digest:
            first = self.checks.bundledRuntime()
            self.assertEqual(first["path"], str(runtime / "llama-server.exe"))
            first_calls = digest.call_count
            self.checks.bundledRuntime()
            self.assertEqual(digest.call_count - first_calls, 1, "第二次仅摘要小清单，不重读 EXE/DLL")
            (runtime / "ggml-cuda.dll").write_bytes(b"bad")
            with self.assertRaises(self.checks.CheckError):
                self.checks.bundledRuntime()

    def test_bundle_extra_dll_and_traversal(self):
        plugin, runtime, manifest = self.bundle()
        with patch.object(self.checks, "PLUGIN_DIR", str(plugin)):
            (runtime / "evil.dll").write_bytes(b"x")
            with self.assertRaises(self.checks.CheckError):
                self.checks.bundledRuntime()
            (runtime / "evil.dll").unlink()  # 只删除本测试新建的小文件
            for name in ("../escape.dll", "C:/escape.dll", "sub\\evil.dll", "llama-server.exe:stream"):
                changed = json.loads(json.dumps(manifest))
                changed["files"][0]["path"] = name
                (runtime / "runtime-manifest.json").write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(self.checks.CheckError):
                    self.checks.bundledRuntime()

    def test_bundle_reparse_point(self):
        plugin, runtime, _ = self.bundle()
        original = Path.lstat
        def mocked(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == runtime:
                return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
            return result
        with patch.object(self.checks, "PLUGIN_DIR", str(plugin)), patch.object(Path, "lstat", mocked):
            with self.assertRaises(self.checks.CheckError):
                self.checks.bundledRuntime()

    def test_joint_budget(self):
        default = self.checks.resourceBudget(self.checks.mergeArgd({}))
        self.assertEqual(default["contextLength"], 16384)
        self.assertEqual(default["maxOutputTokens"], 4096)
        for values in ({"resources.contextLength": 16384, "resources.slots": 3}, {"resources.contextLength": 8192, "resources.slots": 2}, {"maxOutputTokens": 5000}, {"resources.contextLength": 4096}, {"maxOutputTokens": float("nan")}, {"resources.slots": True}):
            with self.assertRaises(self.checks.CheckError, msg=str(values)):
                self.checks.resourceBudget(self.checks.mergeArgd(values))
        two_slots = self.checks.mergeArgd({"resources.contextLength": 16384, "resources.slots": 2})
        self.assertEqual(self.checks.resourceBudget(two_slots)["perSlotContext"], 8192)
        explicit_8192 = self.checks.resourceBudget(self.checks.mergeArgd({"resources.contextLength": 8192}))
        self.assertEqual(explicit_8192["contextLength"], 8192)
        self.assertEqual(explicit_8192["maxOutputTokens"], 4096)
        exact_limit = self.checks.mergeArgd({"resources.maxOutputTokens": 13824, "maxOutputTokens": 13824})
        self.assertEqual(self.checks.resourceBudget(exact_limit)["maxOutputTokens"], 13824)
        over_limit = self.checks.mergeArgd({"resources.maxOutputTokens": 13825, "maxOutputTokens": 13825})
        with self.assertRaises(self.checks.CheckError):
            self.checks.resourceBudget(over_limit)
        service_limited = self.checks.mergeArgd({"resources.maxOutputTokens": 8192, "maxOutputTokens": 8193})
        with self.assertRaises(self.checks.CheckError):
            self.checks.resourceBudget(service_limited)

    def test_retired_gpu_layers_cannot_change_command_or_fingerprint(self):
        service = self.runtime.RuntimeService()
        flags = {
            "alias": True, "special": True, "batch": True, "ubatch": True,
            "flashAttention": True, "mmprojOffload": True, "kvOffload": True,
            "cacheKey": True, "cacheValue": True, "jinja": True,
            "noUi": True, "verbosity": True,
        }
        values = ("all", "auto", 12, 0)
        with patch.object(self.checks, "bundledRuntime", return_value={"path": "llama-server.exe", "identity": {}, "manifestSha256": "test"}), \
                patch.object(self.checks, "requireFile", side_effect=lambda _argd, key, *_args, **_kwargs: key + ".gguf"), \
                patch.object(self.checks, "checkGguf"), \
                patch.object(self.checks, "checkServerBinary", return_value={"flags": flags}):
            commands = [service.buildCommand({"resources.gpuLayers": value})[0] for value in values]
        self.assertTrue(all(command == commands[0] for command in commands))
        self.assertEqual(commands[0][commands[0].index("-ngl") + 1], "all")
        with patch.object(self.checks, "bundledRuntimeStamp", return_value=("runtime", "stable")):
            fingerprints = [service._fingerprintFor({"resources.gpuLayers": value}) for value in values]
        self.assertTrue(all(value == fingerprints[0] for value in fingerprints))

    def test_external_port_never_reused(self):
        service = self.runtime.RuntimeService()
        with patch.object(self.checks, "bundledRuntime", return_value={}), patch.object(self.checks, "isPortBusy", return_value=True), patch.object(self.checks, "probeService") as probe, patch("subprocess.Popen") as popen:
            result = service.ensure_ready({"service.mode": "existing"})
        self.assertEqual(result["errorType"], "PORT_CONFLICT")
        probe.assert_not_called()
        popen.assert_not_called()
        self.assertFalse(service.get_status()["external"])

    def test_own_start_and_cleanup_gates(self):
        service = self.runtime.RuntimeService()
        process = Mock(pid=123)
        process.poll.return_value = None
        with patch.object(self.checks, "bundledRuntime", return_value={}), patch.object(self.checks, "isPortBusy", return_value=False), patch.object(self.checks, "probeService", return_value=self.checks.ok("ready")), patch.object(service, "buildCommand", return_value=(["fixed.exe"], "127.0.0.1", 9931, {}, [])), patch.object(service, "get_execution_info", return_value={"device":"GPU"}), patch.object(self.runtime, "tempRoot", return_value=self.root), patch("subprocess.Popen", return_value=process) as spawn:
            self.assertTrue(service.ensure_ready({})["ok"])
            self.assertFalse(spawn.call_args.kwargs["shell"])
            service._leaseQueue.append("queued")
            self.assertFalse(service.stop_owned()["ok"])
            self.assertFalse(service.maintain({"reason": "memory"})["ok"])
            process.terminate.assert_not_called()
            service._leaseQueue.clear()
            service._leases.add("active")
            self.assertFalse(service.release()["ok"])
            service.release("active")
            self.assertTrue(service.maintain({"reason": "idle"})["ok"])
            process.terminate.assert_called_once()

    def test_cuda_unconfirmed_never_ready(self):
        service = self.runtime.RuntimeService()
        process = Mock(pid=123)
        process.poll.return_value = None
        service._process = process
        with patch.object(self.checks, "probeService", return_value=self.checks.ok("ready")), patch.object(service, "get_execution_info", return_value={"device":"设备未确认"}):
            result = service._waitReady("http://127.0.0.1:9931/v1", 1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["errorType"],"GPU_INITIALIZATION_FAILED")
        process.terminate.assert_called_once()

    def test_fifo_and_queue_cancel(self):
        service = self.runtime.RuntimeService()
        cancelled = threading.Event()
        result = []
        with patch.object(service, "ensure_ready", return_value=self.checks.ok("ready")):
            first = service.acquire("ocr", {"resources.timeoutSeconds": 5})
            def queued():
                try:
                    service.acquire("document", {"resources.timeoutSeconds": 5}, cancel=cancelled)
                except self.checks.CheckError as exc:
                    result.append(exc.errorType)
            thread = threading.Thread(target=queued)
            thread.start()
            limit = time.monotonic() + 2
            while service.get_status()["waitingRequests"] != 1 and time.monotonic() < limit:
                threading.Event().wait(0.005)
            cancelled.set()
            thread.join(2)
            self.assertEqual(result, ["CANCELLED"])
            self.assertEqual(service.get_status()["waitingRequests"], 0)
            service.release(first)

    def test_tile_geometry_and_bounds(self):
        for width, height in ((80, 40), (1200, 7000), (5000, 700), (2300, 2400)):
            boxes = self.budgets.tile_boxes(width, height)
            self.assertLessEqual(len(boxes), 32)
            self.assertEqual(min(b[0] for b in boxes), 0)
            self.assertEqual(min(b[1] for b in boxes), 0)
            self.assertEqual(max(b[2] for b in boxes), width)
            self.assertEqual(max(b[3] for b in boxes), height)
            self.assertTrue(all((b[2]-b[0])*(b[3]-b[1]) <= self.worker.SPOTTING_MAX_PIXELS for b in boxes))
        with self.assertRaises(self.budgets.BudgetError):
            self.budgets.check_size(100000, 10)
        with self.assertRaises(self.budgets.BudgetError):
            self.budgets.tile_boxes(2600, 12000)

    def test_spotting_tiles_restore_and_single(self):
        from PIL import Image
        calls = []
        def predict(path, **kwargs):
            with Image.open(path) as image:
                calls.append((image.size, kwargs))
            return [{"spotting_res": {"rec_texts": ["x"], "rec_polys": [[[5, 5], [25, 5], [25, 20], [5, 20]]]}}]
        for size in ((80, 40), (1200, 3200)):
            calls.clear()
            source = self.image(size)
            with patch.object(self.worker, "_engine", SimpleNamespace(predict=predict)), patch.object(self.worker, "_mode", "spotting"), patch.object(self.worker, "_config", {"tempRoot": str(self.root)}), patch.object(self.worker, "_progress"):
                result = self.worker._spotting({"path": str(source)})
            self.assertEqual(result["code"], 100)
            self.assertFalse(result["partial"])
            self.assertEqual(len(calls), result["metadata"]["textRecognition"]["initialTiles"])
            self.assertTrue(all(call[1]["max_pixels"] == 1605632 for call in calls))
            self.assertTrue(all(item["score"] is None for item in result["data"]))
            self.assertTrue(all(0 <= x <= size[0] and 0 <= y <= size[1] for item in result["data"] for x,y in item["polygon"]))
            if size == (80, 40):
                self.assertEqual(len(calls), 1)
            else:
                self.assertGreater(max(point[1] for block in result["data"] for point in block["polygon"]), 1200)
            self.assertTrue(source.exists())
            self.assertFalse(list(self.root.glob("spotting-*")))

    def test_dedup_repeated_text_different_positions(self):
        def block(x, y, tile):
            return {"text": "same", "polygon": [[x,y],[x+50,y],[x+50,y+20],[x,y+20]], "score": None, "_tile": tile}
        result = self.worker._deduplicate_blocks([block(10,10,1), block(12,11,2), block(10,100,3)])
        self.assertEqual(len(result), 2)
        self.assertEqual([b["polygon"][0][1] for b in result], [10,100])

    def test_truncation_retiles_is_explicit(self):
        source = self.image((900, 700))
        calls = []
        def predict(path, **kwargs):
            calls.append(path)
            return [{"finish_reason": "length" if len(calls) == 1 else "stop", "spotting_res": {"rec_texts": ["x"], "rec_polys": [[[0,0],[20,0],[20,20],[0,20]]]}}]
        with patch.object(self.worker, "_engine", SimpleNamespace(predict=predict)), patch.object(self.worker, "_mode", "spotting"), patch.object(self.worker, "_config", {"tempRoot": str(self.root)}), patch.object(self.worker, "_progress"):
            result = self.worker._spotting({"path": str(source)})
        self.assertEqual(len(calls), 3)
        self.assertTrue(result["partial"])
        self.assertTrue(result["warnings"])

    def test_blank_and_all_failed_are_distinct(self):
        source = self.image()
        with patch.object(self.worker, "_engine", SimpleNamespace(predict=lambda *a, **k: [{"spotting_res": {"rec_texts": [], "rec_polys": []}}])), patch.object(self.worker, "_mode", "spotting"), patch.object(self.worker, "_config", {"tempRoot": str(self.root)}), patch.object(self.worker, "_progress"):
            self.assertEqual(self.worker._spotting({"path": str(source)})["code"], 101)
        with patch.object(self.worker, "_engine", SimpleNamespace(predict=Mock(side_effect=RuntimeError("HTTP error")))), patch.object(self.worker, "_mode", "spotting"), patch.object(self.worker, "_config", {"tempRoot": str(self.root)}), patch.object(self.worker, "_progress"):
            with self.assertRaises(RuntimeError):
                self.worker._spotting({"path": str(source)})

    def test_real_client_hook_overrides_vendor_600(self):
        create = Mock(return_value="future")
        model = SimpleNamespace(genai_client=SimpleNamespace(create_chat_completion=create), _doc_vlm_genai_collect_responses=lambda x: x)
        self.worker._task_deadline = time.monotonic() + 2
        self.worker._request_timeout = 0.5
        self.worker._install_client_limits(SimpleNamespace(vl_rec_model=model))
        model.genai_client.create_chat_completion([], timeout=600, return_future=True)
        self.assertLessEqual(create.call_args.kwargs["timeout"], 0.5)
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="partial"))], usage=SimpleNamespace(completion_tokens=4096))
        future = Mock()
        future.result.return_value = response
        self.assertEqual(model._doc_vlm_genai_collect_responses([future]), ["partial"])
        self.assertTrue(self.worker._truncated({}, self.worker._generation_signals))

    def test_document_polygon_clips_actual_negative_page_edge(self):
        polygon = [[0,-35],[17,1433],[527,1418],[644,1456],[1017,1456],[1077,1341],[1011,1008],[1044,553],[1099,799],[1077,-35]]
        raw = {"width":1100,"height":1500,"parsing_res_list":[{
            "block_label":"table","block_bbox":[0,0,1100,1457],"block_polygon_points":polygon,
            "block_content":"<table><tr><td>PAGE20001</td></tr></table>","block_id":7,"block_order":None,"group_id":4,
        }]}
        original = json.loads(json.dumps(raw))
        blocks, warnings = self.worker._document_blocks(raw, 2)
        self.assertEqual(warnings, [])
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertTrue(all(0 <= x <= 1100 and 0 <= y <= 1500 for x,y in block["polygon"]))
        self.assertNotEqual(block["polygon"], block["bbox"], "preserve the meaningful nonrectangular polygon")
        self.assertGreater(len(block["polygon"]), 4)
        self.assertEqual((block["id"],block["groupId"],block["category"]),(7,4,"table"))
        self.assertIn("PAGE20001", block["content"])
        self.assertEqual(block["official"]["block_polygon_points"], polygon)
        self.assertEqual(raw, original, "the official diagnostic input must stay untouched")

    def test_document_polygon_clips_right_bottom_and_intersects_edges(self):
        raw = {"width":100,"height":80,"parsing_res_list":[{
            "block_label":"text","block_bbox":[10,10,130,110],
            "block_polygon_points":[[10,10],[130,30],[70,110],[20,70]],"block_content":"edge",
        }]}
        original = json.loads(json.dumps(raw))
        blocks, warnings = self.worker._document_blocks(raw, 2)
        self.assertEqual(warnings, [])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["bbox"], [[10,10],[100,10],[100,80],[10,80]])
        self.assertTrue(all(0 <= x <= 100 and 0 <= y <= 80 for x,y in blocks[0]["polygon"]))
        self.assertIn([100,25], blocks[0]["polygon"], "intersect the polygon edge, do not just clamp its vertex")
        self.assertNotEqual(blocks[0]["polygon"], blocks[0]["bbox"])
        self.assertEqual(raw, original)

    def test_document_polygon_outside_or_degenerate_remains_only_in_raw(self):
        for polygon in ([[-40,-40],[-10,-40],[-10,-10],[-40,-10]], [[10,10],[20,20],[30,30]], [[101,10],[120,10],[120,20],[101,20]]):
            raw = {"width":100,"height":80,"parsing_res_list":[{
                "block_label":"text","block_bbox":[-40,-40,120,90],"block_polygon_points":polygon,"block_content":"unmappable",
            }]}
            original = json.loads(json.dumps(raw))
            blocks, warnings = self.worker._document_blocks(raw, 2)
            self.assertEqual(blocks, [], str(polygon))
            self.assertTrue(warnings)
            self.assertEqual(raw, original)

    def test_document_unknown_page_size_never_fabricates_mapped_bounds(self):
        for dimensions in ({}, {"width":0,"height":80}, {"width":100,"height":-1}):
            raw = {**dimensions,"parsing_res_list":[{
                "block_label":"text","block_bbox":[0,0,20,10],"block_polygon_points":[[0,-2],[20,-2],[20,10],[0,10]],"block_content":"unknown page",
            }]}
            original = json.loads(json.dumps(raw))
            blocks, warnings = self.worker._document_blocks(raw, 2)
            self.assertEqual(blocks, [])
            self.assertTrue(warnings, "unknown dimensions are not reported as validated coordinates")
            self.assertEqual(raw, original)

    def test_document_in_bounds_polygon_and_fields_are_unchanged(self):
        polygon = [[5,5],[70,8],[90,30],[60,65],[5,40]]
        raw = {"width":100,"height":80,"parsing_res_list":[{
            "block_label":"figure","block_bbox":[5,5,90,65],"block_polygon_points":polygon,
            "block_content":"figure resource","block_id":8,"block_order":None,"group_id":4,
        }]}
        blocks, warnings = self.worker._document_blocks(raw, 3)
        self.assertEqual(warnings, [])
        self.assertEqual(blocks[0]["polygon"], polygon)
        self.assertEqual((blocks[0]["id"],blocks[0]["groupId"],blocks[0]["content"]),(8,4,"figure resource"))
        self.assertEqual(blocks[0]["official"], raw["parsing_res_list"][0])

    def test_document_restructure_uses_real_page_size_without_changing_raw(self):
        item = {"block_label":"text","block_bbox":[0,0,110,90],"block_polygon_points":[[5,-10],[110,10],[75,90],[5,60]],"block_content":"kept"}
        raws = [{"parsing_res_list":[dict(item)]} for _ in range(2)]
        results = [{"warnings":[],"data":{"pages":[{"page":number,"width":100,"height":80,"raw":dict(raw),"blocks":[],"resources":[{"source":"figure.png","uri":"file:///owned/figure.png"}]}],"blocks":[]}} for number,raw in zip((2,3),raws)]
        engine = SimpleNamespace(restructure_pages=lambda *a,**k:raws)
        with patch.object(self.worker,"_engine",engine), patch.object(self.worker,"_markdown_of",return_value=("![figure](figure.png)",{})):
            self.worker._restructure_completed(results,{})
        for number,result in zip((2,3),results):
            page = result["data"]["pages"][0]
            self.assertEqual((page["page"],page["width"],page["height"]),(number,100,80))
            self.assertEqual(len(page["blocks"]),1)
            self.assertTrue(all(0 <= x <= 100 and 0 <= y <= 80 for x,y in page["blocks"][0]["polygon"]))
            self.assertEqual(page["raw"]["parsing_res_list"][0],item)
            self.assertNotIn("width",page["raw"],"do not falsify the official restructured raw")
            self.assertEqual(page["resources"],[{"source":"figure.png","uri":"file:///owned/figure.png"}])
            self.assertIn("file:///owned/figure.png",page["markdown"])
            self.assertEqual(result["data"]["blocks"][0]["page"],number)

    def test_document_restructure_unknown_bounds_propagate_partial_warning(self):
        raw = {"parsing_res_list":[{"block_label":"text","block_bbox":[0,0,20,10],"block_content":"raw retained"}]}
        results = [{"warnings":[],"data":{"pages":[{"page":number,"width":0,"height":0,"raw":dict(raw),"blocks":[],"resources":[]}],"blocks":[]}} for number in (2,3)]
        engine = SimpleNamespace(restructure_pages=lambda *a,**k:[raw,raw])
        with patch.object(self.worker,"_engine",engine), patch.object(self.worker,"_markdown_of",return_value=("raw retained",{})):
            self.worker._restructure_completed(results,{})
        combined = self.worker._combine_document(self.root/"fixture.pdf",results,[],2)
        self.assertTrue(combined["partial"])
        self.assertTrue(combined["warnings"])
        self.assertEqual(combined["data"]["blocks"],[])
        self.assertEqual([page["page"] for page in combined["data"]["pages"]],[2,3])
        self.assertTrue(all(page["raw"]["parsing_res_list"] for page in combined["data"]["pages"]))

    def test_document_page_range_before_prediction_and_progress(self):
        from PIL import Image
        source = self.root / "pages.tiff"
        frames = [Image.new("RGB", (40,30), (index,0,0)) for index in range(4)]
        frames[0].save(source, save_all=True, append_images=frames[1:])
        for frame in frames:
            frame.close()
        seen, events = [], []
        def page(path, options, number):
            self.assertTrue(Path(path).is_file())
            seen.append(number)
            record = {"page": number, "text": str(number), "markdown": str(number), "blocks": [], "resources": [], "raw": {}}
            return {"code":100, "data":{"pages":[record],"blocks":[],"assets":[]},"warnings":[],"metadata":{}}
        with patch.object(self.worker, "_engine", object()), patch.object(self.worker, "_mode", "document"), patch.object(self.worker, "_config", {"tempRoot":str(self.root)}), patch.object(self.worker, "_document_page", side_effect=page), patch.object(self.worker, "_restructure_completed"), patch.object(self.worker, "_progress", side_effect=lambda stage, **kw: events.append(kw)):
            result = self.worker._document({"path":str(source),"options":{"pageRange":[2,3]}})
        self.assertEqual(seen,[2,3])
        self.assertEqual([p["page"] for p in result["data"]["pages"]],[2,3])
        self.assertEqual([e["page"] for e in events],[2,3])
        self.assertFalse(result["partial"])
        self.assertTrue(source.exists())
        self.assertFalse(list(self.root.glob("document-pages-*")))

    def test_document_deadline_preserves_completed_page(self):
        from PIL import Image
        source = self.root / "deadline.tiff"
        first, second = Image.new("RGB",(30,20)), Image.new("RGB",(30,20))
        first.save(source, save_all=True, append_images=[second])
        first.close(); second.close()
        def page(path, options, number):
            record = {"page":number,"text":"one","markdown":"one","blocks":[],"resources":[],"raw":{}}
            self.worker._task_deadline = time.monotonic() - 1
            return {"code":100,"data":{"pages":[record],"blocks":[],"assets":[]},"warnings":[],"metadata":{}}
        with patch.object(self.worker, "_engine", object()), patch.object(self.worker, "_mode", "document"), patch.object(self.worker, "_config", {"tempRoot":str(self.root)}), patch.object(self.worker, "_document_page", side_effect=page), patch.object(self.worker, "_restructure_completed"), patch.object(self.worker, "_progress"):
            result = self.worker._document({"path":str(source)})
        self.assertEqual(result["code"],100)
        self.assertTrue(result["partial"])
        self.assertEqual(result["metadata"]["failedPages"][0]["page"],2)
        self.assertEqual(result["data"]["pages"][0]["page"],1)

    def test_invalid_page_range_fails(self):
        for page_range in ([0,1],[3,2],[1,"2"],[1,2,3],[True,2]):
            with self.assertRaises(self.budgets.BudgetError):
                self.budgets.selected_pages(page_range, 4)

    def test_pdf_range_renders_selected_only(self):
        from PIL import Image
        source = self.root / "synthetic.pdf"
        source.write_bytes(b"%PDF-mock")
        selected, rendered = [], []
        class Page:
            def __init__(self, index):
                self.index = index
            def get_size(self):
                return 20, 10
            def render(self, scale=1):
                rendered.append((self.index, scale))
                return SimpleNamespace(to_pil=lambda:Image.new("RGB",(40,20)), close=lambda:None)
            def close(self):
                pass
        class Document:
            def __len__(self):
                return 5
            def __getitem__(self, index):
                selected.append(index)
                return Page(index)
            def close(self):
                pass
        with patch.dict(sys.modules, {"pypdfium2":SimpleNamespace(PdfDocument=lambda path:Document())}):
            with self.budgets.document_pages(source, [2,3], self.root, time.monotonic()+5) as (selection, inputs):
                self.assertEqual(selection,[2,3])
                self.assertEqual(selected,[], "打开元数据不渲染任何未选页")
                for number, path in inputs:
                    self.assertTrue(path.is_file())
                    self.assertIn(number,[2,3])
        self.assertEqual(selected,[1,2])
        self.assertEqual(rendered,[(1,2),(2,2)])
        self.assertTrue(source.is_file())
        self.assertFalse(list(self.root.glob("page-*.png")))

    def test_job_assets_are_owned_and_junction_rejected(self):
        from win32_PaddleOCRVL_1_6 import asset_paths
        from PIL import Image
        owned = self.root / "assets"
        self.worker._asset_output_root = asset_paths.output_root(owned)
        image = Image.new("RGB", (20,10), "white")
        try:
            markdown, assets, warnings = self.worker._materialize_assets("![x](a.jpg)", {"a.jpg":image}, "page")
        finally:
            image.close()
        self.assertFalse(warnings)
        target = Path(assets[0]["path"])
        self.assertTrue(target.is_relative_to(owned))
        self.assertTrue(target.is_file())
        self.assertIn(target.resolve().as_uri(), markdown)
        original = Path.lstat
        def mocked(path, *args, **kwargs):
            info = original(path, *args, **kwargs)
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400) if path == owned else info
        with patch.object(Path, "lstat", mocked):
            with self.assertRaises(ValueError):
                asset_paths.output_root(owned)
        with self.assertRaises(ValueError):
            asset_paths.output_root("relative/assets")

    def test_asset_byte_budget_blocks_before_unbounded_write(self):
        from win32_PaddleOCRVL_1_6 import asset_paths
        owned = self.root / "assets"
        self.worker._asset_output_root = asset_paths.output_root(owned)
        class EncodedImage:
            def save(self, target, format=None):
                target.write(b"x" * 20)
        with patch.object(asset_paths, "MAX_ASSET_BYTES", 24), patch.object(asset_paths, "MAX_JOB_BYTES", 32):
            _, assets, warnings = self.worker._materialize_assets("a.png b.png", {"a.png":EncodedImage(), "b.png":EncodedImage()}, "page")
        self.assertEqual(len(assets), 1)
        self.assertEqual(self.worker._asset_output_bytes, 20)
        self.assertTrue(self.worker._asset_budget_exhausted)
        self.assertTrue(warnings)
        self.assertEqual(sum(path.stat().st_size for path in owned.rglob("*") if path.is_file()), 20)
        self.assertFalse(list(owned.rglob("*.tmp")))

    def test_asset_count_budget_is_finite(self):
        from win32_PaddleOCRVL_1_6 import asset_paths
        from PIL import Image
        owned = self.root / "assets"
        self.worker._asset_output_root = asset_paths.output_root(owned)
        self.worker._asset_output_count = 500
        image = Image.new("RGB", (10,10))
        try:
            _, assets, warnings = self.worker._materialize_assets("x.png", {"x.png":image}, "page")
        finally:
            image.close()
        self.assertEqual(assets, [])
        self.assertTrue(self.worker._asset_budget_exhausted)
        self.assertTrue(warnings)
        self.assertFalse([p for p in owned.rglob("*") if p.is_file()])

    def test_progress_does_not_extend_ipc_deadline(self):
        from win32_PaddleOCRVL_1_6.api import _WorkerClient
        client = _WorkerClient()
        client.process = Mock()
        client.process.poll.return_value = None
        client.process.stdin = io.StringIO()
        with patch.object(client, "stop") as stop:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                client.request({"op":"document"}, 0.05)
            self.assertLess(time.monotonic() - started, 0.5)
            stop.assert_called_once()

    def test_ipc_progress_keeps_structured_payload(self):
        from win32_PaddleOCRVL_1_6.api import _WorkerClient
        client = _WorkerClient()
        stages, partial = [], []
        client.on_progress, client.on_partial = stages.append, partial.append
        event = {"event":"progress","stage":"page","page":2,"result":{"data":{"pages":[{"page":2}]}}}
        client._read(io.StringIO(json.dumps(event)+"\n"+json.dumps({"ok":True,"result":42})+"\n"))
        self.assertEqual(stages,["page"])
        self.assertEqual(partial,[event])
        self.assertEqual(client.responses.get_nowait()["result"],42)


def run(plugin_root, work_root, report=None):
    global WORK_ROOT
    WORK_ROOT = Path(work_root).resolve()
    if not WORK_ROOT.is_dir():
        raise FileNotFoundError(WORK_ROOT)
    sys.path.insert(0, str(Path(plugin_root).resolve().parent))
    # 嵌入式解释器的 _pth 不自动载入宿主 site-packages；只读取匹配解释器旁的依赖。
    host_packages = Path(sys.executable).parent.parent / "site-packages"
    if host_packages.is_dir():
        sys.path.append(str(host_packages))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(VLTests))
    print(stream.getvalue())
    if report:
        from datetime import datetime, timezone
        root = Path(plugin_root)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.py")}
        hashes["umi-plugin.json"] = hashlib.sha256((root / "umi-plugin.json").read_bytes()).hexdigest()
        Path(report).write_text(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "python":sys.executable,
            "pluginRoot":str(root.resolve()), "workRoot":str(WORK_ROOT), "testsRun":result.testsRun,
            "failures":len(result.failures), "errors":len(result.errors), "skipped":len(result.skipped), "successful":result.wasSuccessful(),
            "sourceSha256":hashes, "output":stream.getvalue()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return result.wasSuccessful()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plugin_root", type=Path)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.exit(0 if run(args.plugin_root, args.work_root, args.report) else 1)

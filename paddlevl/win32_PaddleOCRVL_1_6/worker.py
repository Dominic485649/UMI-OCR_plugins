"""Private PaddleOCR-VL 1.6 Pipeline worker.

The host process never imports Paddle/PaddleX.  This child owns the official
PaddleOCR-VL pipeline and talks to its shared llama.cpp server over the
official ``llama-cpp-server`` backend.  JSON lines are the only IPC boundary.
"""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import re
import sys
import tempfile
import time
import types
import uuid
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
VENDOR = PLUGIN_DIR / "vendor"
os.environ["PADDLE_PDX_DISABLE_DEVICE_FALLBACK"] = "True"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
sys.path.insert(0, str(PLUGIN_DIR))
try:
    from .runtime_archive_bootstrap import prepare_archives
except ImportError:
    from runtime_archive_bootstrap import prepare_archives
_runtime_archive_paths = prepare_archives(PLUGIN_DIR)
sys.path[0:0] = [*_runtime_archive_paths, str(VENDOR)]

try:
    from . import budgets, asset_paths
    from .checks import tr
except ImportError:  # worker.py 作为独立解释器入口
    import budgets
    import asset_paths
    from checks import tr

_dll_handles = []
if os.name == "nt":
    candidates = [VENDOR / "paddle" / "base"]
    nvidia = VENDOR / "nvidia"
    if nvidia.is_dir():
        candidates.extend(sorted(nvidia.glob("*")))
    for path in candidates:
        candidate = path / "bin"
        if not candidate.is_dir():
            candidate = path
        if candidate.is_dir():
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                _dll_handles.append(os.add_dll_directory(str(candidate)))

SPOTTING_MAX_PIXELS = 1_605_632
SPOTTING_MIN_PIXELS = 112_896
SPOTTING_UPSCALE_LIMIT = 1_500
LABEL_TO_CATEGORY = {
    "doc_title": ("title", tr('标题')),
    "title": ("title", tr('标题')),
    "paragraph_title": ("title", tr('标题')),
    "abstract_title": ("title", tr('标题')),
    "table": ("table", tr('表格')),
    "formula": ("formula", tr('公式')),
    "display_formula": ("formula", tr('公式')),
    "inline_formula": ("formula", tr('行内公式')),
    "formula_number": ("formula", tr('公式编号')),
    "text": ("text", tr('正文')),
    "content": ("text", tr('正文')),
    "abstract": ("text", tr('摘要')),
    "reference": ("reference", tr('参考文献')),
    "number": ("text", tr('页码')),
    "image": ("figure", tr('图片')),
    "figure": ("figure", tr('图片')),
    "chart": ("figure", tr('图表')),
    "seal": ("figure", tr('印章')),
    "table_title": ("caption", tr('表格标题')),
    "table_caption": ("caption", tr('表格说明')),
    "figure_title": ("caption", tr('图片标题')),
    "figure_caption": ("caption", tr('图片说明')),
    "chart_title": ("caption", tr('图表标题')),
    "image_caption": ("caption", tr('图片说明')),
    "vision_footnote": ("caption", tr('图片注释')),
    "header": ("header", tr('页眉')),
    "header_image": ("header", tr('页眉图片')),
    "footer": ("footer", tr('页脚')),
    "footer_image": ("footer", tr('页脚图片')),
    "footnote": ("footnote", tr('脚注')),
    "aside_text": ("aside", tr('旁注')),
    "reference_content": ("reference", tr('参考文献')),
    "spotting": ("text", tr('正文')),
    "ocr": ("text", tr('正文')),
}

_engine = None
_config = None
_mode = ""
_partial_failures = []
_generation_signals = []
_task_deadline = float("inf")
_request_timeout = 120
_asset_output_root = None
_asset_output_bytes = 0
_asset_output_count = 0
_asset_budget_exhausted = False

def _remaining():
    return budgets.remaining(_task_deadline, _request_timeout)


def _begin_task(request, default):
    global _task_deadline, _request_timeout, _asset_output_root, _asset_output_bytes, _asset_output_count, _asset_budget_exhausted
    _asset_output_root = asset_paths.output_root(request["assetOutputRoot"]) if request.get("assetOutputRoot") else None
    _asset_output_bytes, _asset_output_count = asset_paths.inventory(_asset_output_root) if _asset_output_root is not None else (0, 0)
    _asset_budget_exhausted = False
    _generation_signals.clear()
    seconds = request.get("taskTimeoutSeconds", default)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= default:
        raise ValueError(tr('INVALID_CONFIGURATION：任务总预算无效'))
    _task_deadline = time.monotonic() + seconds
    deadline = request.get("deadlineMonotonic")
    if deadline is not None:
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
            raise ValueError(tr('INVALID_CONFIGURATION：任务截止时间无效'))
        _task_deadline = min(_task_deadline, deadline)
    _request_timeout = float((_config or {}).get("timeout", 120))
    _remaining()


def _pipeline_targets(engine):
    outer = getattr(engine, "paddlex_pipeline", None) or engine
    state = vars(outer)
    return list(state.get("_pipelines") or []) or [state.get("_pipeline") or outer]


def _install_client_limits(engine):
    """锁定私有 PaddleX 实际接口：覆盖它硬编码的 timeout=600 并保留 finish_reason。

    不修改 vendor；只包装本 worker 自有的 GenAIClient 和响应收集方法。
    """
    patched = 0
    for target in _pipeline_targets(engine):
        model = getattr(target, "vl_rec_model", None)
        client = getattr(model, "genai_client", None)
        if client is None or not callable(getattr(model, "_doc_vlm_genai_collect_responses", None)):
            continue
        original = client.create_chat_completion
        def limited(messages, *, _original=original, **kwargs):
            kwargs["timeout"] = _remaining()
            return _original(messages, **kwargs)
        client.create_chat_completion = limited
        def collect(self, futures):
            results = []
            for future in futures:
                response = future.result(timeout=_remaining())
                choice = response.choices[0]
                usage = getattr(response, "usage", None)
                _generation_signals.append({
                    "finishReason": getattr(choice, "finish_reason", None),
                    "completionTokens": getattr(usage, "completion_tokens", None),
                })
                results.append(choice.message.content)
            return results
        model._doc_vlm_genai_collect_responses = types.MethodType(collect, model)
        patched += 1
    if not patched:
        raise RuntimeError(tr('PIPELINE_DEPENDENCY_MISSING：当前 PaddleX 不支持受控超时与截断检查，请重新安装配套插件。'))
    return engine


def _truncated(raw, signals):
    reasons = [item.get("finishReason") for item in signals]
    def inspect(value):
        if isinstance(value, dict):
            if value.get("finish_reason") in ("length", "max_tokens") or value.get("finishReason") in ("length", "max_tokens"):
                return True
            return any(inspect(child) for child in value.values())
        if isinstance(value, list):
            return any(inspect(child) for child in value)
        return False
    return any(reason in ("length", "max_tokens") for reason in reasons) or inspect(raw)


def _get(node, key, default=None):
    if isinstance(node, dict):
        value = node.get(key, default)
    else:
        try:
            value = node[key]
        except Exception:
            value = getattr(node, key, default)
    return default if value is None else value


def _json_value(value):
    """Keep all official JSON-compatible values, never send image bytes over IPC."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items() if key != "image"}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_value(tolist())
        except Exception:
            pass
    if hasattr(value, "save"):
        # Images are materialized separately as resources.  Retaining their
        # repr would look like data while being neither stable nor useful.
        return None
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _result_json(result):
    """Read Paddle Result.json without assuming whether it wraps ``res``."""
    raw = getattr(result, "json", result)
    if callable(raw):
        raw = raw()
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict) and isinstance(raw.get("res"), dict):
        raw = raw["res"]
    if not isinstance(raw, dict):
        raise ValueError(tr('PaddleOCR-VL 结果不是 JSON 对象'))
    return _json_value(raw)


def _coerce_points(value):
    if value is None:
        return []
    try:
        raw = value.tolist() if hasattr(value, "tolist") else value
        values = list(raw)
        if len(values) == 4 and all(isinstance(item, (int, float)) for item in values):
            x1, y1, x2, y2 = (float(item) for item in values)
            values = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        points = []
        for item in values:
            if len(item) < 2:
                return []
            x, y = float(item[0]), float(item[1])
            if not math.isfinite(x) or not math.isfinite(y):
                return []
            points.append([x, y])
        return points
    except (TypeError, ValueError, IndexError):
        return []


def _box_from_points(points):
    if len(points) == 4:
        return [[int(round(x)), int(round(y))] for x, y in points]
    if len(points) < 3:
        return []
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [
        [int(round(min(xs))), int(round(min(ys)))],
        [int(round(max(xs))), int(round(min(ys)))],
        [int(round(max(xs))), int(round(max(ys)))],
        [int(round(min(xs))), int(round(max(ys)))],
    ]


def _polygon(value):
    return [[int(round(x)), int(round(y))] for x, y in _coerce_points(value)]


def _device_check(device):
    import paddle

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError(tr('插件 Paddle 不是 CUDA 构建；拒绝 CPU fallback'))
    text = str(device or "gpu:0").lower()
    if not text.startswith("gpu:") or not text[4:].isdigit():
        raise RuntimeError(tr('版面检测设备必须使用 gpu:N，当前为 {device}', device=device))
    index = int(text[4:])
    count = paddle.device.cuda.device_count()
    if index >= count:
        raise RuntimeError(tr('请求 {text}，但只检测到 {count} 个 CUDA 设备', text=text, count=count))
    paddle.device.set_device(text)
    probe = paddle.to_tensor([1.0], place=paddle.CUDAPlace(index))
    if float((probe + 1.0).numpy()[0]) != 2.0:
        raise RuntimeError(tr('CUDA 张量自检失败'))


def _probe_dependencies(config):
    """Verify the worker's own Paddle/PaddleOCR/PaddleX environment.

    It intentionally creates no PaddleOCRVL pipeline and loads no model.  In
    document mode the tiny CUDA tensor operation is still valuable: it turns a
    driver/runtime/device problem into an actionable configuration error before
    llama-server is started or a long document task is queued.
    """
    import paddle
    import paddleocr
    import paddlex

    details = {
        "python": ".".join(str(item) for item in sys.version_info[:3]),
        "paddle": str(getattr(paddle, "__version__", "unknown")),
        "paddleocr": str(getattr(paddleocr, "__version__", "unknown")),
        "paddlex": str(getattr(paddlex, "__version__", "unknown")),
        "cudaCompiled": bool(paddle.is_compiled_with_cuda()),
    }
    required = {"paddleocr": (3, 7, 0), "paddlex": (3, 7, 2)}
    for package, minimum in required.items():
        raw = details[package]
        parts = []
        for chunk in raw.split("."):
            digits = "".join(char for char in chunk if char.isdigit())
            parts.append(int(digits) if digits else 0)
            if len(parts) == 3:
                break
        while len(parts) < 3:
            parts.append(0)
        if tuple(parts) < minimum:
            expect = ".".join(str(item) for item in minimum)
            raise RuntimeError(tr('{package} 版本过低：{raw} < {expect}', package=package, raw=raw, expect=expect))
    if details["cudaCompiled"]:
        details["cudaDeviceCount"] = int(paddle.device.cuda.device_count())
    else:
        details["cudaDeviceCount"] = 0
    if details["cudaDeviceCount"] < 1:
        raise RuntimeError(tr('CUDA 不可用或没有设备；本插件不提供 CPU 回退'))
    device = str((config or {}).get("device", "gpu:0"))
    index = int(device.split(":")[-1]) if device.startswith("gpu:") and device[4:].isdigit() else 0
    major, minor = paddle.device.cuda.get_device_capability(index)
    details["computeCapability"] = "%d.%d" % (major, minor)
    details["minimumComputeCapability"] = "7.5"
    if (major, minor) < (7, 5):
        raise RuntimeError(tr('CUDA 设备低于兼容目标 Turing sm75，请使用支持的 GPU；设备名称不能作为兼容保证'))
    if bool((config or {}).get("requireCuda")):
        _device_check(device)
        details["cudaDevice"] = device
    details["llamaCudaLoadVerified"] = False  # Paddle 检查不等同于 llama CUDA 实际加载

    return details


def _client_kwargs(pipeline, timeout):
    submodules = pipeline.setdefault("SubModules", {})
    recognition = submodules.setdefault("VLRecognition", {})
    genai = recognition.setdefault("genai_config", {})
    client = genai.setdefault("client_kwargs", {})
    client["timeout"] = int(timeout)
    # OpenAI's client retries twice by default.  Document mode already owns a
    # single bounded block retry below; disabling the hidden client retries
    # keeps one failed region to exactly two attempts instead of up to six.
    client["max_retries"] = 0
    # The server alias is a deliberate public-internal boundary.  Paddle's
    # llama-cpp-server client sends it as the OpenAI model field.
    client["model_name"] = "paddlevl"
    return pipeline


def _create_document_engine(config):
    from paddleocr import PaddleOCRVL
    from paddlex.inference import load_pipeline_config

    layout_dir = Path(config["layoutModelDir"])
    if not (layout_dir / "inference.json").is_file():
        raise FileNotFoundError(tr('版面检测模型目录缺少 inference.json: {layout_dir}', layout_dir=layout_dir))
    _device_check(config.get("device", "gpu:0"))
    pipeline = load_pipeline_config("PaddleOCR-VL-1.6")
    pipeline["SubModules"]["LayoutDetection"]["model_dir"] = str(layout_dir)
    _client_kwargs(pipeline, config.get("timeout", 120))
    engine = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_backend="llama-cpp-server",
        vl_rec_server_url=str(config["baseUrl"]),
        vl_rec_api_model_name="paddlevl",
        layout_detection_model_dir=str(layout_dir),
        vl_rec_max_concurrency=1,
        use_layout_detection=True,
        # The v1.6 config defaults these three to false.  Document mode is
        # explicitly the full layout product: charts, seals and image regions
        # are therefore sent through the same VLM flow instead of being left
        # as unprocessed layout-only boxes.
        use_chart_recognition=True,
        use_seal_recognition=True,
        use_ocr_for_image_block=True,
        format_block_content=False,
        merge_layout_blocks=True,
        use_queues=False,
        paddlex_config=pipeline,
        device=str(config.get("device", "gpu:0")),
        enable_hpi=False,
    )
    _install_document_failure_isolation(engine)
    return _install_client_limits(engine)


def _install_document_failure_isolation(engine):
    """Make region VLM failures partial without changing PaddleX vendor code.

    PaddleX batches all cropped regions by visual budget.  A failed request
    otherwise aborts the whole batch, which loses unrelated layout blocks.
    The wrapper keeps the official layout/crop/restructure path and only
    changes the request granularity at this boundary.
    """
    method_name = "_paddleocr_vl_run_vl_recognition_batches"
    # PaddleOCR's public wrapper forwards unknown attributes to its inner
    # PaddleX pipeline.  Testing the outer object first therefore appears to
    # find this method, but assigning there patches an unused shadow method.
    # Prefer the actual execution object whenever it exists.
    outer = getattr(engine, "paddlex_pipeline", None) or engine
    # create_pipeline() returns an AutoParallel wrapper even for one GPU.  Its
    # __getattr__ forwards reads to _pipeline, while predict() executes that
    # inner object, so assigning a shadow method on the wrapper has no effect.
    state = vars(outer)
    targets = list(state.get("_pipelines") or [])
    if not targets:
        targets = [state.get("_pipeline") or outer]
    targets = [target for target in targets if callable(getattr(target, method_name, None))]
    if not targets:
        return engine

    def isolated(self, batch_dict_by_pixel, has_spotting, vlm_kwargs):
        if vlm_kwargs is None:
            vlm_kwargs = {}
        else:
            vlm_kwargs = dict(vlm_kwargs)
        if vlm_kwargs.get("max_new_tokens") is None:
            vlm_kwargs["max_new_tokens"] = 4096
        for pixel_key, pixel_info in batch_dict_by_pixel.items():
            min_px, max_px = pixel_key
            kwargs = {
                "use_cache": True,
                "min_pixels": min_px,
                "max_pixels": max_px,
                **vlm_kwargs,
            }
            results = []
            for image, query, block_id in zip(
                pixel_info.get("images", []),
                pixel_info.get("queries", []),
                pixel_info.get("vlm_block_ids", []),
            ):
                _remaining()
                _progress(tr('识别区域：{region}', region=str(block_id)))
                payload = {"image": image, "query": query}
                last_error = None
                result = None
                for attempt in range(2):
                    try:
                        _remaining()
                        signal_start = len(_generation_signals)
                        candidates = list(
                            self.vl_rec_model.predict(
                                [payload],
                                skip_special_tokens=False if has_spotting else True,
                                **kwargs,
                            )
                        )
                        if len(candidates) != 1 or not isinstance(candidates[0], dict):
                            raise ValueError(tr('VLM 返回结果不是单个对象'))
                        candidate = candidates[0]
                        if "result" not in candidate:
                            raise ValueError(tr('VLM 返回对象缺少 result 字段'))
                        if candidate.get("result") is not None and not isinstance(
                            candidate.get("result"), str
                        ):
                            raise ValueError(tr('VLM result 字段不是字符串'))
                        if _truncated(candidate, _generation_signals[signal_start:]):
                            raise ValueError(tr('OUTPUT_TRUNCATED：区域输出被截断，请提高输出预算或缩小输入区域。'))
                        result = candidate
                        break
                    except Exception as exc:  # region failures are isolated
                        last_error = exc
                if result is None:
                    try:
                        page_index, region_index = (int(block_id[0]), int(block_id[1]))
                    except (TypeError, ValueError, IndexError):
                        page_index, region_index = -1, -1
                    _partial_failures.append(
                        {
                            "pageIndex": page_index,
                            "blockIndex": region_index,
                            "pixelBudget": {
                                "minPixels": int(min_px),
                                "maxPixels": int(max_px),
                            },
                            "prompt": str(query),
                            "retries": 1,
                            "error": str(last_error or tr('未知 VLM 错误'))[:500],
                        }
                    )
                    result = {"result": ""}
                results.append(result)
            pixel_info["vlm_results"] = results
    for target in targets:
        target._paddleocr_vl_run_vl_recognition_batches = types.MethodType(isolated, target)
    return engine


def _create_spotting_engine(config):
    from paddleocr import PaddleOCRVL
    from paddlex.inference import load_pipeline_config

    pipeline = _client_kwargs(load_pipeline_config("PaddleOCR-VL-1.6"), config.get("timeout", 120))

    # Do not give the spotting path a layout model directory.  It must use the
    # official no-layout flow, whose prompt_label="spotting" applies
    # pre_process_for_spotting (2x LANCZOS below 1500x1500) and the official
    # 1605632 pixel visual budget.
    engine = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_backend="llama-cpp-server",
        vl_rec_server_url=str(config["baseUrl"]),
        vl_rec_api_model_name="paddlevl",
        vl_rec_max_concurrency=1,
        use_layout_detection=False,
        format_block_content=False,
        use_queues=False,
        paddlex_config=pipeline,
        enable_hpi=False,
    )
    return _install_client_limits(engine)


def _create_engine(config):
    mode = str(config.get("mode") or "")
    if mode == "document":
        return _create_document_engine(config)
    if mode == "spotting":
        return _create_spotting_engine(config)
    raise ValueError(f"unknown PaddleOCR-VL worker mode: {mode}")


def _stop_engine():
    global _engine, _config, _mode
    engine = _engine
    _engine = None
    _config = None
    _mode = ""
    if engine is not None:
        try:
            close = getattr(engine, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
        del engine
        gc.collect()


def _source_size(source):
    try:
        from PIL import Image

        with Image.open(source) as image:
            return int(image.width), int(image.height)
    except Exception as exc:
        raise RuntimeError(tr('无法读取原图尺寸，不能安全还原文字区域坐标：{error}', error=exc)) from exc


def _extract_spotting(raw, source):
    spotting = raw.get("spotting_res") if isinstance(raw, dict) else None
    if isinstance(spotting, list):
        spotting = spotting[0] if spotting else {}
    if not isinstance(spotting, dict):
        raise ValueError(tr('INVALID_MODEL_OUTPUT：文字定位结果缺少区域数据'))
    texts = spotting.get("rec_texts") or []
    polys = spotting.get("rec_polys") or []
    try:
        texts, polys = list(texts), list(polys)
    except TypeError as exc:
        raise ValueError(tr('INVALID_MODEL_OUTPUT：文字定位文本或坐标不是列表')) from exc
    if not texts and not polys:
        return [], [], {"sourceWidth": 0, "sourceHeight": 0, "upscaled": False, "coordinateScale": 1.0}
    if len(texts) != len(polys):
        raise ValueError(
            tr('INVALID_MODEL_OUTPUT：文字定位文本数量 {texts} 与多边形数量 {polys} 不一致', texts=len(texts), polys=len(polys))
        )
    width, height = _source_size(source)
    upscaled = width < SPOTTING_UPSCALE_LIMIT and height < SPOTTING_UPSCALE_LIMIT
    # PaddleX post_process_for_spotting scales normalized locations using the
    # original block dimensions, not the upscaled VLM input. Do not scale twice.
    scale = 1.0
    blocks, warnings = [], []
    for index, (text, polygon_raw) in enumerate(zip(texts, polys)):
        polygon = _coerce_points(polygon_raw)
        if len(polygon) < 3:
            warnings.append(tr('第 {index} 个文字区域坐标无效，已丢弃。', index=index + 1))
            continue
        restored = []
        for x, y in polygon:
            restored.append([
                int(round(min(max(x * scale, 0.0), width))),
                int(round(min(max(y * scale, 0.0), height))),
            ])
        if len(restored) < 3 or not str(text or "").strip():
            warnings.append(tr('第 {index} 个文字区域内容或坐标无效，已丢弃。', index=index + 1))
            continue
        blocks.append({
            "text": str(text),
            "box": _box_from_points(restored),
            "polygon": restored,
            # llama.cpp / current PaddleOCR-VL spotting does not expose a
            # calibrated confidence.  Do not fabricate one.
            "score": None,
        })
    if not blocks:
        raise ValueError(tr('INVALID_MODEL_OUTPUT：文字定位返回的所有区域都无有效坐标'))
    return blocks, warnings, {
        "sourceWidth": width,
        "sourceHeight": height,
        "upscaled": upscaled,
        "coordinateScale": scale,
    }


def _work_root():
    root = Path((_config or {}).get("tempRoot") or PLUGIN_DIR / "temp_doc")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _deduplicate_blocks(blocks):
    """只消除相邻重叠块中的同字同位置，绝不按文字去重不同位置的重复行。"""
    def bounds(block):
        xs, ys = zip(*block["polygon"])
        return min(xs), min(ys), max(xs), max(ys)
    kept, buckets = [], {}
    # 空间桶仅比较同文字的相邻区域，避免密集长图的全量 O(n²) 去重。
    for item in sorted(blocks, key=lambda b: b.get("_edgeDistance", 0), reverse=True):
        a = bounds(item)
        keys = [(item["text"].strip(), x, y) for x in range(a[0] // 256, a[2] // 256 + 1) for y in range(a[1] // 256, a[3] // 256 + 1)]
        candidates = {index for key in keys for index in buckets.get(key, [])}
        duplicate = False
        for index in candidates:
            other = kept[index]
            if other.get("_tile") == item.get("_tile"):
                continue
            b = bounds(other)
            overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            area = min(max(1, (a[2] - a[0]) * (a[3] - a[1])), max(1, (b[2] - b[0]) * (b[3] - b[1])))
            if overlap / area >= 0.65:
                duplicate = True
                break
        if not duplicate:
            index = len(kept)
            kept.append(item)
            for key in keys:
                buckets.setdefault(key, []).append(index)
    # 按行聚类后从左到右；避免同一行轻微 y 抖动打乱阅读顺序。
    lines = []
    for item in sorted(kept, key=lambda b: (bounds(b)[1], bounds(b)[0])):
        box = bounds(item)
        center = (box[1] + box[3]) / 2
        line = next((line for line in reversed(lines[-32:]) if abs(line[0] - center) <= max(2, min(line[1], box[3] - box[1]) * 0.5)), None)
        if line is None:
            lines.append([center, box[3] - box[1], [item]])
        else:
            line[2].append(item)
    ordered = []
    for _, _, line in sorted(lines, key=lambda row: row[0]):
        for item in sorted(line, key=lambda b: bounds(b)[0]):
            ordered.append({k: v for k, v in item.items() if not k.startswith("_")})
    return ordered


def _spotting(request):
    from PIL import Image
    if _engine is None or _mode != "spotting":
        raise RuntimeError(tr('文字识别 worker 尚未初始化'))
    _begin_task(request, 300)
    source = Path(str(request["path"])).expanduser().resolve()
    width, height = budgets.image_size(source, prepared=bool(request.get("preparedInput")))  # 解码前检查，不先完整展开巨图
    boxes = budgets.tile_boxes(width, height)
    options = dict(request.get("options") or {})
    warnings, blocks, records, failed_tiles = [], [], [], []
    pending = [(box, 0) for box in boxes]
    successful, calls = 0, 0
    with Image.open(source) as opened, tempfile.TemporaryDirectory(prefix="spotting-", dir=_work_root()) as temporary:
        if getattr(opened, "n_frames", 1) != 1:
            raise budgets.BudgetError(tr('INVALID_INPUT：多页图片请使用文档解析，不会静默只识别第一页。'))
        opened.load()
        while pending:
            box, depth = pending.pop(0)
            if calls >= budgets.MAX_TILE_CALLS or time.monotonic() >= _task_deadline:
                failed_tiles.extend({"box": list(b), "error": "TASK_DEADLINE_OR_TILE_BUDGET"} for b, _ in [(box, depth)] + pending)
                break
            calls += 1
            target = source
            if len(boxes) != 1 or depth:
                target = Path(temporary) / ("tile-%d.png" % calls)
                crop = opened.crop(box).convert("RGB")
                try:
                    crop.save(target, format="PNG")
                finally:
                    crop.close()
            signal_start = len(_generation_signals)
            tile_blocks, tile_warnings, raw = [], [], {}
            error = None
            try:
                for attempt in range(2):
                    try:
                        if attempt:
                            if calls >= budgets.MAX_TILE_CALLS:
                                raise budgets.BudgetError(tr('TILE_COUNT_BUDGET：分块重试已达到上限，请拆分图片后重试。'))
                            calls += 1
                        _remaining()
                        _progress(tr('识别图片分块'), tile=calls, totalTiles=len(boxes))
                        pages = list(_engine.predict(str(target), use_layout_detection=False, prompt_label="spotting", temperature=0,
                            min_pixels=SPOTTING_MIN_PIXELS, max_pixels=SPOTTING_MAX_PIXELS,
                            max_new_tokens=int(options.get("maxOutputTokens", 4096)), use_queues=False))
                        if len(pages) != 1:
                            raise ValueError(tr('INVALID_MODEL_OUTPUT：文字定位未返回单页结果，不能当作无字。'))
                        raw = _result_json(pages[0])
                        if raw.get("parsing_res_list") == [] and not raw.get("spotting_res"):
                            tile_blocks, tile_warnings = [], []
                        else:
                            tile_blocks, tile_warnings, _ = _extract_spotting(raw, target)
                        error = None
                        break
                    except Exception as exc:
                        error = exc
                signals = _generation_signals[signal_start:]
                truncated = _truncated(raw, signals)
                if error is not None:
                    failed_tiles.append({"box": list(box), "error": str(error)[:500]})
                    continue
                successful += 1
                children = budgets.split_box(box) if truncated and depth < 2 else []
                records.append({"box": list(box), "depth": depth, "truncated": truncated, "generation": signals})
                if children and calls + len(pending) + len(children) <= budgets.MAX_TILE_CALLS:
                    warnings.append(tr("检测到分块输出截断，已继续细分；请核对结果完整性。"))
                    pending[0:0] = [(child, depth + 1) for child in children]
                    # 原结果也保留，避免子块失败使边缘文字消失；几何去重不会虚构置信度。
                elif truncated:
                    failed_tiles.append({"box": list(box), "error": "OUTPUT_TRUNCATED"})
                warnings.extend(tile_warnings)
                for item in tile_blocks:
                    local = item["polygon"]
                    x1, y1, x2, y2 = box
                    polygon = [[min(width, x + x1), min(height, y + y1)] for x, y in local]
                    blocks.append({**item, "polygon": polygon, "box": _box_from_points(polygon), "_tile": calls,
                        "_edgeDistance": min(min(x for x, _ in local), min(y for _, y in local), x2 - x1 - max(x for x, _ in local), y2 - y1 - max(y for _, y in local))})
            finally:
                if target != source:
                    target.unlink(missing_ok=True)
    if not successful:
        raise RuntimeError(tr("SPOTTING_FAILED：所有图片分块均失败，请检查模型服务与输出预算。{details}", details=str(failed_tiles[:1])))
    if failed_tiles:
        warnings.append(tr("图片仅部分识别完成：部分分块失败、输出截断或总预算已用尽，请拆分图片后重试。"))
    blocks = _deduplicate_blocks(blocks)
    partial = bool(failed_tiles or warnings)
    return {"code": 100 if blocks or partial else 101, "data": blocks, "warnings": warnings, "partial": partial,
        "confidenceSupported": False,
        "metadata": {"source": str(request.get("originalPath") or source), "pipeline": "PaddleOCR-VL-1.6", "mode": tr('文字识别'), "failedTiles": failed_tiles,
            "textRecognition": {"sourceWidth": width, "sourceHeight": height, "coordinateScale": 1.0,
                "imageMaxPixels": SPOTTING_MAX_PIXELS, "minPixels": SPOTTING_MIN_PIXELS, "initialTiles": len(boxes), "calls": calls,
                "tiles": records, "overlapPixels": budgets.TILE_OVERLAP}}}


def _page_number(raw, fallback):
    value = raw.get("page_index", fallback - 1) if isinstance(raw, dict) else fallback - 1
    try:
        return int(value) + 1
    except (TypeError, ValueError):
        return fallback


def _page_dimension(raw, key):
    value = raw.get(key, 0) if isinstance(raw, dict) else 0
    if isinstance(value, list):
        value = value[0] if value else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _table_payload(raw_item, content):
    table = {"html": content if str(content).lstrip().startswith("<table") else None, "cells": None}
    for key in ("table_cells", "cells", "structure", "table_structure"):
        value = raw_item.get(key) if isinstance(raw_item, dict) else None
        if value is not None:
            table["cells"] = _json_value(value)
            break
    return table


def _clean_model_text(value):
    """Remove tokenizer stop markers that llama.cpp may expose in text."""
    return re.sub(r"(?:</s>|<\|endoftext\|>|<\|im_end\|>)", "", str(value or ""), flags=re.IGNORECASE)


def _clip_document_polygon(points, width, height):
    """Clip mapped geometry to known page bounds, retaining the official raw data."""
    points = _coerce_points(points)
    if width <= 0 or height <= 0 or len(points) < 3:
        return []
    # Intersect edges with the page rectangle instead of replacing a meaningful
    # contour with its bounding box or merely moving out-of-bounds vertices.
    for axis, boundary, upper in ((0, 0, False), (0, width, True), (1, 0, False), (1, height, True)):
        if not points:
            return []
        clipped = []
        previous = points[-1]
        previous_inside = previous[axis] <= boundary if upper else previous[axis] >= boundary
        for point in points:
            inside = point[axis] <= boundary if upper else point[axis] >= boundary
            if inside != previous_inside:
                fraction = (boundary - previous[axis]) / (point[axis] - previous[axis])
                intersection = [previous[i] + fraction * (point[i] - previous[i]) for i in (0, 1)]
                intersection[axis] = boundary
                clipped.append(intersection)
            if inside:
                clipped.append(point)
            previous, previous_inside = point, inside
        points = clipped
    rounded = []
    for x, y in points:
        point = [min(width, max(0, int(round(x)))), min(height, max(0, int(round(y))))]
        if not rounded or point != rounded[-1]:
            rounded.append(point)
    if len(rounded) > 1 and rounded[0] == rounded[-1]:
        rounded.pop()
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(rounded, rounded[1:] + rounded[:1]))
    return rounded if len(rounded) >= 3 and area else []


def _document_blocks(raw, page_number):
    values = raw.get("parsing_res_list") or []
    if not isinstance(values, list):
        raise ValueError(tr('INVALID_MODEL_OUTPUT：parsing_res_list 不是列表'))
    width, height = _page_dimension(raw, "width"), _page_dimension(raw, "height")
    blocks, warnings = [], []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            warnings.append(tr('第 {page_number} 页第 {index} 个官方块不是对象，已保留在 raw 中但未映射。', page_number=page_number, index=index + 1))
            continue
        label = str(item.get("block_label") or "unknown").strip().lower()
        category, category_name = LABEL_TO_CATEGORY.get(label, (label or "unknown", label or tr('未知')))
        original_bbox = _coerce_points(item.get("block_bbox"))
        original_polygon = _coerce_points(item.get("block_polygon_points")) or original_bbox
        clipped_bbox = _clip_document_polygon(original_bbox, width, height)
        bbox = []
        if clipped_bbox:
            xs, ys = zip(*clipped_bbox)
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
            bbox = [[left, top], [right, top], [right, bottom], [left, bottom]]
        polygon = _clip_document_polygon(original_polygon, width, height)
        if len(bbox) != 4 or len(polygon) < 3:
            warnings.append(tr('第 {page_number} 页第 {index} 个块坐标无效，已保留在 raw 中但未映射。', page_number=page_number, index=index + 1))
            continue
        content = _clean_model_text(item.get("block_content"))
        order = item.get("block_order")
        try:
            order = int(order) if order is not None else None
        except (TypeError, ValueError):
            order = None
        block_id = item.get("block_id")
        if block_id is None:
            # Paddle normally supplies block_id after its official
            # restructuring.  Keep a deterministic page-local ID if an older
            # result omits it; the untouched result remains in ``official``.
            block_id = f"{page_number}:{index}"
        group_id = item.get("group_id")
        if group_id is None:
            group_id = block_id
        block = {
            "id": block_id,
            "globalId": item.get("global_block_id"),
            "groupId": group_id,
            "globalGroupId": item.get("global_group_id"),
            "order": order,
            "category": category,
            "categoryName": category_name,
            "label": label,
            "bbox": bbox,
            "polygon": polygon,
            "content": content,
            "pipelineIndex": index,
            "table": _table_payload(item, content) if category == "table" else None,
            "formula": {"latex": content or None} if category == "formula" else None,
            "official": _json_value(item),
        }
        blocks.append(block)
    # Paddle assigns block_order only to Markdown text-flow labels; tables,
    # figures, headers and footers deliberately receive null.  In that case
    # parsing_res_list itself is the official full-layout order.
    if blocks and all(item["order"] is not None for item in blocks):
        blocks.sort(key=lambda item: (item["order"], str(item["id"])))
    return blocks, warnings


def _markdown_of(result):
    markdown_data = getattr(result, "markdown", None)
    if callable(markdown_data):
        markdown_data = markdown_data()
    if markdown_data is None:
        markdown_data = _get(result, "markdown", {})
    if not isinstance(markdown_data, dict):
        return "", {}
    text = _clean_model_text(markdown_data.get("markdown_texts") or markdown_data.get("markdown") or "")
    images = markdown_data.get("markdown_images") or markdown_data.get("images") or {}
    return text, images


def _safe_asset_name(name, fallback):
    value = Path(str(name or fallback)).name
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    if not value:
        value = fallback
    # _materialize_assets always encodes a PNG.  Do not retain an upstream
    # .jpg/.webp suffix for those bytes: exported Markdown consumers use the
    # extension as the MIME hint.
    return str(Path(value).with_suffix(".png"))


def _materialize_assets(markdown, images, source_stem):
    """物化前后都约束 HTTP job 预算，编码期间也不允许超限临时文件。"""
    global _asset_output_bytes, _asset_output_count, _asset_budget_exhausted
    if not images:
        return markdown, [], []
    parent = asset_paths.output_root(_asset_output_root) if _asset_output_root is not None else _work_root()
    root = parent / f"{source_stem}_{uuid.uuid4().hex[:10]}_assets"
    root.mkdir(parents=True, exist_ok=True)
    records, warnings = [], []
    iterable = images.items() if isinstance(images, dict) else enumerate(images)
    for index, image in iterable:
        key = index
        if isinstance(image, dict):
            image_obj = image.get("img")
            source_name = image.get("path") or key
        else:
            image_obj = image
            source_name = key
        save = getattr(image_obj, "save", None)
        if not callable(save):
            warnings.append(tr('图片资源 {index!s} 没有可保存的图像对象，已跳过资源导出。', index=index))
            continue
        name = _safe_asset_name(source_name, f"image_{len(records) + 1}.png")
        target = root / name
        number = 2
        while target.exists():
            target = root / f"{target.stem}_{number}{target.suffix}"
            number += 1
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            if _asset_output_root is not None:
                if _asset_output_count >= asset_paths.MAX_ASSETS or _asset_output_bytes >= asset_paths.MAX_JOB_BYTES:
                    raise ValueError("ASSET_OUTPUT_BUDGET: job asset count/bytes exhausted")
                limit = min(asset_paths.MAX_ASSET_BYTES, asset_paths.MAX_JOB_BYTES - _asset_output_bytes)
                with temporary.open("wb") as stream:
                    image_obj.save(asset_paths.LimitedWriter(stream, limit), format="PNG")
                size = temporary.stat().st_size
                if size > limit:
                    raise ValueError("ASSET_OUTPUT_BUDGET: encoded image exceeds budget")
                _asset_output_bytes += size
                _asset_output_count += 1
            else:
                image_obj.save(temporary, format="PNG")
            os.replace(temporary, target)
        except Exception as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            warnings.append(tr("保存图片资源 {name} 失败：{error}。", name=source_name, error=str(exc)))
            if _asset_output_root is not None:
                _asset_budget_exhausted = True
                warnings.append(tr("任务图片资源达到单图 16 MiB、累计 128 MiB 或 500 个资源的预算，已停止后续处理并保留已完成结果。"))
                break
            continue
        uri = target.resolve().as_uri()
        source_text = str(source_name).replace("\\", "/")
        markdown = markdown.replace(source_text, uri)
        records.append({
            "id": f"asset-{len(records) + 1}",
            "name": target.name,
            "path": str(target),
            "uri": uri,
            "source": source_text,
        })
    return markdown, records, warnings


def _document_page(source, options, original_page):
    _partial_failures.clear()
    _remaining()
    _progress(tr('检测版面与裁剪区域'))
    pages = list(_engine.predict(
        str(source), use_layout_detection=True, use_chart_recognition=True,
        use_seal_recognition=True, use_ocr_for_image_block=True, layout_shape_mode="poly",
        temperature=0, max_new_tokens=int(options.get("maxOutputTokens", 4096)), use_queues=False,
    ))
    if len(pages) != 1:
        raise ValueError(tr('INVALID_MODEL_OUTPUT：单页文档推理未返回单页结果，不能当作无字。'))
    if not pages:
        return {
            "code": 101, "text": "", "markdown": "", "warnings": [],
            "data": {"schemaVersion": 1, "pages": [], "blocks": [], "assets": []},
            "metadata": {"source": str(source), "pipeline": "PaddleOCR-VL-1.6"},
        }

    # This is the official page-level layout / crop / reading-order pipeline.
    # Calling restructure_pages after prediction retains Paddle's own IDs and
    # global ordering; no whole-page Markdown prompt is used here.
    # Keep the official result as one result *per source page*.  In addition
    # to being what the preview needs, this is where Paddle assigns global
    # block/group IDs and applies its table/title restructuring.  Asking it to
    # concatenate here would intentionally collapse all source pages into one
    # result and make ``data.pages`` lose their original page membership.
    _progress(tr('重组解析结果'))
    restructured = list(_engine.restructure_pages(
        pages,
        merge_tables=bool(options.get("mergeTables", True)),
        relevel_titles=bool(options.get("relevelTitles", True)),
        concatenate_pages=False,
    ))
    if not restructured:
        raise RuntimeError(tr('完整 PaddleOCR-VL Pipeline 没有返回重组结果'))

    warnings, page_data, flat_blocks = [], [], []
    page_markdowns = []
    for fallback_index, page in enumerate(restructured, 1):
        raw = _result_json(page)
        page_number = original_page
        raw["page_index"] = original_page - 1
        blocks, page_warnings = _document_blocks(raw, page_number)
        warnings.extend(page_warnings)
        plain_text = "\n".join(block["content"] for block in blocks if block["content"].strip())
        page_markdown, _ = _markdown_of(page)
        page_markdowns.append(page_markdown)
        page_data.append({
            "page": page_number,
            "width": _page_dimension(raw, "width"),
            "height": _page_dimension(raw, "height"),
            "text": plain_text,
            "markdown": page_markdown,
            "resources": [],
            "layout": _json_value(raw.get("layout_det_res")),
            "blocks": blocks,
            "raw": raw,
        })
        for block in blocks:
            flat_blocks.append({"page": page_number, **block})

    partial_blocks = []
    for failure in _partial_failures:
        details = dict(failure)
        page_index = failure.get("pageIndex", -1)
        block_index = failure.get("blockIndex", -1)
        block = None
        if isinstance(page_index, int) and isinstance(block_index, int) and 0 <= page_index < len(page_data):
            block = next(
                (
                    item
                    for item in page_data[page_index]["blocks"]
                    if item.get("pipelineIndex") == block_index
                ),
                None,
            )
        if block is not None:
            details.update(
                {
                    "page": page_data[page_index]["page"],
                    "id": block.get("id"),
                    "category": block.get("category"),
                    "categoryName": block.get("categoryName"),
                    "bbox": block.get("bbox"),
                    "polygon": block.get("polygon"),
                }
            )
            warnings.append(
                tr('第 {page} 页区域 {value} 的内容识别失败，已保留版面区域并置为空内容。', page=details['page'], value=details.get('id') or block_index)
            )
        else:
            warnings.append(
                tr('第 {page_index} 页第 {block_index} 个区域的内容识别失败，已保留原始版面结果。', page_index=page_index + 1, block_index=block_index + 1)
            )
        partial_blocks.append(details)

    merged_markdowns, markdown_images = [], {}
    for merged_page in restructured:
        markdown_text, images = _markdown_of(merged_page)
        if markdown_text.strip():
            merged_markdowns.append(markdown_text)
        if isinstance(images, dict):
            for key, image in images.items():
                # Paddle's path is the stable key used in markdown.  Preserve
                # the first image under a duplicated key rather than silently
                # overwrite a later page's resource.
                name = str(key)
                if name in markdown_images:
                    name = f"{len(markdown_images) + 1}_{name}"
                markdown_images[name] = image
        elif isinstance(images, list):
            for image in images:
                markdown_images[str(len(markdown_images) + 1)] = image
    concatenate_pages = bool(options.get("concatenatePages", True))
    if concatenate_pages:
        full_markdown = "\n\n".join(merged_markdowns)
    else:
        full_markdown = "\n\n".join(
            f"<!-- page {page['page']} -->\n{page['markdown']}"
            for page in page_data
            if page["markdown"].strip()
        )
    if not full_markdown.strip():
        full_markdown = "\n\n".join(text for text in page_markdowns if text.strip())
    full_markdown, assets, asset_warnings = _materialize_assets(
        full_markdown, markdown_images if options.get("exportAssets", True) else {}, source.stem
    )
    warnings.extend(asset_warnings)
    # Individual page Markdown is mapped to the same durable resource files,
    # but each page exposes only the assets it actually references.
    for page in page_data:
        text = page["markdown"]
        page_assets = []
        for asset in assets:
            source_ref = str(asset.get("source") or "")
            if source_ref and source_ref in text:
                page_assets.append(dict(asset))
            text = text.replace(asset["source"], asset["uri"])
        page["markdown"] = text
        page["resources"] = page_assets

    text_separator = "\n\n" if concatenate_pages else "\n\n\f\n\n"
    text = text_separator.join(page["text"] for page in page_data if page["text"].strip())
    if not text:
        text = "\n".join(block["content"] for block in flat_blocks if block["content"].strip())
    if not text and not full_markdown.strip() and not flat_blocks:
        return {
            "code": 101, "text": "", "markdown": "", "warnings": warnings,
            "data": {"schemaVersion": 1, "pages": page_data, "blocks": [], "assets": assets},
            "metadata": {"source": str(source), "pipeline": "PaddleOCR-VL-1.6"},
        }
    data = {
        "schemaVersion": 1,
        "pages": page_data,
        "blocks": flat_blocks,
        "assets": assets,
        # Compatibility fields for older document consumers.  They contain no
        # separate, simplified inference result; both point at DocumentResultV1.
        "text": text,
        "markdown": full_markdown,
    }
    return {
        "code": 100,
        "text": text,
        "markdown": full_markdown,
        "data": data,
        "warnings": warnings,
        "metadata": {
            "schema": "DocumentResultV1",
            "source": str(source),
            "sourceStem": source.stem,
            "pageCount": len(page_data),
            "blockCount": len(flat_blocks),
            "pipeline": "PaddleOCR-VL-1.6",
            "partialBlocks": partial_blocks,
            "mode": tr('文档解析'),
            "useLayoutDetection": True,
            "layoutShapeMode": "poly",
            "fullRegionRecognition": [tr('正文'), tr('标题'), tr('表格'), tr('公式'), tr('图表'), tr('印章'), tr('图片')],
            "processBoundary": "separate-worker-stdio-ipc",
            "pageInputPolicy": tr('原始页面交给官方版面分析；视觉预算仅由区域 VLM 请求处理。'),
            "concatenatePages": concatenate_pages,
        },
    }


def _combine_document(source, results, failed_pages, total):
    pages, blocks, assets, warnings, partial_blocks = [], [], [], [], []
    for result in results:
        data = result.get("data") or {}
        pages.extend(data.get("pages") or [])
        blocks.extend(data.get("blocks") or [])
        assets.extend(data.get("assets") or [])
        warnings.extend(result.get("warnings") or [])
        partial_blocks.extend((result.get("metadata") or {}).get("partialBlocks") or [])
    # 每页局部 asset-N 会冲突；按任务生成唯一 id，同时保持页面资源引用一致。
    for index, asset in enumerate(assets, 1):
        asset["id"] = "asset-%d" % index
    by_path = {asset.get("path"): asset for asset in assets}
    for page in pages:
        page["resources"] = [dict(by_path.get(asset.get("path"), asset)) for asset in page.get("resources", [])]
    if failed_pages:
        warnings.append(tr("文档仅部分完成：部分页面失败或总执行预算已用尽，请缩小页范围后重试。"))
    text = "\n\n".join(page.get("text", "") for page in pages if page.get("text"))
    markdown = "\n\n".join(page.get("markdown", "") for page in pages if page.get("markdown"))
    partial = bool(failed_pages or partial_blocks or warnings)
    return {"code": 100 if text or markdown or blocks or partial else 101, "partial": partial,
        "text": text, "markdown": markdown, "warnings": warnings,
        "data": {"schemaVersion": 1, "pages": pages, "blocks": blocks, "assets": assets, "text": text, "markdown": markdown},
        "metadata": {"schema": "DocumentResultV1", "source": str(source), "sourceStem": source.stem,
            "pageCount": len(pages), "selectedPageCount": total, "blockCount": len(blocks),
            "pipeline": "PaddleOCR-VL-1.6", "partialBlocks": partial_blocks, "failedPages": failed_pages,
            "mode": tr('文档解析'), "useLayoutDetection": True, "processBoundary": "separate-worker-stdio-ipc"}}


def _restructure_completed(results, options):
    """跨页只重组已完成 JSON，不保留所有页面解码图；仍使用 vendor 正式 API。"""
    if len(results) < 2:
        return
    originals = [page for result in results for page in result["data"]["pages"]]
    raw_pages = [{"res": dict(page["raw"])} for page in originals]
    restructured = list(_engine.restructure_pages(raw_pages, merge_tables=bool(options.get("mergeTables", True)),
        relevel_titles=bool(options.get("relevelTitles", True)), concatenate_pages=False))
    if len(restructured) != len(originals):
        raise ValueError(tr('INVALID_MODEL_OUTPUT：跨页重组改变页数，已保留逐页结果。'))
    prepared = []
    for record, result in zip(originals, restructured):
        raw = _result_json(result)
        raw["page_index"] = record["page"] - 1
        mapping_raw = raw
        width, height = _page_dimension(record, "width"), _page_dimension(record, "height")
        if width > 0 and height > 0:
            mapping_raw = {**raw, "width": width, "height": height}
        blocks, warnings = _document_blocks(mapping_raw, record["page"])
        markdown, images = _markdown_of(result)
        resources = record.get("resources", [])
        for reference in images if isinstance(images, dict) else []:
            candidates = [asset for asset in resources if asset.get("source") == reference]
            if not candidates:
                # 官方重组可按 bbox 重建资源名并改变扩展名；只在本页唯一同名块间关联。
                candidates = [asset for asset in resources if Path(asset.get("source", "")).stem == Path(reference).stem]
            if len(candidates) == 1:
                asset = candidates[0]
                aliases = asset.setdefault("sourceReferences", [asset["source"]])
                if reference not in aliases:
                    aliases.append(reference)
                markdown = markdown.replace(reference, asset["uri"])
            elif resources:
                warnings.append(tr('第 {page} 页重组后的图片资源无法唯一关联：{reference}', page=record["page"], reference=reference))
        for asset in resources:
            for reference in [asset["source"], *asset.get("sourceReferences", [])]:
                markdown = markdown.replace(reference, asset["uri"])
        prepared.append((record, raw, blocks, markdown, warnings))
    for record, raw, blocks, markdown, warnings in prepared:
        record.update(raw=raw, blocks=blocks, markdown=markdown, text="\n".join(b["content"] for b in blocks if b["content"].strip()))
    page_warnings = {id(record): warnings for record, _, _, _, warnings in prepared}
    for result in results:
        for page in result["data"]["pages"]:
            for warning in page_warnings[id(page)]:
                if warning not in result.setdefault("warnings", []):
                    result["warnings"].append(warning)
        result["data"]["blocks"] = [{"page": page["page"], **block} for page in result["data"]["pages"] for block in page["blocks"]]


def _document(request):
    if _engine is None or _mode != "document":
        raise RuntimeError("Document worker engine is not initialized")
    _begin_task(request, 1800)
    source = Path(str(request["path"])).expanduser().resolve()
    original_source = Path(str(request.get("originalPath") or source))
    options = dict(request.get("options") or {})
    results, failures, selection = [], [], []
    with tempfile.TemporaryDirectory(prefix="document-pages-", dir=_work_root()) as temporary:
        with budgets.document_pages(source, options.get("pageRange"), temporary, _task_deadline,
                                    prepared=bool(request.get("preparedInput"))) as (selection, inputs):
            iterator = iter(inputs)
            try:
                for number in selection:
                    try:
                        _remaining()
                        rendered_number, page_source = next(iterator)
                        if rendered_number != number:
                            raise ValueError(tr('INVALID_MODEL_OUTPUT：页号与渲染输入不一致'))
                    except Exception as exc:
                        failures.extend({"page": p, "error": str(exc)[:500]} for p in selection[selection.index(number):])
                        break
                    try:
                        result = _document_page(page_source, options, number)
                        result.setdefault("metadata", {}).update(source=str(original_source), sourceStem=original_source.stem)
                        results.append(result)
                        _progress(tr('页面解析完成'), page=number, completedPages=len(results), totalPages=len(selection),
                            result=result, warnings=result.get("warnings", []), failed=False)
                        if _asset_budget_exhausted:
                            failures.extend({"page": p, "error": "ASSET_OUTPUT_BUDGET"} for p in selection[selection.index(number) + 1:])
                            break
                    except Exception as exc:
                        failure = {"page": number, "error": str(exc)[:500]}
                        failures.append(failure)
                        _progress(tr('页面解析失败'), page=number, completedPages=len(results), totalPages=len(selection),
                            result=None, warnings=[str(exc)], failed=True)
                        if time.monotonic() >= _task_deadline:
                            failures.extend({"page": p, "error": "TASK_DEADLINE"} for p in selection[selection.index(number) + 1:])
                            break
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
    if not results and failures:
        return {"code": 908 if time.monotonic() >= _task_deadline else 904,
            "data": tr("文档没有页面成功解析，请检查模型服务、输入文件与执行预算。"), "partial": False,
            "warnings": [item["error"] for item in failures], "metadata": {"failedPages": failures},
            "errorType": "TASK_DEADLINE" if time.monotonic() >= _task_deadline else "PIPELINE_ERROR"}
    try:
        _restructure_completed(results, options)
    except Exception as exc:
        if results:
            results[0].setdefault("warnings", []).append(tr("跨页重组失败，已保留逐页结果：{error}", error=str(exc)))
    final = _combine_document(original_source, results, failures, len(selection))
    final["metadata"]["concatenatePages"] = bool(options.get("concatenatePages", True))
    if _asset_output_root is not None:
        final["metadata"]["assetBudget"] = {"bytes": _asset_output_bytes, "count": _asset_output_count,
            "maxBytes": asset_paths.MAX_JOB_BYTES, "maxAssetBytes": asset_paths.MAX_ASSET_BYTES, "maxAssets": asset_paths.MAX_ASSETS}
    if not options.get("concatenatePages", True):
        final["text"] = "\n\n\f\n\n".join(page["text"] for page in final["data"]["pages"] if page["text"])
        final["markdown"] = "\n\n".join("<!-- page %s -->\n%s" % (page["page"], page["markdown"]) for page in final["data"]["pages"] if page["markdown"])
        final["data"].update(text=final["text"], markdown=final["markdown"])
    return final


_protocol_stdout = sys.stdout

def _progress(stage, **details):
    _protocol_stdout.write(json.dumps({"event": "progress", "stage": tr(stage), **details}, ensure_ascii=False, allow_nan=False) + "\n")
    _protocol_stdout.flush()


def _reply(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    sys.stdout.flush()


def _error_type(message):
    text = str(message or "")
    for prefix in ("INVALID_MODEL_OUTPUT", "INVALID_PAGE_RANGE", "INPUT_FILE_BUDGET", "IMAGE_DECODE_BUDGET", "TILE_COUNT_BUDGET", "DOCUMENT_PAGE_BUDGET", "DOCUMENT_PIXEL_BUDGET", "TASK_DEADLINE", "OUTPUT_TRUNCATED", "INVALID_CONFIGURATION", "ASSET_OUTPUT_BUDGET", "ASSET_OUTPUT_INVALID"):
        if text.startswith(prefix):
            return prefix
    lower = text.lower()
    if "cuda" in lower or "gpu:" in lower or "driver" in lower:
        return "GPU_INITIALIZATION_FAILED"
    if "inference.json" in lower or "版面检测模型" in text:
        return "LAYOUT_MODEL_MISSING"
    if "no module named" in lower or "cannot import" in lower or "导入" in text:
        return "PIPELINE_DEPENDENCY_MISSING"
    return "PIPELINE_ERROR"


def main():
    global _engine, _config, _mode
    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request.get("op")
            if op == "start":
                _stop_engine()
                started = time.perf_counter_ns()
                _config = dict(request.get("config") or {})
                _mode = str(_config.get("mode") or "")
                with contextlib.redirect_stdout(sys.stderr):
                    _progress(tr('加载版面模型') if _mode == "document" else tr('准备文字定位模型'))
                    _engine = _create_engine(_config)
                _reply({
                    "ok": True,
                    "status": "loaded",
                    "mode": _mode,
                    "runtime_archives": list(_runtime_archive_paths),
                    "archive_modules": {
                        name: str(module.__file__) for name, module in tuple(sys.modules.items())
                        if getattr(module, "__file__", None) and any(
                            str(module.__file__).replace("\\", "/").startswith(path.replace("\\", "/") + "/")
                            for path in _runtime_archive_paths)
                    },
                    "model_initialization_ms": round((time.perf_counter_ns() - started) / 1_000_000, 3),
                })
            elif op == "probe":
                with contextlib.redirect_stdout(sys.stderr):
                    result = _probe_dependencies(request.get("config") or {})
                _reply({"ok": True, "result": result})
                return
            elif op == "spotting":
                started = time.perf_counter_ns()
                with contextlib.redirect_stdout(sys.stderr):
                    result = _spotting(request)
                _reply({"ok": True, "result": result, "timing": {"total_ms": round((time.perf_counter_ns() - started) / 1_000_000, 3)}})
            elif op in ("document", "parse"):
                started = time.perf_counter_ns()
                with contextlib.redirect_stdout(sys.stderr):
                    result = _document(request)
                _reply({"ok": True, "result": result, "timing": {"total_ms": round((time.perf_counter_ns() - started) / 1_000_000, 3)}})
            elif op == "stop":
                _stop_engine()
                _reply({"ok": True, "status": "stopped"})
                return
            else:
                raise ValueError(f"unknown worker operation: {op}")
        except Exception as exc:
            message = str(exc)
            _reply({"ok": False, "error": tr(message), "errorType": _error_type(message)})


if __name__ == "__main__":
    main()

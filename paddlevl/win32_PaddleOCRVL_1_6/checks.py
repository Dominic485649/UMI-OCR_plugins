# ==========================================================
# =============== 外部依赖检测（路径 / 版本 / 协议） ===============
# ==========================================================

"""本插件依赖的模型与服务【全部由用户自备】，因此"检测"是插件的核心职责之一。

模块只依赖标准库，可以在没有 paddle / paddleocr 的环境里独立运行；
健康检查、页控制器、后端实现和回归测试都复用同一份检测逻辑，
避免"页面说没问题、解析时又报另一个错"这种不一致。

三类检测缺一不可：

    路径  文件/目录是否存在、是不是预期的类型、内容是否完整
    版本  GGUF 容器版本、llama.cpp 服务端构建、Python 依赖版本
    协议  服务是否真的说 OpenAI 兼容的那套话、模型列表是否可用

【每一条失败都必须给出可操作的下一步】——具体改哪个配置项、把哪个文件放到
哪里、装哪个版本——而不是只丢一个错误码。这是本模块存在的理由。
"""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading
import socket
import struct
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ID = os.path.basename(PLUGIN_DIR)
MANIFEST_PATH = os.path.join(PLUGIN_DIR, "umi-plugin.json")

# GGUF 容器格式。v1 与 v2 的 KV/张量计数字段宽度与 v3 不同，
# 当前的多模态 mmproj 一律是 v3；低于 v3 直接判定为不可用并说明原因。
GGUF_MAGIC = b"GGUF"
GGUF_MIN_VERSION = 3
GGUF_MAX_VERSION = 3

# 依赖版本下限。低于这些版本没有 v1.6 流水线，或者版面检测子模块的
# 配置键名不同，硬跑只会得到难以解释的 KeyError。
REQUIRED_PACKAGES = (
    ("paddleocr", (3, 7, 0)),
    ("paddlex", (3, 7, 2)),
    ("paddle", (0, 0, 0)),
)

ERROR_TYPES = {
    901: "INVALID_CONFIGURATION",
    902: "LLAMA_CPP_INCOMPATIBLE",
    903: "MODEL_NOT_READY",
    904: "PIPELINE_ERROR",
    905: "PIPELINE_DEPENDENCY_MISSING",
    908: "REQUEST_TIMEOUT",
}

OPTION_ERROR_TYPES = {
    "service.mainModel": "MODEL_FILE_MISSING",
    "service.mmprojModel": "MMPROJ_FILE_MISSING",
    "pipeline.layoutModelDir": "LAYOUT_MODEL_MISSING",
}


def errorTypeFor(code, fallback="UNKNOWN_ERROR", option=None):
    """Stable machine-readable error type beside Umi-OCR's numeric codes."""
    return OPTION_ERROR_TYPES.get(str(option or ""), ERROR_TYPES.get(code, fallback))


class CheckError(Exception):
    """一次检测失败。message 必须是可操作的中文指引。"""

    def __init__(self, message, code=901, **details):
        super().__init__(message)
        self.message = message
        self.code = code
        self.errorType = str(details.get("errorType") or errorTypeFor(code, option=details.get("option")))
        # Callers convert this exception back into checks.fail(..., **details).
        # Preserve its type at that common boundary instead of requiring each
        # caller to remember a second argument.
        self.details = {**details, "errorType": self.errorType}


def ok(message, **details):
    return {"ok": True, "message": message, "details": details}


def fail(message, errorType="UNKNOWN_ERROR", **details):
    """Return one consistent error shape without dropping a CheckError type."""
    errorType = str(errorType or "UNKNOWN_ERROR")
    return {"ok": False, "message": message, "errorType": errorType, "details": details}


# ========================= 【清单：唯一事实源】 =========================


_manifestCache = None


def loadManifest():
    """读取插件自己的 umi-plugin.json 。"""
    global _manifestCache
    if _manifestCache is None:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as stream:
            _manifestCache = json.load(stream)
    return _manifestCache


def _flattenOptions(node, prefix, out):
    for key, value in node.items():
        if key in ("title", "type"):
            continue
        if not isinstance(value, dict):
            continue
        path = f"{prefix}{key}"
        if value.get("type") == "group":
            _flattenOptions(value, f"{path}.", out)
        elif "default" in value:
            out[path] = value["default"]
    return out


def manifestDefaults():
    """把 capability 与共享服务的默认值摊平。

    宿主将 pluginServices.<插件>.<服务>.* 去掉前缀后与 capability 参数合并，
    因此所有默认值仍只以清单为事实源。
    """
    manifest = loadManifest()
    defaults = {}
    for service in manifest.get("services", []):
        config = service.get("config")
        if isinstance(config, dict):
            _flattenOptions(config, "", defaults)
    for capability in manifest.get("capabilities", []):
        config = capability.get("config")
        if not isinstance(config, dict):
            continue
        for section in ("global_options", "local_options"):
            node = config.get(section)
            if isinstance(node, dict):
                _flattenOptions(node, "", defaults)
    return defaults


# 旧配置保留在用户文件中，但绝不再参与运行路径或服务所有权判断。
RETIRED_KEYS = frozenset((
    "service.mode", "service.host", "service.serverExecutable",
    "lifecycle.prestartMode", "lifecycle.prestartDelayMinutes",
    "lifecycle.autoStopMode", "lifecycle.idleStopMinutes",
    "resources.gpuLayers", "resources.imageMaxTokens",
))


def mergeArgd(*argds):
    """只在兼容边界忽略具名旧键，不删除用户配置。"""
    merged = manifestDefaults()
    for argd in argds:
        if argd:
            merged.update({k: v for k, v in argd.items() if v is not None and k not in RETIRED_KEYS})
    return merged


_translationCache = {}


def tr(template, **values):
    """先翻译模板再插值；独立 worker 通过环境传递宿主语言，不导入 Qt。"""
    try:
        try:
            # 正式宿主只更新 plain 模块，不能先导入 canonical 创建第二份语言状态。
            from plugin_i18n import Translator, LangCode
        except ImportError:
            from py_src.imports.plugin_i18n import Translator, LangCode
        key = ("host", LangCode)
        if key not in _translationCache:
            _translationCache[key] = Translator(__file__, "i18n.csv")
        template = _translationCache[key](template)
    except ImportError:
        import csv
        language = os.environ.get("UMI_PLUGIN_LANGUAGE", "zh_CN")
        key = ("worker", language)
        if key not in _translationCache:
            translations = {}
            try:
                with open(os.path.join(PLUGIN_DIR, "i18n.csv"), encoding="utf-8", newline="") as stream:
                    rows = csv.reader(stream)
                    header = next(rows)
                    index = header.index(language) if language in header else 0 if language.startswith("zh_") else 1
                    translations = {row[0]: row[index] for row in rows if len(row) > index and row[index]}
            except (OSError, StopIteration, ValueError):
                pass
            _translationCache[key] = translations
        template = _translationCache[key].get(template, template)
    return template.format(**values) if values else template


_runtimeCache = {}
_runtimeLock = threading.RLock()


def _fileStamp(path):
    info = path.stat()
    return (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _runtimePath(root, relative):
    """拒绝绝对路径、ADS、遍历和所有 reparse point（包括 junction）。"""
    relative = str(relative)
    parts = PurePosixPath(relative).parts
    if not parts or "\\" in relative or ":" in relative or relative.startswith("/") or any(p in (".", "..") for p in parts):
        raise ValueError("invalid relative runtime path")
    target = root.joinpath(*parts)
    current = Path(PLUGIN_DIR).absolute()
    root_info = current.lstat()
    if stat.S_ISLNK(root_info.st_mode) or getattr(root_info, "st_file_attributes", 0) & 0x400:
        raise ValueError("plugin root is a reparse point")
    for part in target.relative_to(current).parts:
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("runtime reparse point is not allowed")
    target.resolve(strict=True).relative_to(Path(PLUGIN_DIR).resolve(strict=True))
    if not target.is_file():
        raise ValueError("runtime entry is not a file")
    return target


def bundledRuntimeStamp():
    """配置指纹可以用于未安装完整的插件，不在 configure 阶段运行 EXE。"""
    path = Path(PLUGIN_DIR) / "resources" / "llama.cpp" / "runtime-manifest.json"
    try:
        if path.stat().st_size > 1_048_576:
            return str(path), "invalid"
        # 清单很小，其摘要同时绑定发行 identity 与文件哈希，不以用户路径作指纹。
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return str(path), None


def bundledRuntime():
    """核验固定内置运行时；每次检查归属/stat，仅文件改变时重新 SHA-256。"""
    root = Path(PLUGIN_DIR).absolute() / "resources" / "llama.cpp"
    try:
        with _runtimeLock:
            manifest = _runtimePath(root, "runtime-manifest.json")
            if manifest.stat().st_size > 1_048_576:
                raise ValueError("runtime manifest is too large")
            payload_bytes = manifest.read_bytes()
            payload = json.loads(payload_bytes)
            identity = payload.get("identity") or {}
            if payload.get("schemaVersion") != 1 or not re.fullmatch(r"[0-9a-fA-F]{40}", str(identity.get("commit", ""))):
                raise ValueError("invalid runtime schema or commit")
            architectures = identity.get("architectures")
            if not isinstance(architectures, list) or "75" not in architectures or not all(isinstance(a, str) for a in architectures):
                raise ValueError("runtime must declare Turing sm75 support")
            if not isinstance(identity.get("cudaToolkit"), str) or not identity["cudaToolkit"] or not isinstance(identity.get("buildOptions"), dict):
                raise ValueError("runtime build identity is incomplete")
            entries = payload.get("files")
            if not isinstance(entries, list) or not entries or len(entries) > 4096:
                raise ValueError("invalid runtime file list")
            declared = set()
            for entry in entries:
                relative = entry.get("path", "")
                folded = relative.casefold()
                if folded in declared or folded == "runtime-manifest.json":
                    raise ValueError("duplicate or self-referencing runtime entry")
                declared.add(folded)
                target = _runtimePath(root, relative)
                expected = entry.get("sha256", "")
                if type(entry.get("size")) is not int or entry["size"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", expected):
                    raise ValueError("invalid runtime hash or size")
                stamp = _fileStamp(target)
                if stamp[1] != entry["size"]:
                    raise ValueError("runtime file size mismatch: " + relative)
                cache_key = str(target)
                cached = _runtimeCache.get(cache_key)
                if cached != (stamp, expected):
                    digest = hashlib.sha256()
                    with target.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1_048_576), b""):
                            digest.update(chunk)
                    if _fileStamp(target) != stamp or digest.hexdigest() != expected:
                        raise ValueError("runtime file hash mismatch: " + relative)
                    _runtimeCache[cache_key] = (stamp, expected)
            # 额外 DLL 同样可能被 Windows 加载，不能只验证清单列出的子集。
            actual = set()
            for current, dirs, names in os.walk(root, followlinks=False):
                for name in dirs:
                    item = Path(current) / name
                    info = item.lstat()
                    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                        raise ValueError("runtime directory is a reparse point")
                for name in names:
                    relative = (Path(current) / name).relative_to(root).as_posix()
                    if relative.casefold() != "runtime-manifest.json":
                        _runtimePath(root, relative)
                        actual.add(relative.casefold())
                if len(actual) > 4096:
                    raise ValueError("too many runtime files")
            if actual != declared or "llama-server.exe" not in actual or not any("license" in p.lower() for p in actual):
                raise ValueError("runtime files/licenses do not match manifest")
            return {
                "path": str(root / "llama-server.exe"), "root": str(root),
                "identity": identity, "manifestSha256": hashlib.sha256(payload_bytes).hexdigest(),
                "fileCount": len(entries), "minimumComputeCapability": "7.5",
            }
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise CheckError(tr("内置 llama.cpp 运行时缺失或损坏：{error}。请完整重新安装 PaddleVL 插件，不要选择外部程序。", error=str(exc)), 902, errorType="BUNDLED_RUNTIME_INVALID") from exc


def resourceBudget(argd, local=True):
    """llama 的 -c 是所有槽共享总量；为视觉和提示预留实际联合预算。"""
    context = timeoutSeconds(argd, "resources.contextLength", 16384, 2048, 131072)
    slots = timeoutSeconds(argd, "resources.slots", 1, 1, 8)
    service_output = timeoutSeconds(argd, "resources.maxOutputTokens", 4096, 256, 65536)
    output = timeoutSeconds(argd, "maxOutputTokens", service_output, 256, 65536) if local else service_output
    per_slot = context // slots
    # Paddle 分支固定 1605632 pixels，对应最多 2048 视觉 tokens；额外留 512 提示。
    reserve = 2048 + 512
    if output > service_output or output + reserve > per_slot:
        raise CheckError(tr("输出预算 {output} 超出服务预算或单槽上下文 {context}（须预留 {reserve} 个视觉与提示 token）。请减少输出、槽数或增加上下文。", output=output, context=per_slot, reserve=reserve), option="maxOutputTokens")
    return {"contextLength": context, "slots": slots, "perSlotContext": per_slot, "maxOutputTokens": output, "reservedInputTokens": reserve}


def taskSeconds(argd, default=1800):
    # 任务参数只能收紧预算；排队耗时由 acquire 单独约束。
    return timeoutSeconds(argd, "taskTimeoutSeconds", default, 1, default)


# ========================= 【路径】 =========================


def requireFile(argd, key, title, suffixes=None, hint=""):
    raw = str(argd.get(key, "") or "").strip()
    if not raw:
        raise CheckError(
            tr('未设置【{title}】。请在【全局设置 → 插件管理 → {plugin}】里选择该文件。{hint}', title=title, plugin=loadManifest().get('name', PLUGIN_ID), hint=hint),
            code=901,
            option=key,
        )
    path = os.path.abspath(os.path.expanduser(raw))
    if os.path.isdir(path):
        raise CheckError(
            tr('【{title}】指向的是一个目录，不是文件：{path} 。请重新选择具体文件。', title=title, path=path),
            code=901,
            option=key,
            path=path,
        )
    if not os.path.isfile(path):
        raise CheckError(
            tr('【{title}】不存在：{path} 。文件可能被移动、改名或删除；请重新选择正确的路径。{hint}', title=title, path=path, hint=hint),
            code=901,
            option=key,
            path=path,
        )
    if suffixes and os.path.splitext(path)[1].lower() not in suffixes:
        expect = " / ".join(sorted(suffixes))
        raise CheckError(
            tr('【{title}】的扩展名不是 {expect} ：{path} 。请确认选择的文件类型正确。', title=title, expect=expect, path=path),
            code=901,
            option=key,
            path=path,
        )
    if os.path.getsize(path) == 0:
        raise CheckError(
            tr('【{title}】是空文件：{path} 。通常是下载或复制中断，请重新获取该文件。', title=title, path=path),
            code=901,
            option=key,
            path=path,
        )
    return path


def requireModelDir(argd, key, title):
    """版面检测模型目录：必须是目录，且含 PaddleX 推理配置。"""
    raw = str(argd.get(key, "") or "").strip()
    if not raw:
        raise CheckError(
            tr('未设置【{title}】。本插件不含版面检测模型，请自行准备一个 PaddleX 推理模型目录（目录内含 inference.json），并在全局设置里指向它。', title=title),
            code=901,
            option=key,
        )
    path = os.path.abspath(os.path.expanduser(raw))
    if not os.path.isdir(path):
        raise CheckError(
            tr('【{title}】不是目录：{path} 。这里需要的是【解压后的模型目录】，不是压缩包，也不是单个模型文件。', title=title, path=path),
            code=901,
            option=key,
            path=path,
        )
    config = os.path.join(path, "inference.json")
    if not os.path.isfile(config):
        entries = sorted(os.listdir(path))[:8]
        nested = [n for n in entries if os.path.isfile(os.path.join(path, n, "inference.json"))]
        extra = (
            tr('检测到子目录 {value} 里才有 inference.json ，请改为选择该子目录。', value=nested[0])
            if nested
            else tr('当前目录内容：{value} 。', value=', '.join(entries) or tr('(空)'))
        )
        raise CheckError(
            tr('【{title}】中缺少 inference.json ：{path} 。{extra}', title=title, path=path, extra=extra),
            code=901,
            option=key,
            path=path,
        )
    for name in ("inference.json", "inference.pdiparams", "inference.yml"):
        artifact = os.path.join(path, name)
        if not os.path.isfile(artifact) or os.path.getsize(artifact) == 0:
            raise CheckError(tr('【{title}】缺少有效模型文件：{artifact}', title=title, artifact=artifact), code=901, option=key, path=path)
    return path


# ========================= 【版本：GGUF 容器】 =========================


def _readGgufString(stream):
    raw = stream.read(8)
    if len(raw) < 8:
        raise CheckError(tr('GGUF 头部在字符串长度处被截断。'), code=901)
    (length,) = struct.unpack("<Q", raw)
    if length > 1 << 20:
        raise CheckError(tr('GGUF 头部声明了异常长的字符串，文件可能已损坏。'), code=901)
    return stream.read(length).decode("utf-8", errors="replace")


# GGUF KV 值类型编号 → struct 格式；复合类型（数组=9）单独处理。
_GGUF_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}


def _skipGgufValue(stream, valueType):
    if valueType in _GGUF_SCALARS:
        fmt = _GGUF_SCALARS[valueType]
        stream.read(struct.calcsize(fmt))
        return
    if valueType == 8:  # string
        _readGgufString(stream)
        return
    if valueType == 9:  # array
        header = stream.read(12)
        if len(header) < 12:
            raise CheckError(tr('GGUF 头部在数组处被截断。'), code=901)
        itemType, count = struct.unpack("<IQ", header)
        if count > 1 << 24:
            raise CheckError(tr('GGUF 头部声明了异常长的数组，文件可能已损坏。'), code=901)
        for _ in range(count):
            _skipGgufValue(stream, itemType)
        return
    raise CheckError(tr('GGUF 头部含未知的值类型 {valuetype} ，文件可能已损坏。', valuetype=valueType), code=901)


def readGgufHeader(path, maxKeys=512):
    """只读 GGUF 文件头，不读张量数据。

    返回 {"version", "tensorCount", "kvCount", "architecture", "keys"} 。
    读头部意味着即使是几十 GB 的模型，检测也只有毫秒级开销。
    """
    with open(path, "rb") as stream:
        magic = stream.read(4)
        if magic != GGUF_MAGIC:
            raise CheckError(
                tr('{filename} 不是 GGUF 文件（文件头是 {magic!r}）。请确认下载完整、没有选成 safetensors / bin / 压缩包。', filename=os.path.basename(path), magic=magic),
                code=901,
                path=path,
            )
        head = stream.read(20)
        if len(head) < 20:
            raise CheckError(
                tr('{filename} 的 GGUF 头部不完整，文件很可能没下载完。', filename=os.path.basename(path)),
                code=901,
                path=path,
            )
        version, tensorCount, kvCount = struct.unpack("<IQQ", head)
        info = {
            "version": version,
            "tensorCount": tensorCount,
            "kvCount": kvCount,
            "architecture": "",
            "keys": [],
        }
        if not GGUF_MIN_VERSION <= version <= GGUF_MAX_VERSION:
            return info  # 版本判定交给调用方，先把信息带回去
        for _ in range(min(kvCount, maxKeys)):
            try:
                key = _readGgufString(stream)
                rawType = stream.read(4)
                if len(rawType) < 4:
                    break
                (valueType,) = struct.unpack("<I", rawType)
                if key == "general.architecture" and valueType == 8:
                    info["architecture"] = _readGgufString(stream)
                else:
                    _skipGgufValue(stream, valueType)
                info["keys"].append(key)
            except CheckError:
                break
            except Exception:
                break
    return info


def checkGguf(path, title, expectProjector=False):
    """校验 GGUF 容器版本，并粗判主模型 / 投影文件是否放反了。"""
    info = readGgufHeader(path)
    version = info["version"]
    if version < GGUF_MIN_VERSION:
        raise CheckError(
            tr('【{title}】是 GGUF v{version} ，本插件需要 v{gguf_min_version} 。这是 2023 年的旧容器格式，多模态投影无法加载；请重新获取按新版 llama.cpp 转换的 GGUF 文件。', title=title, version=version, gguf_min_version=GGUF_MIN_VERSION),
            code=901,
            path=path,
            ggufVersion=version,
        )
    if version > GGUF_MAX_VERSION:
        raise CheckError(
            tr('【{title}】是 GGUF v{version} ，比本插件已知的 v{gguf_max_version} 更新。请升级插件，或改用与你的 llama.cpp 服务端匹配的 GGUF 文件。', title=title, version=version, gguf_max_version=GGUF_MAX_VERSION),
            code=901,
            path=path,
            ggufVersion=version,
        )
    architecture = (info.get("architecture") or "").lower()
    isProjector = "clip" in architecture or "proj" in architecture or bool(
        [k for k in info["keys"] if k.startswith("clip.")]
    )
    if expectProjector and architecture and not isProjector:
        raise CheckError(
            tr('【{title}】看起来不是视觉投影文件（general.architecture = {architecture}）。视觉投影文件通常名为 mmproj-*.gguf ；请检查是否把主模型误选到了这一栏。', title=title, architecture=info['architecture']),
            code=901,
            path=path,
            architecture=info["architecture"],
        )
    if not expectProjector and isProjector:
        raise CheckError(
            tr('【{title}】看起来是视觉投影文件（general.architecture = {architecture}）。请把它填到【视觉投影 GGUF】一栏，并在这一栏选择主模型。', title=title, architecture=info['architecture']),
            code=901,
            path=path,
            architecture=info["architecture"],
        )
    return info


# ========================= 【版本：llama.cpp 服务端】 =========================


def _runBinary(path, args, timeout=20):
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            [path, *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            shell=False,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired:
        raise CheckError(
            tr('运行 {filename} {items} 超过 {timeout} 秒无响应。该可执行文件可能不是 llama.cpp 服务端，或正在等待输入。', filename=os.path.basename(path), items=' '.join(args), timeout=timeout),
            code=901,
            path=path,
        )
    except OSError as exc:
        raise CheckError(
            tr('无法运行 {filename} ：{error}。请确认它是 Windows x86_64 可执行文件，且未被杀毒软件拦截。', filename=os.path.basename(path), error=exc),
            code=901,
            path=path,
        )
    return completed.returncode, completed.stdout.decode("utf-8", errors="replace")


def checkServerBinary(path, timeout=20):
    """确认服务端能跑起来，并且支持多模态所需的 --mmproj 参数。

    只跑 --version / --help ，不加载任何模型，因此几乎没有开销。
    """
    details = {"path": path}
    code, output = _runBinary(path, ["--version"], timeout)
    text = output.strip()
    details["versionOutput"] = text[-400:]
    if code != 0 and not text:
        raise CheckError(
            tr('{filename} --version 以代码 {code} 退出且没有任何输出。请完整重新安装配套 PaddleVL 插件，并检查 NVIDIA 驱动与运行库；不要选择外部程序。', filename=os.path.basename(path), code=code),
            code=901,
            **details,
        )
    _, helpText = _runBinary(path, ["--help"], timeout)
    details["supportsMmproj"] = "--mmproj" in helpText
    details["supportsHost"] = "--host" in helpText
    details["flags"] = {
        "model": "--model" in helpText or "-m," in helpText,
        "imageMaxTokens": "--image-max-tokens" in helpText,
        "mmprojOffload": "--mmproj-offload" in helpText,
        "noMmprojOffload": "--no-mmproj-offload" in helpText,
        "flashAttention": "--flash-attn" in helpText or "-fa" in helpText,
        "kvOffload": "--kv-offload" in helpText,
        "noKvOffload": "--no-kv-offload" in helpText,
        "cacheKey": "--cache-type-k" in helpText or "-ctk" in helpText,
        "cacheValue": "--cache-type-v" in helpText or "-ctv" in helpText,
        "sleepIdle": "--sleep-idle-seconds" in helpText,
        "jinja": "--jinja" in helpText,
        "cors": "--cors-origins" in helpText,
        "noUi": "--no-ui" in helpText,
        "alias": "--alias" in helpText or " -a" in helpText,
        "batch": "--batch-size" in helpText or " -b" in helpText,
        "ubatch": "--ubatch-size" in helpText or " -ub" in helpText,
        "temperature": "--temp" in helpText or "--temperature" in helpText,
        # PaddleOCR-VL Spotting emits its text/location delimiters as special
        # tokens.  Without this server capability llama.cpp silently returns
        # plain text and no coordinates, which is not a valid OCR result.
        "special": "--special" in helpText or "-sp," in helpText,
        "gpuLayers": "--gpu-layers" in helpText or "-ngl," in helpText,
        "verbosity": "--verbosity" in helpText,
        "context": "--ctx-size" in helpText or "-c," in helpText,
        "parallel": "--parallel" in helpText or "-np," in helpText,
        "predict": "--predict" in helpText or "-n," in helpText,
        "port": "--port" in helpText,
    }
    if not details["supportsMmproj"]:
        raise CheckError(
            tr('{filename} 缺少 --mmproj 多模态参数。内置运行时与插件不匹配，请完整重新安装配套 PaddleVL 插件。', filename=os.path.basename(path)),
            code=901,
            **details,
        )
    if not details["supportsHost"]:
        raise CheckError(
            tr('{filename} 缺少 --host 服务端参数。内置运行时不完整，请完整重新安装配套 PaddleVL 插件，不使用外部可执行文件回退。', filename=os.path.basename(path)),
            code=901,
            **details,
        )
    missingCore = [
        name for name in (
            "model", "temperature", "special", "gpuLayers", "context", "parallel",
            "predict", "port",
        ) if not details["flags"].get(name)
    ]
    if missingCore:
        raise CheckError(
            tr('{filename} 缺少本插件需要的 llama-server 参数：{items}。请完整重新安装配套 PaddleVL 插件；插件不会猜测或硬传不受支持的参数。', filename=os.path.basename(path), items=', '.join(missingCore)),
            code=902,
            **details,
        )
    return details


# ========================= 【协议：OpenAI 兼容服务】 =========================


def endpoint(argd):
    """地址不可配置；历史 host 即使传入也不参与运行。"""
    port = timeoutSeconds(argd, "service.port", 9931, 1, 65535)
    return "127.0.0.1", port, f"http://127.0.0.1:{port}/v1"


def timeoutSeconds(argd, key="resources.timeoutSeconds", default=120, low=5, high=3600):
    try:
        raw = argd.get(key, default)
        if isinstance(raw, bool) or isinstance(raw, float) and (not raw.is_integer()):
            raise ValueError()
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        raise CheckError(tr("{key} 必须是整数。", key=key), code=901, option=key)
    if not low <= value <= high:
        raise CheckError(tr('{key} 必须在 {low} 到 {high} 之间。', key=key, low=low, high=high), code=901, option=key)
    return value


def _httpJson(url, timeout):
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def probeService(baseUrl, timeout=30, alias="paddlevl"):
    """就绪检测：先健康端点，再 OpenAI 模型列表和固定 alias。

    返回 {"ok", "message", "details"} ，其中 details["model"] 是服务当前加载的模型 id 。
    每一种失败都对应一句明确的下一步，而不是把 URLError 原样抛给用户。
    """
    rootUrl = baseUrl[:-3] if baseUrl.endswith("/v1") else baseUrl
    healthUrl = f"{rootUrl}/health"
    try:
        request = Request(healthUrl, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout):
            pass  # llama.cpp 的健康端点只以 HTTP 状态码为契约，不假定响应体格式。
    except HTTPError as exc:
        if exc.code == 503:
            return fail(tr('llama.cpp 正在加载模型（/health 返回 503）。'), errorType="MODEL_NOT_READY", url=healthUrl, status=503, loading=True)
        return fail(tr('服务健康检查返回 HTTP {code} {reason}。', code=exc.code, reason=exc.reason), errorType="SERVICE_PROTOCOL_ERROR", url=healthUrl, status=exc.code)
    except (TimeoutError, socket.timeout):
        return fail(tr('连接 {healthurl} 超过 {timeout} 秒无响应。', healthurl=healthUrl, timeout=timeout), errorType="REQUEST_TIMEOUT", url=healthUrl)
    except URLError as exc:
        return fail(tr('无法连接 {healthurl} ：{error}。', healthurl=healthUrl, error=getattr(exc, 'reason', exc)), errorType="MODEL_NOT_READY", url=healthUrl)
    except OSError as exc:
        return fail(tr('访问 {healthurl} 失败：{error}。', healthurl=healthUrl, error=exc), errorType="MODEL_NOT_READY", url=healthUrl)

    url = f"{baseUrl}/models"
    try:
        payload = _httpJson(url, timeout)
    except HTTPError as exc:
        if exc.code in (401, 403):
            return fail(
                tr('服务拒绝访问（HTTP {code}）。该服务启用了鉴权，但本插件不支持 API Key 。请改用未启用鉴权的本地服务。', code=exc.code),
                errorType="SERVICE_PROTOCOL_ERROR",
                url=url,
                status=exc.code,
            )
        if exc.code == 404:
            return fail(
                tr('服务在 {url} 返回 404 。地址端口对得上，但它没有提供 OpenAI 兼容接口——通常是把端口指向了别的程序。请确认这是 llama.cpp 的服务端口。', url=url),
                errorType="SERVICE_PROTOCOL_ERROR",
                url=url,
                status=exc.code,
            )
        return fail(
            tr('服务返回 HTTP {code} {reason} 。请查看服务端日志确认它已完成模型加载。', code=exc.code, reason=exc.reason),
            errorType="SERVICE_PROTOCOL_ERROR",
            url=url,
            status=exc.code,
        )
    except (TimeoutError, socket.timeout):
        return fail(
            tr('连接 {url} 超过 {timeout} 秒无响应。若服务正在加载大模型，请等待加载完成，或调高【请求超时】。', url=url, timeout=timeout),
            errorType="REQUEST_TIMEOUT",
            url=url,
        )
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        return fail(
            tr('无法连接 {url} ：{reason}。请确认 llama.cpp 服务已经启动，且地址、端口与模型加载状态正确。', url=url, reason=reason),
            errorType="MODEL_NOT_READY",
            url=url,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        return fail(
            tr('{url} 返回的不是合法 JSON（{error}）。端口上运行的多半不是 llama.cpp 服务，请核对端口号。', url=url, error=exc),
            errorType="SERVICE_PROTOCOL_ERROR",
            url=url,
        )
    except OSError as exc:
        return fail(tr('访问 {url} 失败：{error}。', url=url, error=exc), errorType="MODEL_NOT_READY", url=url)

    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list) or not models:
        return fail(
            tr('{url} 的响应里没有 data 模型列表。这个服务不符合 OpenAI 兼容协议，或者尚未加载任何模型；请检查服务端启动参数是否带了 -m 主模型。', url=url),
            errorType="SERVICE_PROTOCOL_ERROR",
            url=url,
        )
    modelIds = [item.get("id") for item in models if isinstance(item, dict) and isinstance(item.get("id"), str)]
    if not modelIds:
        return fail(
            tr('{url} 的模型列表里没有 id 字段，响应不符合 OpenAI 兼容协议。', url=url),
            errorType="SERVICE_PROTOCOL_ERROR",
            url=url,
        )
    if alias and alias not in modelIds:
        return fail(
            tr('服务已就绪，但没有加载本插件要求的本地视觉语言模型。请在插件管理中检查本插件的模型文件与端口设置。'),
            errorType="MODEL_ALIAS_MISMATCH",
            url=url,
            models=modelIds,
            requiredModelAlias=alias,
        )
    modelId = alias or modelIds[0]
    return ok(
        tr('本地模型服务已就绪。'),
        url=url,
        model=modelId,
        modelCount=len(models),
    )


def isPortBusy(host, port, timeout=0.5):
    probeHost = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    try:
        with socket.create_connection((probeHost, port), timeout=timeout):
            return True
    except OSError:
        return False


# ========================= 【版本：Python 依赖】 =========================


def parseVersion(text):
    parts = []
    for chunk in str(text).strip().split("."):
        digits = ""
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def checkPythonDependencies():
    """检测插件私有闭包是否完整、版本是否够用。

    只在健康检查与页面诊断里调用——它会真正导入后端，代价不小，
    绝不能出现在宿主的启动路径上。
    """
    details = {"pluginDir": PLUGIN_DIR, "python": "%d.%d.%d" % sys.version_info[:3]}
    vendor = os.path.join(PLUGIN_DIR, "vendor")
    details["vendorDir"] = vendor
    details["vendorExists"] = os.path.isdir(vendor)
    for name, minimum in REQUIRED_PACKAGES:
        try:
            module = __import__(name)
        except Exception as exc:
            return fail(
                tr('无法导入 {name} ：{error}。插件的私有依赖没有随归档完整解压，或与当前 Python 运行时不匹配。请重新把插件归档完整解压到 Umi-OCR 本体目录（不要只覆盖部分文件），然后重启。', name=name, error=exc),
                **details,
            )
        version = str(getattr(module, "__version__", "0"))
        details[f"{name}Version"] = version
        if minimum > (0, 0, 0) and parseVersion(version) < minimum:
            expect = ".".join(str(n) for n in minimum)
            return fail(
                tr('{name} 版本过低：{version} < {expect} 。本插件的流水线配置依赖 {expect2} 及以上的键名与接口。请重新解压与本插件版本配套的归档。', name=name, version=version, expect=expect, expect2=expect),
                **details,
            )
    return ok(tr('插件私有依赖完整，版本满足要求。'), **details)


def checkLayoutDevice(device):
    """版面检测设备可用性。CUDA 不可用时【明确失败】，不静默回退 CPU 。"""
    text = str(device or "").strip().lower()
    if not text.startswith("gpu:") or not text[4:].isdigit():
        raise CheckError(
            tr('版面检测设备必须写成 gpu:N（例如 gpu:0），当前为：{device} 。', device=device),
            code=901,
            option="pipeline.cudaDevice",
        )
    index = int(text[4:])
    try:
        import paddle
    except Exception as exc:
        raise CheckError(
            tr('无法导入 paddle ：{error}。请重新完整解压插件归档。', error=exc),
            code=905,
        )
    if not paddle.is_compiled_with_cuda():
        raise CheckError(
            tr('插件闭包里的 PaddlePaddle 不是 CUDA 构建，版面检测无法运行。本插件不提供 CPU 回退；请安装 CUDA 版本的插件归档，或改用不需要版面检测的其它方案。'),
            code=905,
        )
    try:
        count = paddle.device.cuda.device_count()
    except Exception as exc:
        raise CheckError(
            tr('查询 CUDA 设备失败：{error}。请确认已安装 NVIDIA 驱动，且驱动版本满足插件所用 CUDA 运行时的要求。', error=exc),
            code=905,
        )
    if index >= count:
        raise CheckError(
            tr('配置的是 {text} ，但本机只检测到 {count} 个 CUDA 设备。请把【版面检测设备】改为 gpu:0 ~ gpu:{count2} 。', text=text, count=count, count2=max(count - 1, 0)),
            code=905,
            option="pipeline.cudaDevice",
            deviceCount=count,
        )
    paddle.device.set_device(text)
    return {"device": text, "deviceCount": count}

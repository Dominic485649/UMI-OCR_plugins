# ==================================================
# =============== 插件健康检查 ===============
# ==================================================

"""由宿主在【插件管理】里按需调用（契约 §5），不在启动路径上。

【职责边界】本函数拿不到用户配置，因此它只回答一个问题：
**这个插件自己装完整了吗？**

    - 插件目录里该有的文件是否都在（部分覆盖解压是最常见的故障）；
    - 私有依赖闭包能否导入、版本是否够用。

外部模型目录与 llama.cpp 服务是否可用，取决于用户填的配置，只能由
api.Api.health(argd) 与 ocr_api.Api.health(argd) 回答——它们由【全局设置 →
模型服务】和对应能力的健康检查触发。两者都通过，后端才真正可用。

message 必须给出可操作的下一步，不能只报错误码。
"""

import os

from . import checks
from .checks import tr

# 少任何一个，插件都跑不起来。清单本身也在列：它是唯一事实源。
REQUIRED_FILES = (
    "umi-plugin.json",
    "__init__.py",
    "api.py",
    "ocr_api.py",
    "runtime.py",
    "worker.py",
    "checks.py",
    "budgets.py",
    "asset_paths.py",
    "service.py",
    "page.py",
    "StructuredDocument.qml",
)


def _missingFiles():
    return [
        name
        for name in REQUIRED_FILES
        if not os.path.isfile(os.path.join(checks.PLUGIN_DIR, name))
    ]


def check():
    details = {"pluginDir": checks.PLUGIN_DIR, "externalAssets": "unverified"}

    missing = _missingFiles()
    details["missingFiles"] = missing
    if missing:
        return checks.fail(
            (
                tr('插件目录缺少 {missing} 个文件：{items} 。请把插件归档【完整】重新解压到 Umi-OCR 本体目录，不要只覆盖其中一部分文件，然后重启。', missing=len(missing), items=', '.join(missing))
            ),
            errorType="PIPELINE_DEPENDENCY_MISSING",
            **details,
        )

    try:
        manifest = checks.loadManifest()
        details["version"] = manifest.get("version", "")
    except Exception as exc:
        return checks.fail(
            (
                tr('插件清单 umi-plugin.json 无法解析：{error}。文件可能在解压或编辑中损坏，请重新解压插件归档。', error=exc)
            ),
            errorType="INVALID_CONFIGURATION",
            **details,
        )

    try:
        details["bundledRuntime"] = checks.bundledRuntime()
    except checks.CheckError as exc:
        return checks.fail(exc.message, **details, **exc.details)

    vendor = os.path.join(checks.PLUGIN_DIR, "vendor")
    details.update({
        "vendorDir": vendor,
        "vendorExists": os.path.isdir(vendor),
        "architecture": "separate-worker-stdio-ipc",
    })
    if not os.path.isdir(vendor):
        return checks.fail(
            tr('插件私有 vendor 闭包缺失，请重新完整解压 PaddleOCR-VL 插件归档。'),
            errorType="PIPELINE_DEPENDENCY_MISSING",
            **details,
        )

    return {
        "ok": True,
        "message": (
            checks.tr("插件内置运行时与私有依赖目录完整；模型由用户导入。CUDA 目标为 Turing sm75 及以后，实际 GPU/驱动支持须在能力健康检查与模型启动时验证，不能仅凭显卡名称判断。")
        ),
        "details": details,
    }

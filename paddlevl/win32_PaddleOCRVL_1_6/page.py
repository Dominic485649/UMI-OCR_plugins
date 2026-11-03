"""Controller for the PaddleOCR-VL document page.

The page owns only its MissionDocument queue and presentation/export state.
It never owns the llama-server process: both document parsing and screenshot
OCR use the plugin-level RuntimeService configured from global settings.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from . import checks
from .checks import tr
from .runtime import tempRoot


class StructuredDocumentPage:
    def __init__(self, ctrlKey, connector):
        self.ctrlKey = ctrlKey
        self.connector = connector
        self._missionId = ""
        self._callbackGeneration = 0
        self._temporaryInputs = []

    # ========================= 【qml → python】 =========================

    def initPlugin(self):
        try:
            from py_src.document.image_input import prepared_document_input
            manifest = checks.loadManifest()
        except ImportError:
            return checks.fail(tr('此版 PaddleVL 需要 Umi-OCR 3.1.2 或更新版本，请同时更新本体。'))
        except Exception as exc:
            return checks.fail(tr('读取插件清单失败：{error}。插件目录可能不完整，请重新解压归档。', error=exc))
        return checks.ok(
            tr('文档解析页面已就绪。模型服务在【全局设置 → 插件管理】统一管理。'),
            pluginId=manifest.get("id", checks.PLUGIN_ID),
            pluginName=manifest.get("name", ""),
            version=manifest.get("version", ""),
            documentNameFilter=self._documentNameFilter(),
        )

    def parseDocuments(self, paths, values=None):
        paths = [str(path) for path in (paths or []) if str(path).strip()]
        if not paths:
            return tr('[Error] 没有选择任何文档。')
        mission = self._missionDocument()
        if mission is None:
            return tr('[Error] 无法访问宿主的结构化文档任务管理器。请确认宿主 API 2.9 已完整安装。')
        if self._missionId:
            return tr('[Error] 当前页面已有解析任务；请先取消或等待其完成。')
        from uuid import uuid4
        self._callbackGeneration += 1
        generation = self._callbackGeneration
        self._missionId = mission_id = str(uuid4())
        msnInfo = {
            "onStart": lambda _info: self._sendMission(generation, "onMissionStart", mission_id),
            "onGet": lambda _info, msn, result: self._sendMission(
                generation, "onMissionGet", mission_id, str(msn.get("path", "")), result),
            "onEnd": lambda _info, message: self._onEnd(generation, mission_id, message),
            "onProgress": lambda stage: self._sendMission(generation, "onMissionProgress", mission_id, str(stage)),
            "onPartial": lambda event: self._sendMission(generation, "onMissionPartial", mission_id, event),
            "argd": dict(values or {}),
            "_reserved_msnID": mission_id,
        }
        missionId = mission.addMissionList(msnInfo, [{"path": path} for path in paths])
        if isinstance(missionId, str) and missionId.startswith("[Error]"):
            self._missionId = ""
            return missionId
        return str(missionId)

    def pasteImage(self, imgID, values=None):
        """Materialize a clipboard image for MissionDocument without a second OCR path."""
        try:
            from py_src.image_controller.image_provider import PixmapProvider

            pixmap = PixmapProvider.getPixmap(str(imgID or ""))
            if not pixmap:
                return tr('[Error] 剪贴板图像已失效，请重新粘贴。')
            data = PixmapProvider.toBytes(pixmap)
            if not data:
                return tr('[Error] 无法读取剪贴板图像字节。')
            fd, raw_path = tempfile.mkstemp(prefix="umi-paddlevl-paste-", suffix=".png", dir=tempRoot())
            os.close(fd)
            path = Path(raw_path)
            path.write_bytes(data)
            self._temporaryInputs.append(path)
            mission_id = self.parseDocuments([str(path)], values)
            if isinstance(mission_id, str) and mission_id.startswith("[Error]"):
                return mission_id
            return checks.ok(
                tr('已准备剪贴板图像。'),
                missionId=str(mission_id),
                path=str(path),
            )
        except Exception as exc:
            return tr('[Error] 准备剪贴板图像失败：{error}', error=exc)

    def cancel(self):
        if not self._missionId:
            return checks.ok(tr('当前没有运行中的文档解析任务。'))
        mission = self._missionDocument()
        if mission is None:
            return checks.fail(tr('无法访问宿主的结构化文档任务管理器。'))
        mission.stopMissionList(self._missionId)
        return checks.ok(tr('已请求取消当前文档解析任务；共享模型服务会继续遵循全局生命周期策略。'))

    def exportResult(self, result, outputDir, formatName="all"):
        """原子导出 Markdown／JSON；显式 TXT 兼容请求仅改变扩展名。"""
        if not isinstance(result, dict) or result.get("code") != 100:
            return checks.fail(tr('没有可导出的成功文档解析结果。请先完成一次解析。'))
        raw = str(outputDir or "").strip()
        if not raw:
            return checks.fail(tr('请先选择导出目录。'))
        destination = Path(os.path.abspath(os.path.expanduser(raw)))
        if not destination.is_dir():
            return checks.fail(tr('导出目录不存在：{destination}。请重新选择一个已存在且可写的目录。', destination=destination))
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        stem = self._safeStem(metadata.get("sourceStem") or Path(str(metadata.get("source") or "structured_document")).stem)
        markdown = str(result.get("markdown", data.get("markdown", "")) or "")
        requested = str(formatName or "all").lower()
        if requested not in ("all", "txt", "md", "json"):
            return checks.fail(tr('导出格式只能是 txt、md、json 或 all。'))
        exported = {}
        try:
            assets, markdown, resourceMap = self._copyAssets(
                data.get("assets") or [], destination, stem, markdown
            )
            if requested == "txt":
                path = destination / f"{stem}.md.txt"
                self._atomicText(path, markdown)
                exported["txt"] = str(path)
            if requested in ("all", "md"):
                path = destination / f"{stem}.md"
                self._atomicText(path, markdown)
                exported["markdown"] = str(path)
            if requested in ("all", "json"):
                path = destination / f"{stem}.json"
                # ``data.pages`` carries Markdown and resource references too.
                # Keep the exported JSON self-contained: no page may retain a
                # file:// URI into the plugin's temporary directory after the
                # corresponding image has been copied next to the .md file.
                serializable = self._replaceResources(result, resourceMap)
                serializable["markdown"] = markdown
                serializable["data"] = dict(serializable.get("data") or {})
                serializable["data"]["markdown"] = markdown
                serializable["data"]["assets"] = assets
                self._atomicText(path, json.dumps(serializable, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
                exported["json"] = str(path)
        except (OSError, TypeError, ValueError) as exc:
            return checks.fail(tr('导出失败：{error}。请确认目录可写、磁盘空间充足，且结果没有被其它程序占用。', error=exc))
        return checks.ok(tr('已导出 {items}。', items='、'.join((Path(path).name for path in exported.values()))), **exported, assets=assets)

    def close(self):
        # The page must never stop the shared service.  Cancelling a page-owned
        # MissionDocument task is safe and avoids callbacks targeting a deleted
        # QML object.
        self._callbackGeneration += 1
        self.cancel()
        self.connector = None
        for path in self._temporaryInputs:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._temporaryInputs.clear()

    # ========================= 【任务回调】 =========================

    def _onEnd(self, generation, mission_id, message):
        if generation != self._callbackGeneration:
            return
        self._missionId = ""
        self._sendMission(generation, "onMissionEnd", mission_id, str(message))

    def _sendMission(self, generation, funcName, *args):
        if not self.connector or generation != self._callbackGeneration:
            return
        # 先冻结结果，再在 Qt 实际交付时复核，避免旧任务回调覆盖新任务。
        try:
            self.connector.callFunc(self._deliverMission, generation, funcName, *deepcopy(args))
        except Exception:
            pass

    def _deliverMission(self, generation, funcName, *args):
        if self.connector and generation == self._callbackGeneration:
            self.connector.callQml(self.ctrlKey, funcName, *args)

    # ========================= 【内部】 =========================

    @staticmethod
    def _missionDocument():
        try:
            from py_src.mission.mission_document import MissionDocument

            return MissionDocument
        except Exception:
            return None

    @staticmethod
    def _documentNameFilter():
        from py_src.utils.image_formats import document_name_filter
        return document_name_filter()

    @staticmethod
    def _safeStem(value):
        value = "".join(char if char.isalnum() or char in "-_ ." else "_" for char in str(value)).strip(" .")
        return value or "structured_document"

    @staticmethod
    def _atomicText(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @staticmethod
    def _copyAssets(records, destination, stem, markdown):
        if not records:
            return [], markdown, {}
        assetsDir = destination / f"{stem}_assets"
        copied = []
        references = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            source = Path(str(record.get("path") or ""))
            if not source.is_file():
                raise FileNotFoundError(
                    tr('文档资源已不存在，无法导出：{path}。请重新解析该文档后再导出。', path=source)
                )
            name = Path(str(record.get("name") or source.name)).name or source.name
            for node in (assetsDir, *assetsDir.parents):
                if node.is_symlink() or node.is_junction():
                    raise ValueError("ASSET_OUTPUT_INVALID: linked export directory")
            assetsDir.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as stream:
                sourceHash = hashlib.file_digest(stream, "sha256").digest()
            target = assetsDir / name
            number = 2
            while target.exists():
                if target.is_symlink() or target.is_junction():
                    raise ValueError("ASSET_OUTPUT_INVALID: linked export file")
                # 再次导出同一结果复用相同字节，MD 与兼容 TXT 的相对资源引用保持一致。
                if target.is_file() and target.stat().st_size == source.stat().st_size:
                    with target.open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").digest() == sourceHash:
                            break
                target = assetsDir / f"{Path(name).stem}_{number}{Path(name).suffix}"
                number += 1
            if not target.exists():
                handle = tempfile.NamedTemporaryFile(dir=assetsDir, prefix=".asset-", suffix=".tmp", delete=False)
                temporary = Path(handle.name)
                handle.close()
                try:
                    shutil.copy2(source, temporary)
                    os.replace(temporary, target)
                except Exception:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise
            relative = f"{assetsDir.name}/{target.name}"
            for reference in (record.get("uri"), source.resolve().as_uri(), record.get("source"), *record.get("sourceReferences", [])):
                if reference:
                    normalized = str(reference).replace("\\", "/")
                    markdown = markdown.replace(normalized, relative)
                    references[normalized] = relative
            references[str(source).replace("\\", "/")] = relative
            # The export is portable: do not retain a machine-specific
            # absolute path in its JSON.  Markdown and page resources both
            # use this same relative reference beside the exported .md/.json.
            copied.append({
                "id": record.get("id"), "name": target.name,
                "path": relative, "relativePath": relative, "uri": relative,
                "source": record.get("source"),
            })
        return copied, markdown, references

    @staticmethod
    def _replaceResources(value, references, key=None):
        """Return a JSON-safe copy with temporary asset references made relative."""
        # ``source`` is provenance, not a live resource locator.  Preserve it
        # verbatim so downstream software can still correlate an exported
        # asset with the name reported by Paddle.
        if key == "source":
            return value
        if isinstance(value, str):
            result = value
            # Longest first prevents a short source path from partially
            # replacing a longer file URI that contains the same name.
            for source, relative in sorted(references.items(), key=lambda item: len(item[0]), reverse=True):
                result = result.replace(source, relative)
            return result
        if isinstance(value, list):
            return [StructuredDocumentPage._replaceResources(item, references) for item in value]
        if isinstance(value, dict):
            return {
                childKey: StructuredDocumentPage._replaceResources(item, references, childKey)
                for childKey, item in value.items()
            }
        return value

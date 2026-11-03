# PaddleVL 插件 3.1.2

这是 Umi-OCR 的独立文档与图片识别插件，对应 Umi-OCR 3.1.2／Host API 2.9，兼容声明为 `>=2.9,<3.0`。插件、本体和 CUDA 各自维护版本，不要求同版。

- [主程序 3.1.2](https://github.com/Dominic485649/UMI-OCR/releases/tag/v3.1.2)
- [PaddleVL 3.1.2](https://github.com/Dominic485649/UMI-OCR_plugins/releases/tag/v3.1.2)
- [CUDA 3.0.5](https://github.com/Dominic485649/UMI-OCR_plugins/releases/tag/v3.0.5)

实际发布状态、文件与摘要以对应 Release 为准。先完整解压主程序，再将 PaddleVL 归档解压到同一根目录，不覆盖正在运行的程序。超过附件上限时，下载全部分卷、`archive-chunks.json` 和 `Restore-ReleaseArchive.ps1`，使用本体附带的 Python／恢复工具还原完整 `.tar.xz`；恢复会校验每块及完整归档，拒绝覆盖已有文件。

## 能力与配置

PaddleOCR-VL 1.6 Spotting 用于图片 OCR，输出文本与多边形；模型不提供识别分数时保留未知置信度，不伪造百分比。结构化文档解析默认通过 `/api/document/*` 提供，不展示文档任务页。HTTP 的启用、监听与鉴权仍由用户配置，不会自动打开。

两个能力共享一个插件级本地 `llama-server` 和用户选择的 GGUF／mmproj。服务默认端口为 `9931`，已有明确配置的其他端口不强制迁移。端口占用时不接管外部程序。模型、上下文、GPU 层数及启动／空闲策略在共享模型设置中维护。

插件附带锁定的 llama.cpp 及 Python 运行依赖，但不含 GGUF、mmproj 或 PP-DocLayoutV3 权重；由用户自行提供并确认适用许可。不会下载、修改或量化原模型。Windows x64 是本次目标，GPU／驱动仍需通过实际检查，不以目标架构列表保证所有设备兼容。

## 本次修复及验证边界

修复文档坐标映射、Markdown／公式／代码预览、跨页图片重复、任务结束后的安全模型切换和图片格式对齐；补齐跨页表格所需的 Beautiful Soup／SoupSieve 依赖闭包。界面输出统一 Markdown，HTTP TXT 兼容为同内容 `.md.txt`。

已执行的真实 PDF HTTP／GPU 定向检查覆盖选页、整篇、页面标识、坐标边界及 MD/TXT 内容一致性；具体源与载荷绑定由本次正式交付记录说明。截图捕获、HDR、混屏／多屏继续 NOT_RUN；没有用户原始文档，不声称所有文档、模型、硬件或历史完整矩阵都已重测。

## 对应源码与许可

适配器、QML、构建脚本、测试、依赖锁和补丁完整保留在本目录。固定原生运行时的完整适用源码、构建输入／结果、来源及许可保留于安装后的 `UmiOCR-data/plugins/win32_PaddleOCRVL_1_6/resources/llama.cpp/`。公开对应源码位于插件仓库的 `v3.1.2` 标签，适配器在 `paddlevl/`，原生来源材料在 `paddlevl-native/`。

适配器按 AGPL-3.0-or-later 提供，见 `LICENSE` 与 `COPYING`；运行依赖遵守各自许可。收费交付或支持不限制接收者复制、修改或再分发覆盖源码的权利。未新增模型或商标授权主张，亦不声称第三方专有运行库属于 AGPL。

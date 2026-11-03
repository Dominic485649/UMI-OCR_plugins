# Release Notes

## 3.1.2（本地候选，未发布）

- 需要 Umi-OCR 3.1.2／Host API 2.9；复用宿主统一图片解码和受限离线富 Markdown 预览。
- 修复 PDF 推理／预览分辨率及图片 DPI 差异导致的版面框偏移；以结果页尺寸映射到实际预览像素，不重复乘视图缩放。
- 原始块内容仅在最终输出时格式化，修复跨页重组图片块被重复生成；保留合法跨页图片、原始块顺序及官方重组语义，补全重组资源别名映射。
- 补齐官方跨页表格合并所需的 Beautiful Soup／SoupSieve 纯 Python 依赖和原始许可；增加实际官方表格路径的构建门禁，避免多页文档因延迟导入缺失而仅部分完成。
- 文档输入与图片 OCR 共用 JPEG、PNG、WebP、AVIF、JXL、JPEG XR、BMP、TIFF 等解码能力，保留原文件身份、EXIF 方向、透明白底、TIFF 逐页及预算限制。
- 默认显示预览，保留 Markdown 源码、JSON 和结构块；默认导出 Markdown／JSON 和图片，TXT 仅作同内容 `.md.txt` 兼容适配。
- Spotting 明确声明不提供识别置信度；部分完成显示具体原因，不伪装整篇成功。
- GGUF、mmproj、版面模型和量化精度不变，不下载或修改权重；本次不重建 CUDA 插件，也不包含 GitHub 发布授权。

## 3.0.0（仅本地，未发布）

- PaddleVL 独立升级至 3.0.0，适配 Umi-OCR 3.x 对已知旧插件的迁移检查，保留 Host API 2.8 契约。
- 默认 llama.cpp 服务端口统一为 9931，覆盖设置、端点缺省回退及真实验收脚本；保留用户显式指定的其他端口。
- 本次构建和安装仅供本地使用，不推送、创建标签或 Release，后续合并到下一次发布。
- 模型、llama-server、GGUF/mmproj 与 PP-DocLayoutV3 仍由用户自备，不随插件分发。

## 2.8.1 (unreleased)

- Added the PaddleOCR-VL 1.6 **文字识别** capability for the existing OCR
  workflow, with official Spotting coordinates/polygons and nullable VLM
  confidence.
- Kept **文档解析** as an independent full Pipeline capability; it preserves
  layout blocks, official IDs/order, Markdown, JSON, table/formula metadata and
  exported image resources.
- Added one shared, globally managed local llama.cpp service for both modes.
  Its CLI capabilities are checked at runtime; paths and resource settings are
  supplied by the user and no model asset is modified.
- This note documents source changes only. It is not a packaged or published
  release, and real PP-DocLayoutV3/GPU acceptance remains environment-specific.

## 1.0.0

- Migrated the structured document adapter, its QML page, build scripts,
  dependency locks, and mock regression out of the Umi-OCR base repository.
- Established this independent AGPL-3.0-or-later source-delivery project.
- No model weights, runtime servers, GGUF files, mmproj files, or layout-model
  assets are included in this release.

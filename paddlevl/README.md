# PaddleOCR-VL 1.6 插件

当前源码版本 **3.1.3**，独立于主程序和 CUDA 插件，要求 Host API `>=2.9,<3.0`。本轮为本地候选，不据此宣称已公开发布。[主项目](https://github.com/Dominic485649/UMI-OCR) · [公开插件](https://github.com/Dominic485649/UMI-OCR_plugins)

## 能力与模型

- 图片 OCR 使用官方 v1.6 Spotting，返回文字与四点多边形；模型不提供置信度时保持未知，不伪造百分比。
- 完整文档解析使用 PP-DocLayoutV3 与官方 v1.6 pipeline，提供 Markdown、结构化 JSON、阅读顺序和资源；默认通过 HTTP 使用，文档 UI 默认不展示。
- 两项能力共用插件内置 llama.cpp 服务，默认端口 9931，仅监听回环，不接管外部服务；已显式配置的其它端口仍有效。
- 主 GGUF、视觉投影 GGUF 和版面模型可手动导入，或在模型路径行点击“下载”获取固定 revision 的官方文件。下载后放到**用户实际安装的本插件根目录 `model/`**；不在启动时自动下载，不改原始权重、不另选量化档、不搬迁外部模型。
- 下载显示进度、支持取消，完整校验后回填；同名冲突不覆盖，权限失败不静默换目录。模型下载由支持该功能的主程序完成，不触发推理服务预热。

## 安装与分发

先完整解压主程序，再将 PaddleVL `.tar.xz` 解压到同一软件根目录，插件位于 `UmiOCR-data/plugins/win32_PaddleOCRVL_1_6/`。更新前正常退出，保留用户配置、任务资源与 `model/`。

最简运行归档包含插件、内置运行时、私有依赖和必要许可/源码指引，不附带完整源码与审计树。对应源码独立提供，见 [SOURCE-CODE.md](SOURCE-CODE.md)。本轮不捆绑 GGUF、mmproj 或版面权重；用户可按需从官方获取，模型按各自许可使用。

## 构建与验证

`Build-Plugin.ps1` 使用显式 `VendorRoot`、`RuntimeLock` 和 `LlamaRuntimeBundle`，复用已核对的依赖/服务闭包；不会静默从宿主安装依赖。每轮使用新的 `StagingRoot`。

默认 `DistributionLayout=minimal`，完整输入保存在 staging；最简运行目录默认为 `<StagingRoot>-runtime`，分包回执位于 `build/audit/`。`SkipArchive` 只准备，不授予公开发布权限。构建按宿主 Python ABI（当前 3.13.16），不单独将 worker 升到缺少 Paddle GPU wheel 的 ABI。

针对本轮改动验证模型声明、宿主下载/配置、分包闭包、真实重装和代表性运行；截图/HDR/混屏继续 NOT_RUN，不将旧记录或元数据探测冒充新权重完整下载。

## 许可

插件适配器采用 AGPL-3.0-or-later，保留 [LICENSE](LICENSE)、[COPYING](COPYING) 与第三方声明。收费交付或支持不限制接收者复制、修改和再分发覆盖内容的权利。运行库、NVIDIA 与模型分别适用其条款，不新增模型/商标授权主张。

# PaddleVL 3.1.2 对应源码交付

本组件运行代码及依赖复用已审 3.1.2 构建，不重新编译或更换模型。原始构建使用 `Build-Plugin.ps1 -LocalCandidate`；本次在全新交付目录补齐公开来源、准确文案及实际源码／运行载荷绑定，再单独核验正式交付条件。原始本地构建记录、归档及失败证据不改写为新的通过记录。

## 包内材料

- 本目录的 `source-records/` 完整保留适配器、QML、构建脚本、测试、依赖锁、PaddleX 补丁与许可说明。该目录不是用户状态目录。
- llama.cpp 固定提交为 `74bca3950054eed1795f525cb9cf06bfe796492d`。完整适用源码、构建脚本、输入／结果和来源材料位于安装后的 `UmiOCR-data/plugins/win32_PaddleOCRVL_1_6/resources/llama.cpp/source/`，相邻 `LICENSES/` 保留许可和专有运行库条款。
- 原生源码 TAR SHA256 为 `c71b9c01e263232c854dea1c9992d6e1a42b9e96d4a687e05f1f0c0fc13dae12`。公开 Git 仅对该原 TAR 做无损 xz 表示；解压后的总长度和 SHA256 必须与包内 TAR 相同，不是另一个来源快照。
- 分发包不含 GGUF、mmproj 或版面模型权重，未获得或宣称额外模型／商标许可。

## 公开来源

- 仓库：https://github.com/Dominic485649/UMI-OCR_plugins
- 标签：`v3.1.2`
- 适配器对应源码：`paddlevl/`
- 固定原生运行时来源和许可：`paddlevl-native/`
- 具体不可变 commit、匿名标签校验结果、逐文件清单及实际交付验证由本次 `PUBLIC_DELIVERY.json` 记录；此说明本身不替代来源门禁通过。

本包中的 Python worker、原生服务及各自依赖按其声明的进程边界通信。主程序与 CUDA 插件保持各自独立的版本、源码、许可和正式验收。开源许可及第三方专有条款分别适用，不以版本一致、源码链接或用户发布同意替代许可事实。

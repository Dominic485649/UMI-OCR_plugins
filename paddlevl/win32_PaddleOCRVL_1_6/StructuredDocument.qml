import QtQuick 2.15
import QtQuick.Controls 2.15
import DocPreviewConnector 1.0

import "../../qt_res/qml/TabPages"
import "../../qt_res/qml/Widgets"
import "../../qt_res/qml/Widgets/ImageViewer"
import "../../qt_res/qml/Configs"
import "../../qt_res/qml/Widgets/Sequence.js" as Sequence

TabPage {
    id: page
    property var inputPaths: []
    property string currentPath: ""
    property int currentPage: 1
    property int pageCount: 0
    property bool previewLoading: false
    property string previewError: ""
    property string documentNameFilter: ""
    property var documentInfo: ({})
    property var documentResults: ({})
    property var documentStates: ({})
    property var pageFailures: ({})
    property var runPaths: []
    property var finishedPaths: []
    property bool parsing: false
    property bool cancelRequested: false
    property string activeMissionId: ""
    property var structuredResult: documentResults[currentPath] || ({})
    property var currentResultPage: findResultPage(currentPage)
    property var currentBlocks: currentResultPage ? currentResultPage.blocks || [] : []
    property var resultBlocks: visibleBlocks()
    property string selectedBlockKey: ""
    property bool wholeDocument: false
    property alias mainTab: mainTabs.currentIndex
    property alias resultTab: resultTabs.currentIndex
    property string statusText: qsTr("从左侧打开、粘贴或拖入图片/PDF；模型服务由全局设置统一管理。")
    property real pageProgress: 0
    configsComp: configs

    function mergedValues() {
        return Object.assign({}, qmlapp.globalConfigs.getValueDict(), configs.getValueDict())
    }

    function inputPath(path) {
        const normalized = String(path || "").replace(/\\/g, "/").toLowerCase()
        return inputPaths.find(item => item.replace(/\\/g, "/").toLowerCase() === normalized) || ""
    }

    function pathName(path) { return String(path).replace(/\\/g, "/").split("/").pop() }

    function addPaths(paths, pasted=false) {
        if((parsing && !pasted) || !paths || paths.length === 0) return
        inputPaths = Array.from(new Set(paths.map(path => String(path)).filter(path => path.length)))
        if(!inputPaths.length) return
        documentResults = ({})
        documentStates = ({})
        documentInfo = ({})
        pageFailures = ({})
        if(!pasted) { runPaths = []; finishedPaths = []; pageProgress = 0 }
        selectDocument(inputPaths[0])
        if(!pasted) statusText = qsTr("已选择 %1 个文件。").arg(inputPaths.length)
    }

    function selectDocument(path) {
        if(inputPaths.indexOf(path) < 0) return
        currentPath = path
        currentPage = 1
        pageCount = Number((documentInfo[path] || {}).pageCount || 0)
        selectedBlockKey = ""
        resetResultScroll()
        refreshPreview()
    }

    function openFiles() { if(!parsing) openDialog.open() }

    function prepareRun(paths) {
        runPaths = paths.slice()
        finishedPaths = []
        pageProgress = 0
        activeMissionId = ""
        cancelRequested = false
        const results = Object.assign({}, documentResults)
        const states = Object.assign({}, documentStates)
        const failures = Object.assign({}, pageFailures)
        for(const path of paths) { delete results[path]; states[path] = "pending"; delete failures[path] }
        documentResults = results
        documentStates = states
        pageFailures = failures
        parsing = true
        mainTab = 1
    }

    function paste() {
        if(parsing) return
        const result = qmlapp.imageManager.getPaste()
        if(result.error) { qmlapp.popup.message(qsTr("粘贴失败"), result.error, "error"); return }
        if(result.paths && result.paths.length > 0) { addPaths(result.paths); return }
        if(!result.imgID) { qmlapp.popup.simple(qsTr("剪贴板中没有图片或文档路径"), ""); return }
        prepareRun([])
        const response = callPy("pasteImage", result.imgID, mergedValues())
        if(!response || !response.ok || !response.details || !response.details.path) {
            parsing = false
            qmlapp.popup.message(qsTr("粘贴失败"), typeof response === "string" ? response : response && response.message || qsTr("无法准备剪贴板图像"), "error")
            return
        }
        addPaths([response.details.path], true)
        runPaths = inputPaths.slice()
        if(parsing) activeMissionId = response.details.missionId
        statusText = qsTr("正在解析剪贴板图像…")
    }

    function startParse() {
        if(parsing) return
        if(!inputPaths.length) { openFiles(); return }
        const previous = {results: documentResults, states: documentStates, failures: pageFailures,
            paths: runPaths, finished: finishedPaths, progress: pageProgress}
        prepareRun(inputPaths)
        statusText = qsTr("正在通过完整文档解析 Pipeline 处理 %1 个文件…").arg(inputPaths.length)
        const response = callPy("parseDocuments", inputPaths, mergedValues())
        if(typeof response === "string" && response.length && !response.startsWith("[Error]")) {
            if(parsing) activeMissionId = response
            return
        }
        parsing = false
        documentResults = previous.results
        documentStates = previous.states
        pageFailures = previous.failures
        runPaths = previous.paths
        finishedPaths = previous.finished
        pageProgress = previous.progress
        statusText = String(response || qsTr("文档解析失败。"))
        qmlapp.popup.message(qsTr("无法开始文档解析"), statusText, "error")
    }

    function cancelParse() {
        if(!parsing || cancelRequested) return
        cancelRequested = true
        const response = callPy("cancel")
        statusText = response && response.message ? response.message : qsTr("已请求取消当前任务。")
        if(response && response.ok === false) cancelRequested = false
    }

    function refreshPreview() {
        if(!currentPath) return
        previewLoading = true
        previewError = ""
        preview.clear()
        previewConnector.preview(currentPath, currentPage, "")
    }

    function setCurrentPage(number) {
        const target = Math.max(1, Math.min(Math.max(1, pageCount), Math.floor(Number(number) || 1)))
        if(currentPage === target) return
        currentPage = target
        selectedBlockKey = ""
        resetResultScroll()
        refreshPreview()
    }

    function findResultPage(number) {
        const pages = structuredResult.data && structuredResult.data.pages || []
        return pages.find(item => Number(item.page) === Number(number)) || null
    }

    function visibleBlocks() {
        if(!wholeDocument) return currentBlocks.map(block => Object.assign({}, block, {page: currentPage}))
        const blocks = []
        for(const item of structuredResult.data && structuredResult.data.pages || [])
            for(const block of item.blocks || []) blocks.push(Object.assign({}, block, {page: Number(item.page)}))
        return blocks
    }

    function blockKey(block) {
        if(!block) return ""
        return `${block.page || currentPage}:${block.globalId !== null && block.globalId !== undefined ? block.globalId : block.id}`
    }

    function selectBlock(block) {
        if(block.page && Number(block.page) !== currentPage) setCurrentPage(block.page)
        selectedBlockKey = blockKey(block)
        for(let index = 0; index < resultBlocks.length; index++) {
            if(blockKey(resultBlocks[index]) === selectedBlockKey) {
                blockList.currentIndex = index
                blockList.positionViewAtIndex(index, ListView.Contain)
                break
            }
        }
    }

    function resultText(index=resultTab) {
        const result = wholeDocument ? structuredResult : currentResultPage
        if(!result) return ""
        if(index === 0 || index === 1) return result.markdown || (result.data || {}).markdown || ""
        return Object.keys(result).length ? JSON.stringify(result, null, 2) : ""
    }

    function resultStatus() {
        if(!currentPath) return qsTr("尚未选择文档")
        const failure = (pageFailures[currentPath] || {})[currentPage]
        if(!wholeDocument && failure) return qsTr("此页解析失败：%1").arg(failure)
        const state = documentStates[currentPath] || ""
        if(!wholeDocument && !currentResultPage)
            return parsing && (state === "pending" || state === "running") ? qsTr("此页尚未返回解析结果，可继续预览原始页面。") : qsTr("此页没有解析结果。")
        if(state === "complete" || state === "empty") return qsTr("解析完成")
        if(parsing) return qsTr("解析中，已完成页可查看；整篇结果尚未最终合并。")
        if(structuredResult.partial) {
            const details = Sequence.warnings(structuredResult.warnings).join("\n")
            return qsTr("任务未完整结束，已保留完成页的结果。") + (details ? "\n" + details : "")
        }
        return qsTr("此页没有解析结果。")
    }

    function resetResultScroll() {
        Qt.callLater(() => {
            textView.resetScroll()
            for(const view of [markdownView, jsonView]) {
                if(view.contentItem) { view.contentItem.contentX = 0; view.contentItem.contentY = 0 }
            }
            blockList.positionViewAtBeginning()
        })
    }
    onWholeDocumentChanged: resetResultScroll()
    onResultTabChanged: resetResultScroll()

    function copyResult() {
        const text = resultText()
        if(!text) { qmlapp.popup.simple(qsTr("没有可复制的结果"), ""); return }
        qmlapp.utilsConnector.copyText(text)
        qmlapp.popup.simple(qsTr("已复制结果"), qsTr("共 %1 个字符").arg(text.length))
    }

    function exportAll() {
        if(!structuredResult || structuredResult.code !== 100) { qmlapp.popup.simple(qsTr("没有可导出的结果"), ""); return }
        exportDialog.open()
    }

    function resetPage() {
        if(parsing) return
        previewConnector.clear()
        preview.clear()
        inputPaths = []
        currentPath = ""
        currentPage = 1
        pageCount = 0
        documentInfo = ({})
        documentResults = ({})
        documentStates = ({})
        pageFailures = ({})
        runPaths = []
        finishedPaths = []
        pageProgress = 0
        activeMissionId = ""
        selectedBlockKey = ""
        previewLoading = false
        previewError = ""
        statusText = qsTr("已清空输入和结果，解析设置保持不变。")
    }

    function closePage() {
        if(parsing) {
            qmlapp.popup.dialog("", qsTr("任务仍在运行。关闭此页会取消当前解析，但不会停止共享模型服务。是否继续？"),
                flag => { if(flag) delPage() }, "warning", {yesText: qsTr("取消任务并关闭")})
            return
        }
        delPage()
    }

    // ========================= 【Python → QML】 =========================

    function acceptMission(missionId) {
        if(!parsing || !missionId || activeMissionId && activeMissionId !== missionId) return false
        if(!activeMissionId) activeMissionId = missionId
        return true
    }
    function onMissionProgress(missionId, stage) {
        if(acceptMission(missionId)) statusText = stage + "…"
    }
    function onMissionStart(missionId) { acceptMission(missionId) }

    function onMissionPartial(missionId, event) {
        if(!acceptMission(missionId) || !event) return
        const path = inputPath(event.path)
        const number = Number(event.page)
        if(!path || !Number.isInteger(number) || number < 1) return
        const states = Object.assign({}, documentStates)
        states[path] = "running"
        documentStates = states
        if(event.failed) {
            const failures = Object.assign({}, pageFailures)
            failures[path] = Object.assign({}, failures[path] || {})
            failures[path][number] = (event.warnings || []).join("\n") || qsTr("文档解析失败。")
            pageFailures = failures
        }
        const result = event.result
        // Python QVariantList 是 QML 序列，不一定满足 Array.isArray。
        const incomingPages = result && (result.code === 100 || result.code === 101) && result.data && result.data.pages
            ? Array.from(result.data.pages).filter(item => item && typeof item === "object" && Number.isInteger(Number(item.page)) && Number(item.page) > 0) : []
        if(incomingPages.length) {
            const previous = documentResults[path] || ({})
            const oldData = previous.data || ({})
            const pages = Array.from(oldData.pages || [])
            for(const item of incomingPages) {
                const index = pages.findIndex(old => Number(old.page) === Number(item.page))
                if(index < 0) pages.push(item)
                else pages[index] = item
            }
            pages.sort((a, b) => Number(a.page) - Number(b.page))
            const results = Object.assign({}, documentResults)
            const assets = Array.from(oldData.assets || [])
            for(const asset of result.data.assets || [])
                if(!assets.some(old => old.path === asset.path && old.id === asset.id)) assets.push(asset)
            const hasContent = pages.some(item => (item.blocks || []).length || item.text || item.markdown)
            results[path] = {code: hasContent ? 100 : 101, partial: true,
                text: pages.map(item => item.text || "").join("\n\n"),
                markdown: pages.map(item => item.markdown || "").join("\n\n"),
                warnings: Array.from(previous.warnings || []).concat(Array.from(event.warnings || [])),
                data: {schemaVersion: 1, pages: pages, assets: assets,
                    blocks: pages.reduce((all, item) => all.concat((item.blocks || []).map(block => Object.assign({}, block, {page: item.page}))), [])},
                metadata: {source: path, pageCount: pages.length, incomplete: true}}
            documentResults = results
        }
        pageProgress = Number(event.totalPages) > 0 ? Math.min(1, Math.max(0, Number(event.completedPages || 0) / Number(event.totalPages))) : 0
    }

    function onMissionGet(missionId, rawPath, result) {
        if(!acceptMission(missionId)) return
        const path = inputPath(rawPath)
        if(!path) return
        if(finishedPaths.indexOf(path) < 0) finishedPaths = finishedPaths.concat([path])
        pageProgress = 0
        const states = Object.assign({}, documentStates)
        if(result && (result.code === 100 || result.code === 101)) {
            const results = Object.assign({}, documentResults)
            results[path] = result
            documentResults = results
            states[path] = result.partial ? "partial" : result.code === 101 ? "empty" : "complete"
            const count = result.metadata && result.metadata.pageCount || (result.data && result.data.pages || []).length
            const blocks = result.metadata && result.metadata.blockCount || 0
            statusText = result.partial ? qsTr("文档部分完成：%1 页，%2 个版面块；详情见结果提示。").arg(count).arg(blocks)
                : result.code === 101 ? qsTr("解析完成，未检测到可提取内容。")
                : qsTr("文档解析完成：%1 页，%2 个版面块。").arg(count).arg(blocks)
        } else {
            states[path] = documentResults[path] ? "partial" : "failed"
            statusText = result && typeof result.data === "string" ? result.data : qsTr("文档解析失败。")
            qmlapp.popup.message(qsTr("文档解析失败"), statusText, "error")
        }
        documentStates = states
    }

    function onMissionEnd(missionId, message) {
        if(!acceptMission(missionId)) return
        parsing = false
        cancelRequested = false
        const states = Object.assign({}, documentStates)
        for(const path of runPaths)
            if(states[path] === "pending" || states[path] === "running") states[path] = documentResults[path] ? "partial" : "failed"
        documentStates = states
        if(message && message.startsWith("[Error]")) {
            statusText = message
            qmlapp.popup.message(qsTr("文档解析异常"), message, "error")
        } else if(message && message.startsWith("[Warning]")) statusText = message
    }

    Component.onCompleted: {
        const state = callPy("initPlugin")
        if(!state || !state.ok) statusText = state && state.message ? state.message : qsTr("文档解析页面初始化失败。")
        else documentNameFilter = state.details && state.details.documentNameFilter || ""
        configs.reload()
    }

    Configs {
        id: configs
        visible: false
        autoLoad: false
        category_: "StructuredDocument"
        configDict: ({"document": qmlapp.globalConfigs.documentManager.deploy(this, "document")})
    }

    FileDialog_ {
        id: openDialog
        title: qsTr("选择要解析的图片或 PDF")
        selectMultiple: true
        nameFilters: page.documentNameFilter ? [page.documentNameFilter] : []
        onAccepted: page.addPaths(fileUrls_)
    }
    FileDialog_ {
        id: exportDialog
        title: qsTr("选择导出目录")
        selectFolder: true
        onAccepted: {
            const result = page.callPy("exportResult", page.structuredResult, fileUrls_.length ? fileUrls_[0] : "", "all")
            if(result && result.ok) qmlapp.popup.simple(qsTr("导出完成"), result.message || "")
            else qmlapp.popup.message(qsTr("导出失败"), result && result.message ? result.message : qsTr("未知错误"), "error")
        }
    }

    DocPreviewConnector {
        id: previewConnector
        Component.onDestruction: clear()
        onPreviewInfo: function(info) {
            if(!info || info.path !== page.currentPath || info.page !== page.currentPage) return
            const documents = Object.assign({}, page.documentInfo)
            documents[info.path] = info
            page.documentInfo = documents
            page.pageCount = Number(info.pageCount || 0)
            if(info.error) page.previewError = info.error === "[Warning] is_encrypted" ? qsTr("加密文档暂不支持预览，请先解密。") : info.error
        }
        onPreviewImg: function(imgID) {
            page.previewLoading = false
            if(!imgID || imgID.startsWith("[Error]") || imgID.startsWith("[Warning]")) {
                preview.clear()
                if(imgID && !page.previewError) page.previewError = imgID
                return
            }
            preview.showImgID(imgID)
        }
    }

    component PageNavigator: Row {
        property string namePrefix: ""
        spacing: size_.smallSpacing
        height: size_.line * 1.75
        Button_ {
            objectName: namePrefix + "Previous"
            text_: "‹"
            toolTip: qsTr("上一页")
            width: size_.line * 1.75
            height: parent.height
            enabled: page.currentPage > 1
            onClicked: page.setCurrentPage(page.currentPage - 1)
        }
        SpinBox {
            objectName: namePrefix + "PageNumber"
            width: size_.line * 5
            height: parent.height
            font.pixelSize: size_.text
            editable: true
            from: 1
            to: Math.max(1, page.pageCount)
            value: page.currentPage
            enabled: page.pageCount > 0
            onValueModified: page.setCurrentPage(value)
        }
        Text_ { text: "/ " + page.pageCount; height: parent.height; verticalAlignment: Text.AlignVCenter }
        Button_ {
            objectName: namePrefix + "Next"
            text_: "›"
            toolTip: qsTr("下一页")
            width: size_.line * 1.75
            height: parent.height
            enabled: page.currentPage < page.pageCount
            onClicked: page.setCurrentPage(page.currentPage + 1)
        }
    }

    DoubleSwitchableLayout {
        id: doubleLayout
        anchors.fill: parent
        saveKey: "StructuredDocument_1"
        itemA: Panel {
            objectName: "structuredPreviewPanel"
            anchors.fill: parent
            Item {
                id: taskControls
                anchors.top: parent.top
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.margins: size_.spacing
                height: size_.line * 2
                Button_ {
                    id: runButton
                    objectName: "structuredRunCancel"
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.bottom: parent.bottom
                    width: size_.line * 8
                    bold_: true
                    bgColor_: theme.coverColor1
                    bgHoverColor_: theme.coverColor2
                    text_: page.cancelRequested ? qsTr("正在取消…") : page.parsing ? qsTr("取消任务") : qsTr("开始解析")
                    enabled: !page.cancelRequested
                    onClicked: page.parsing ? page.cancelParse() : page.startParse()
                }
                Text_ {
                    anchors.left: parent.left
                    anchors.right: runButton.left
                    anchors.rightMargin: size_.spacing
                    anchors.top: parent.top
                    font.pixelSize: size_.smallText
                    text: page.statusText
                    elide: Text.ElideRight
                }
                HProgressBar {
                    anchors.left: parent.left
                    anchors.right: runButton.left
                    anchors.rightMargin: size_.spacing
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: size_.line * 0.1
                    height: size_.line * 0.5
                    color: theme.bgColor
                    highlightColor: page.parsing ? theme.yesColor : theme.coverColor4
                    percent: page.runPaths.length ? Math.min(1, (page.finishedPaths.length + page.pageProgress) / page.runPaths.length) : 0
                }
            }
            Column {
                id: previewHeader
                anchors.top: taskControls.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.margins: size_.spacing
                anchors.topMargin: size_.smallSpacing
                spacing: size_.smallSpacing
                Row {
                    width: parent.width
                    height: size_.line * 1.75
                    spacing: size_.smallSpacing
                    Button_ { objectName: "structuredOpen"; height: parent.height; text_: qsTr("打开"); enabled: !page.parsing; onClicked: page.openFiles() }
                    Button_ { objectName: "structuredPaste"; height: parent.height; text_: qsTr("粘贴"); enabled: !page.parsing; onClicked: page.paste() }
                    Button_ { objectName: "structuredClear"; height: parent.height; text_: qsTr("清空"); enabled: !page.parsing && page.inputPaths.length > 0; onClicked: page.resetPage() }
                }
                ComboBox {
                    objectName: "structuredDocumentSelector"
                    width: parent.width
                    height: visible ? size_.line * 1.75 : 0
                    visible: page.inputPaths.length > 1
                    model: page.inputPaths.map(path => ({label: page.pathName(path), path: path}))
                    textRole: "label"
                    font.pixelSize: size_.text
                    currentIndex: page.inputPaths.indexOf(page.currentPath)
                    onActivated: page.selectDocument(page.inputPaths[currentIndex])
                }
                Item {
                    width: parent.width
                    height: size_.line * 1.75
                    Text_ {
                        anchors.left: parent.left
                        anchors.right: previewNavigation.left
                        anchors.rightMargin: size_.spacing
                        anchors.verticalCenter: parent.verticalCenter
                        text: page.currentPath ? page.pathName(page.currentPath) : qsTr("页面预览")
                        elide: Text.ElideMiddle
                        color: theme.subTextColor
                    }
                    PageNavigator { id: previewNavigation; namePrefix: "preview"; anchors.right: parent.right }
                }
            }
            ImageScale {
                id: preview
                objectName: "structuredPagePreview"
                anchors.top: previewHeader.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                anchors.margins: size_.spacing
                overlayLayer: Item {
                    width: preview.imageSW
                    height: preview.imageSH
                    visible: !page.previewLoading
                    Repeater {
                        model: page.currentBlocks
                        delegate: Item {
                            id: region
                            objectName: "documentRegion:" + page.blockKey(modelData)
                            property var points: {
                                const source = modelData.polygon && modelData.polygon.length >= 3 ? modelData.polygon : modelData.bbox
                                const result = page.currentResultPage
                                if(!result || !(result.width > 0) || !(result.height > 0)
                                        || !(preview.imageSW > 0) || !(preview.imageSH > 0)) return []
                                const sx = preview.imageSW / result.width, sy = preview.imageSH / result.height
                                return Array.from(source || []).map(point => [Number(point[0]) * sx, Number(point[1]) * sy])
                            }
                            property var bounds: {
                                let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
                                for(const point of points || []) {
                                    minX = Math.min(minX, Number(point[0])); minY = Math.min(minY, Number(point[1]))
                                    maxX = Math.max(maxX, Number(point[0])); maxY = Math.max(maxY, Number(point[1]))
                                }
                                return isFinite(minX) ? [minX, minY, Math.max(1, maxX-minX), Math.max(1, maxY-minY)] : [0, 0, 0, 0]
                            }
                            x: bounds[0]; y: bounds[1]; width: bounds[2]; height: bounds[3]
                            onPointsChanged: regionCanvas.requestPaint()
                            Canvas {
                                id: regionCanvas
                                anchors.fill: parent
                                onPaint: function(paintRegion) {
                                    const ctx = getContext("2d")
                                    ctx.clearRect(0, 0, width, height)
                                    if(!region.points || region.points.length < 3) return
                                    ctx.beginPath()
                                    ctx.moveTo(region.points[0][0] - region.x, region.points[0][1] - region.y)
                                    for(let i = 1; i < region.points.length; ++i) ctx.lineTo(region.points[i][0] - region.x, region.points[i][1] - region.y)
                                    ctx.closePath()
                                    const active = page.selectedBlockKey === page.blockKey(modelData)
                                    ctx.fillStyle = active ? "#3A22A05A" : "#2022A05A"
                                    ctx.strokeStyle = active ? "#22A05A" : "#4A9BFF"
                                    ctx.lineWidth = active ? 3 : 1
                                    ctx.fill(); ctx.stroke()
                                }
                                Connections { target: page; function onSelectedBlockKeyChanged() { regionCanvas.requestPaint() } }
                            }
                            MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: page.selectBlock(modelData) }
                        }
                    }
                }
                Text_ {
                    anchors.fill: parent
                    anchors.margins: size_.spacing
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    wrapMode: Text.Wrap
                    visible: !page.currentPath || !!page.previewError || page.previewLoading
                    text: page.previewError || (page.previewLoading ? qsTr("正在读取页面…") : qsTr("拖入图片或 PDF"))
                }
            }
        }
        itemB: Panel {
            objectName: "structuredResultPanel"
            anchors.fill: parent
            TabPanel {
                id: mainTabs
                objectName: "structuredMainTabs"
                anchors.fill: parent
                anchors.margins: size_.spacing
                isMenuTop: doubleLayout.isRow
                menuHeight: size_.line * 2
                tabsModel: [
                    {key: "configs", title: qsTr("设置"), component: configs.panelComponent},
                    {key: "results", title: qsTr("结果"), component: resultArea}
                ]
            }
        }
    }
    Binding { target: configs.panelComponent; property: "enabled"; value: !page.parsing }
    Binding { target: configs.panelComponent.ctrlBar; property: "enabled"; value: !page.parsing }

    Item {
        id: resultArea
        visible: false
        anchors.fill: parent
        property alias ctrlBar: resultControls
        Row {
            id: resultControls
            spacing: size_.smallSpacing
            Button_ { objectName: "structuredCopy"; height: parent.height; text_: qsTr("复制"); onClicked: page.copyResult() }
            Button_ { objectName: "structuredExport"; height: parent.height; text_: qsTr("导出"); onClicked: page.exportAll() }
        }
        Item {
            id: resultNavigation
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            height: size_.line * 1.75
            ComboBox {
                objectName: "structuredResultScope"
                anchors.left: parent.left
                width: size_.line * 6
                height: parent.height
                model: [qsTr("当前页"), qsTr("整篇")]
                font.pixelSize: size_.text
                currentIndex: page.wholeDocument ? 1 : 0
                onActivated: page.wholeDocument = currentIndex === 1
            }
            PageNavigator { namePrefix: "result"; anchors.right: parent.right }
        }
        Text_ {
            id: resultState
            objectName: "structuredResultStatus"
            anchors.top: resultNavigation.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.topMargin: size_.smallSpacing
            color: theme.subTextColor
            font.pixelSize: size_.smallText
            wrapMode: Text.Wrap
            text: page.resultStatus()
        }
        TabPanel {
            id: resultTabs
            objectName: "structuredResultTabs"
            anchors.top: resultState.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            anchors.topMargin: size_.smallSpacing
            menuHeight: size_.line * 1.75
            tabsModel: [
                {key: "preview", title: qsTr("预览"), component: textView},
                {key: "markdown", title: qsTr("Markdown 源码"), component: markdownView},
                {key: "json", title: qsTr("JSON"), component: jsonView},
                {key: "blocks", title: qsTr("结构块"), component: blockList}
            ]
        }
        DocumentMarkdownPreview {
            id: textView
            objectName: "structuredRenderedPreview"
            anchors.fill: parent
            visible: false
            text: page.resultText(0)
            resources: page.wholeDocument ? ((page.structuredResult.data || {}).assets || [])
                : ((page.currentResultPage || {}).resources || [])
        }
        ScrollView {
            id: markdownView
            objectName: "structuredMarkdownScroll"
            anchors.fill: parent
            visible: false
            clip: true
            contentWidth: availableWidth
            TextArea {
                objectName: "structuredMarkdownContent"
                width: markdownView.availableWidth
                readOnly: true
                selectByMouse: true
                selectByKeyboard: true
                textFormat: TextEdit.PlainText
                wrapMode: TextEdit.Wrap
                font.family: theme.dataFontFamily
                font.pixelSize: size_.text
                color: theme.textColor
                background: null
                text: page.resultText(1)
            }
        }
        ScrollView {
            id: jsonView
            objectName: "structuredJsonScroll"
            anchors.fill: parent
            visible: false
            clip: true
            contentWidth: Math.max(availableWidth, jsonText.implicitWidth)
            TextArea {
                id: jsonText
                objectName: "structuredJsonContent"
                width: Math.max(jsonView.availableWidth, implicitWidth)
                readOnly: true
                selectByMouse: true
                selectByKeyboard: true
                textFormat: TextEdit.PlainText
                wrapMode: TextEdit.NoWrap
                font.family: theme.dataFontFamily
                font.pixelSize: size_.text
                color: theme.textColor
                background: null
                text: page.resultText(2)
            }
        }
        ListView {
            id: blockList
            objectName: "structuredBlocks"
            anchors.fill: parent
            visible: false
            model: page.resultBlocks
            clip: true
            delegate: Rectangle {
                width: blockList.width
                height: Math.max(size_.line * 1.7, blockText.implicitHeight + size_.smallSpacing * 2)
                color: page.selectedBlockKey === page.blockKey(modelData) ? theme.coverColor2 : "transparent"
                border.width: 1
                border.color: theme.coverColor3
                radius: size_.btnRadius
                Text_ {
                    id: blockText
                    anchors.fill: parent
                    anchors.margins: size_.smallSpacing
                    wrapMode: Text.Wrap
                    text: (page.wholeDocument ? qsTr("第 %1 页").arg(modelData.page) + " · " : "")
                        + `[${modelData.order === null || modelData.order === undefined ? "-" : modelData.order}] ${modelData.categoryName || modelData.category}: ${modelData.content || ""}`
                }
                MouseArea { anchors.fill: parent; onClicked: page.selectBlock(modelData) }
            }
            ScrollBar.vertical: ScrollBar { }
        }
    }
    DropArea_ { anchors.fill: parent; enabled: !page.parsing; callback: page.addPaths }
}

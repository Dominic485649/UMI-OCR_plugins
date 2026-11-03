"""输入/解码/分块预算。仅触及任务自有临时目录，不修改用户输入。"""
from __future__ import annotations

import contextlib
import math
import time
from pathlib import Path

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000
MAX_IMAGE_SIDE = 32_000
MAX_TILES = 32
MAX_TILE_CALLS = 64
MAX_DOCUMENT_PAGES = 500
MAX_DOCUMENT_PIXELS = 160_000_000
TILE_PIXELS = 1_605_632
TILE_SIDE = 2048
TILE_OVERLAP = 128


try:
    from .checks import tr
except ImportError:
    from checks import tr


class BudgetError(RuntimeError):
    def __init__(self, message):
        super().__init__(tr(message))


def remaining(deadline, request_timeout):
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError(tr("TASK_DEADLINE：任务执行预算已用尽，请减少页数或拆分图片后重试。"))
    return max(0.01, min(float(request_timeout), left))


def check_file(source, max_bytes=MAX_FILE_BYTES):
    size = Path(source).stat().st_size
    if size <= 0 or size > max_bytes:
        raise BudgetError(tr('INPUT_FILE_BUDGET：输入文件须为非空且不超过 64 MiB，请拆分或另存后重试。'))


def check_size(width, height):
    if width < 1 or height < 1 or width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE or width * height > MAX_IMAGE_PIXELS:
        raise BudgetError(tr('IMAGE_DECODE_BUDGET：图片超过 32000 单边或 3200 万像素解码预算，请拆分图片后重试。'))
    return int(width), int(height)


def image_size(source, prepared=False):
    from PIL import Image
    check_file(source, 128 * 1024 * 1024 if prepared else MAX_FILE_BYTES)
    with Image.open(source) as image:
        return check_size(*image.size)


def tile_boxes(width, height, max_pixels=TILE_PIXELS):
    """完整覆盖四边，重叠不裁掉边缘区域；不把整幅长图先缩小。"""
    check_size(width, height)
    if width * height <= max_pixels and max(width, height) <= TILE_SIDE:
        return [(0, 0, width, height)]
    tile_w = min(width, TILE_SIDE, max(1, int(math.sqrt(max_pixels))))
    tile_h = min(height, TILE_SIDE, max(1, max_pixels // tile_w))
    # 窄长图尽量利用整宽；横长图同理。
    if width <= tile_w:
        tile_h = min(height, TILE_SIDE, max_pixels // width)
    elif height <= tile_h:
        tile_w = min(width, TILE_SIDE, max_pixels // height)
    def starts(length, tile):
        if length <= tile:
            return [0]
        step = max(1, tile - min(TILE_OVERLAP, tile // 4))
        values = list(range(0, length - tile + 1, step))
        if values[-1] != length - tile:
            values.append(length - tile)
        return values
    boxes = [(x, y, min(width, x + tile_w), min(height, y + tile_h)) for y in starts(height, tile_h) for x in starts(width, tile_w)]
    if len(boxes) > MAX_TILES:
        raise BudgetError(tr('TILE_COUNT_BUDGET：图片需要超过 32 个分块，请按页或区域拆分后重试。'))
    return boxes


def split_box(box):
    """截断时最多再分两层，沿长边分割并保留重叠。"""
    x1, y1, x2, y2 = box
    if max(x2 - x1, y2 - y1) < 384:
        return []
    overlap = 64
    if x2 - x1 >= y2 - y1:
        middle = (x1 + x2) // 2
        return [(x1, y1, middle + overlap, y2), (middle - overlap, y1, x2, y2)]
    middle = (y1 + y2) // 2
    return [(x1, y1, x2, middle + overlap), (x1, middle - overlap, x2, y2)]


def selected_pages(page_range, count):
    if count < 1:
        return []
    if page_range is None:
        start, end = 1, count
    else:
        if not isinstance(page_range, (list, tuple)) or len(page_range) != 2 or any(type(v) is not int for v in page_range):
            raise BudgetError(tr('INVALID_PAGE_RANGE：页范围必须是 [起始页, 结束页] 两个整数。'))
        start, end = page_range
        if start < 1 or end < start or start > count:
            raise BudgetError(tr('INVALID_PAGE_RANGE：页范围超出文档，请检查原始页号。'))
        end = min(end, count)
    if end - start + 1 > MAX_DOCUMENT_PAGES:
        raise BudgetError(tr('DOCUMENT_PAGE_BUDGET：每个任务最多处理 500 页，请缩小页范围。'))
    return list(range(start, end + 1))


@contextlib.contextmanager
def document_pages(source, page_range, temp_dir, deadline, prepared=False):
    """predict 无页范围参数：先只渲染选中页，再交给官方 predict/restructure。

    pypdfium2 的 PdfDocument[index]/page.get_size()/page.render(scale=2)/
    bitmap.to_pil() 均已核对私有闭包实际签名；不调用 list(doc) 或整份 reader。
    """
    from PIL import Image
    source, temp_dir = Path(source), Path(temp_dir)
    check_file(source, 128 * 1024 * 1024 if prepared else MAX_FILE_BYTES)
    if source.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(source))
        try:
            selection = selected_pages(page_range, len(doc))
            def iterate():
                total_pixels = 0
                for number in selection:
                    remaining(deadline, 1)
                    page = doc[number - 1]
                    try:
                        width, height = page.get_size()
                        if not math.isfinite(width) or not math.isfinite(height):
                            raise BudgetError(tr('IMAGE_DECODE_BUDGET：PDF 页面尺寸无效。'))
                        check_size(math.ceil(width * 2), math.ceil(height * 2))
                        total_pixels += math.ceil(width * 2) * math.ceil(height * 2)
                        if total_pixels > MAX_DOCUMENT_PIXELS:
                            raise BudgetError(tr('DOCUMENT_PIXEL_BUDGET：选中页超过累计渲染预算，请减少页数。'))
                        bitmap = page.render(scale=2)
                        try:
                            image = bitmap.to_pil()
                            try:
                                target = temp_dir / ("page-%d.png" % number)
                                image.save(target, format="PNG")
                            finally:
                                image.close()
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
                    try:
                        yield number, target
                    finally:
                        target.unlink(missing_ok=True)
            yield selection, iterate()
        finally:
            doc.close()
    else:
        with Image.open(source) as image:
            check_size(*image.size)
            selection = selected_pages(page_range, getattr(image, "n_frames", 1))
            def iterate():
                total_pixels = 0
                for number in selection:
                    remaining(deadline, 1)
                    image.seek(number - 1)
                    check_size(*image.size)
                    total_pixels += image.width * image.height
                    if total_pixels > MAX_DOCUMENT_PIXELS:
                        raise BudgetError(tr('DOCUMENT_PIXEL_BUDGET：选中页超过累计解码预算，请减少页数。'))
                    target = temp_dir / ("page-%d.png" % number)
                    from PIL import ImageOps
                    oriented = ImageOps.exif_transpose(image)
                    rgba = oriented.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    composited = Image.alpha_composite(background, rgba)
                    frame = composited.convert("RGB")
                    try:
                        frame.save(target, format="PNG")
                    finally:
                        frame.close()
                        composited.close()
                        background.close()
                        rgba.close()
                        if oriented is not image:
                            oriented.close()
                    try:
                        yield number, target
                    finally:
                        target.unlink(missing_ok=True)
            yield selection, iterate()

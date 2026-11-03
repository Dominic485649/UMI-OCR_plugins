"""在实际私有依赖中执行锁定 PaddleX 跨页表格合并，不初始化模型或伪造推理结果。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vendor', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--html-deps', type=Path)
    args = parser.parse_args()
    vendor = args.vendor.resolve()
    sys.path[:0] = ([str(args.html_deps.resolve())] if args.html_deps else []) + [str(vendor), str(vendor.parent / 'vendor.zip')]
    from bs4 import BeautifulSoup
    import bs4
    import soupsieve
    assert bs4.__version__ == '4.15.0'
    assert soupsieve.__version__ == '3.0.4'
    source = vendor / 'paddlex/inference/pipelines/layout_parsing/merge_table.py'
    spec = importlib.util.spec_from_file_location('locked_paddlex_table_merge', source)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    def block(index, content, width=100):
        return SimpleNamespace(label='table', content=content, bbox=[0, 0, width, 100],
            global_block_id=index, global_group_id=index)
    first = block(0, '<table><tr><th>Header</th></tr><tr><td>FIRST12345</td></tr></table>')
    second = block(1, '<table><tr><th>Header</th></tr><tr><td>SECOND67890</td></tr></table>')
    pages = official.merge_tables_across_pages([[first], [second]])
    soup = BeautifulSoup(first.content, 'html.parser')
    assert len(soup.select('tr')) == 3 and len(soup.select('th')) == 1
    assert 'FIRST12345' in soup.get_text() and 'SECOND67890' in soup.get_text()
    assert pages[1][0].content == '' and pages[1][0].global_group_id == 0
    other = block(1, '<table><tr><td>SECOND</td></tr></table>', width=200)
    assert official.can_merge_tables([first], first, [other], other)[0] is False
    image = SimpleNamespace(label='image', content='<img src="keep.png">')
    assert official.can_merge_tables([first, image], first, [second], second)[0] is False
    merged_cells = BeautifulSoup('<table><tr><td rowspan="2">A</td><td colspan="2">B</td></tr><tr><td>C</td><td>D</td></tr></table>', 'html.parser')
    assert official.calculate_table_total_columns(merged_cells) == 3
    assert 'paddle' not in sys.modules
    result = {'status': 'PASS', 'modelInference': 'NOT_RUN', 'officialPaddleXCode': str(source),
        'officialCodeSha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'beautifulsoup4': bs4.__version__, 'soupsieve': soupsieve.__version__,
        'checks': ['实际官方跨页表格合并', '重复表头仅保留一次', '跨页块分组ID保留', '不同宽度/图像阻隔不误合并', 'rowspan/colspan列数']}
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

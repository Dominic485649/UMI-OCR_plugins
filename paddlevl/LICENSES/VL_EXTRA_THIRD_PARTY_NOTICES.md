# PaddleOCR-VL extra runtime notices

The document plugin adds dependencies for PaddleX title-level restructuring
and cross-page HTML table merging after layout detection. The first four
packages below use BSD 3-Clause; Beautiful Soup and SoupSieve use MIT. The base
GPU closure already supplies the matching NumPy and typing-extensions runtime.
Exact wheel hashes are retained in `requirements-vl-extra.lock`; unmodified
Python sources and original per-package notices remain inside vendor.

| Package | Version | Source | Wheel SHA-256 |
| --- | --- | --- | --- |
| joblib | 1.5.3 | https://github.com/joblib/joblib/tree/1.5.3 | `5fc3c5039fc5ca8c0276333a188bbd59d6b7ab37fe6632daa76bc7f9ec18e713` |
| scikit-learn | 1.7.2 | https://github.com/scikit-learn/scikit-learn/tree/1.7.2 | `63a9afd6f7b229aad94618c01c252ce9e6fa97918c5ca19c9a17a087d819440c` |
| SciPy | 1.18.0 | https://github.com/scipy/scipy/tree/v1.18.0 | `1afac4a847207c7ff8efd321734a50b06d0280b3b2a2c0fc2f413101747ad7c7` |
| threadpoolctl | 3.6.0 | https://github.com/joblib/threadpoolctl/tree/3.6.0 | `43a0b8fd5a2928500110039e43a5eed8480b918967083ea48dc3ab9f13c4a7fb` |

## BSD 3-Clause License

Copyright (c) the respective copyright holders.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## HTML table dependencies (MIT)

| Package | Version | Source | Wheel SHA-256 |
| --- | --- | --- | --- |
| Beautiful Soup | 4.15.0 | https://pypi.org/project/beautifulsoup4/4.15.0/ | `d6f88de62e1d4e38ecb1077eb9724cd0eff29d2a08ca16a401e9b9e93f117cf9` |
| SoupSieve | 3.0.4 | https://pypi.org/project/soupsieve/3.0.4/ | `8412826614f9c632227067d0985aeb9f584bd15efc852955430e097a9fe7afbf` |

### Beautiful Soup original notice

Beautiful Soup is made available under the MIT license:

 Copyright (c) Leonard Richardson

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

Beautiful Soup incorporates code from the html5lib library, which is
also made available under the MIT license. Copyright (c) James Graham
and other contributors

Beautiful Soup has an optional dependency on the soupsieve library,
which is also made available under the MIT license. Copyright (c)
Isaac Muse

### SoupSieve original notice

MIT License

Copyright (c) 2018 - 2026 Isaac Muse <isaacmuse@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

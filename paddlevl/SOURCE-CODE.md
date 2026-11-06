# PaddleOCR-VL 3.1.3 local-candidate source notice

This archive retains the minimal corresponding plugin source under source/;
it does not contain GGUF, mmproj, or PP-DocLayoutV3 weights.
The fixed llama.cpp runtime, its licenses and source/provenance are included in
`win32_PaddleOCRVL_1_6/resources/llama.cpp/` inside the installable plugin.

- Plugin repository commit: 0fd11167573f08de7629a7fae01d1a9fe3501244
- Uncommitted source changes included: True. The commit alone is not the artifact identity; retained source files and runtime hashes bind this local candidate.
- Local tag: v3.1.3 (local tag pending at build time)
- Upstream/reference repository: https://github.com/Dominic485649/Umi-OCR-Plugin
- Public source URL: **PENDING** (no user-owned 3.1.3 remote is published)
- Build entry: Build-Plugin.ps1 -LocalCandidate
- Runtime lock SHA-256: f8dfac1cca93dee10a4b309a75f3d2002e7c7ce4b2314c249c8d544601d0d683
- Bundled llama.cpp manifest SHA-256: a7722c76d3e452de9de7e0752c6630ebae96cc4611f23553c5d6460d2900b18d
- Bundled llama.cpp build identity, exact source archive, build argv, binary
  origins and license terms: `resources/llama.cpp/source/` and `LICENSES/`.
- VL extra lock SHA-256: 7688687015600573b8d471e139180d8f79bb00edd68e4e64bc9dbcd3a9b575f5
- PaddleOCR 3.7.0 wheel: https://pypi.org/project/paddleocr/3.7.0/
  (SHA-256: c0f0a81ad4112727f30c6fcf986ac0ef6a120d31ee0991a01fae0357ee32d338); restored PaddleOCRVL files come from this
  exact wheel.
- PaddleX 3.7.2 wheel: https://pypi.org/project/paddlex/3.7.2/
  (SHA-256: f1678bf650bbaccfd8f0d4e49d0ae631b4685c829fdae6e802ccd90d4fcb9a7f); the VL package contains this independent
  locked PaddleX runtime after the recorded compatibility patch.
- PaddleX VL source patch: tools/patch_paddlex_vl_3_7_2.py narrows the upstream
  `ocr` gate to the bundled `ocr-core` feature for the llama.cpp-server
  path; patched file SHA-256: 88b5e26e05927be0dd1ab6b1a9e6a59be0bb5d416797399cb5382474b086ea64.
- Other third-party source URLs/revisions/hashes: recorded by the included
  lock and package metadata; no unpinned source is implied.

The candidate is not approved for public release until the exact source URL is
available without an additional fee.

"""Static Windows packaging contract tests; no model, network or GPU required."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import tarfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "bundle_llama_runtime.py"
spec = importlib.util.spec_from_file_location("bundle_llama_runtime", TOOL)
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def pe_bytes(imports=(), delayed=(), machine=0x8664):
    data = bytearray(4096)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HH", data, 0x84, machine, 1)
    struct.pack_into("<H", data, 0x94, 240)
    optional = 0x98
    struct.pack_into("<H", data, optional, 0x20B)
    struct.pack_into("<Q", data, optional + 24, 0x140000000)
    struct.pack_into("<I", data, optional + 108, 16)
    struct.pack_into("<IIII", data, optional + 240 + 8, 0xE00, 0x1000, 0xE00, 0x200)
    strings = 0xA00
    for entry, names, table, width in ((1, imports, 0x200, 20), (13, delayed, 0x500, 32)):
        if not names:
            continue
        struct.pack_into("<II", data, optional + 112 + entry * 8, table + 0xE00, (len(names) + 1) * width)
        for index, name in enumerate(names):
            encoded = name.encode("ascii") + b"\0"
            data[strings:strings + len(encoded)] = encoded
            fields = [0] * (width // 4)
            if entry == 1:
                fields[3] = strings + 0xE00
            else:
                fields[0], fields[1] = 1, strings + 0xE00
            struct.pack_into("<" + "I" * len(fields), data, table + index * width, *fields)
            strings += len(encoded)
    return bytes(data)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="llama-contract-", dir=TEST_ROOT)
        self.root = Path(self.temp.name) / "runtime"
        self.root.mkdir()
        for name in bundle.LICENSE_FILES | {
            "source/provenance.json", f"source/llama.cpp-{bundle.COMMIT}-runtime-source.tar",
            "source/build-inputs.json", "source/build_bundled_llama.ps1",
        }:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic packaging fixture; not a real license/runtime")
        (self.root / "llama-server.exe").write_bytes(pe_bytes(["ggml-cuda.dll", "msvcp140.dll", "KERNEL32.dll"]))
        (self.root / "ggml-cuda.dll").write_bytes(pe_bytes(["cudart64_13.dll", "cublas64_13.dll", "nvcuda.dll"]))
        (self.root / "cudart64_13.dll").write_bytes(pe_bytes(["KERNEL32.dll"]))
        (self.root / "cublas64_13.dll").write_bytes(pe_bytes(["KERNEL32.dll", "cublasLt64_13.dll"]))
        (self.root / "cublasLt64_13.dll").write_bytes(pe_bytes(["KERNEL32.dll"]))
        (self.root / "msvcp140.dll").write_bytes(pe_bytes(["api-ms-win-crt-runtime-l1-1-0.dll"]))
        identity = {"commit": bundle.COMMIT, "cudaToolkit": "13.4.92", "architectures": bundle.ARCHITECTURES,
                    "buildOptions": {"GGML_CUDA": "ON", "GGML_NATIVE": "OFF"}}
        with tarfile.open(self.root / f"source/llama.cpp-{bundle.COMMIT}-runtime-source.tar", "w") as archive:
            info = tarfile.TarInfo("CMakeLists.txt")
            payload = b"# synthetic corresponding source"
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        source_hash = bundle.sha256(self.root / f"source/llama.cpp-{bundle.COMMIT}-runtime-source.tar")
        (self.root / "source/build-inputs.json").write_text(json.dumps(dict(identity, sourceArchiveSha256=source_hash)), encoding="utf-8")
        (self.root / "source/provenance.json").write_text(json.dumps({"sourceArchiveSha256": source_hash, "buildInputArchiveSha256": source_hash}), encoding="utf-8")
        self.manifest = {
            "schemaVersion": 1,
            "identity": {"commit": bundle.COMMIT, "cudaToolkit": "13.4.92", "architectures": bundle.ARCHITECTURES,
                         "buildOptions": {"GGML_CUDA": "ON", "GGML_NATIVE": "OFF"}},
            "files": [{"path": p.relative_to(self.root).as_posix(), "size": p.stat().st_size, "sha256": bundle.sha256(p)}
                      for p in sorted(self.root.rglob("*")) if p.is_file()],
        }
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.root / "runtime-manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_valid_manifest_and_closure(self):
        self.assertEqual(bundle.validate(self.root)["identity"]["commit"], bundle.COMMIT)

    def test_normal_and_delayed_imports(self):
        path = Path(self.temp.name) / "delay.dll"
        path.write_bytes(pe_bytes(["kernel32.dll"], ["vcruntime140.dll"]))
        self.assertEqual(bundle.pe_imports(path), ["kernel32.dll", "vcruntime140.dll"])

    def test_reject_x86(self):
        path = Path(self.temp.name) / "x86.dll"
        path.write_bytes(pe_bytes(machine=0x14C))
        with self.assertRaisesRegex(ValueError, "x64"):
            bundle.pe_imports(path)

    def test_reject_non_pe(self):
        path = Path(self.temp.name) / "invalid.dll"
        path.write_bytes(b"not PE")
        with self.assertRaises(ValueError):
            bundle.pe_imports(path)

    def test_hash_tamper(self):
        path = self.root / "llama-server.exe"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "hash"):
            bundle.validate(self.root)

    def test_missing_dll(self):
        # Construct a bundle without a required DLL rather than relying on hash failure.
        (self.root / "cublas64_13.dll").unlink()
        self.manifest["files"] = [p for p in self.manifest["files"] if p["path"] != "cublas64_13.dll"]
        self.save()
        with self.assertRaisesRegex(ValueError, "closure"):
            bundle.validate(self.root)

    def test_unknown_dependency(self):
        (self.root / "cublas64_13.dll").write_bytes(pe_bytes(["not-reviewed.dll"]))
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            bundle.closure_check(self.root)

    def test_unneeded_binary(self):
        (self.root / "unused.dll").write_bytes(pe_bytes())
        with self.assertRaisesRegex(ValueError, "Unneeded"):
            bundle.closure_check(self.root)

    def test_driver_must_not_be_bundled(self):
        (self.root / "nvcuda.dll").write_bytes(pe_bytes())
        with self.assertRaisesRegex(ValueError, "driver"):
            bundle.closure_check(self.root)

    def test_unlisted_file(self):
        (self.root / "unexpected.txt").write_text("extra")
        with self.assertRaisesRegex(ValueError, "coverage"):
            bundle.validate(self.root)

    def test_manifest_self_reference(self):
        self.manifest["files"][0]["path"] = "runtime-manifest.json"
        self.save()
        with self.assertRaisesRegex(ValueError, "self"):
            bundle.validate(self.root)

    def test_windows_paths_and_traversal(self):
        for name in ("../outside", "a/../b", "a\\b", "C:/outside", "/absolute", "x:stream", "a//b", "a/./b", "a./b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                bundle.safe_relative(name)

    def test_case_collision(self):
        self.manifest["files"].append(dict(self.manifest["files"][0], path=self.manifest["files"][0]["path"].upper()))
        self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            bundle.validate(self.root)

    def test_wrong_identity(self):
        self.manifest["identity"]["architectures"] = ["89"]
        self.save()
        with self.assertRaisesRegex(ValueError, "architecture"):
            bundle.validate(self.root)

    def test_source_and_license_required(self):
        name = "LICENSES/MSVC-license.html"
        (self.root / name).unlink()
        self.manifest["files"] = [p for p in self.manifest["files"] if p["path"] != name]
        self.save()
        with self.assertRaisesRegex(ValueError, "licenses/source"):
            bundle.validate(self.root)

    def test_no_overwrite_and_verified_copy(self):
        destination = Path(self.temp.name) / "copy"
        bundle.copy_bundle(self.root, destination)
        self.assertEqual(bundle.validate(destination), self.manifest)
        with self.assertRaisesRegex(ValueError, "no overwrite"):
            bundle.copy_bundle(self.root, destination)

    def test_model_rejected(self):
        path = self.root / "model.gguf"
        path.write_bytes(b"no models")
        self.manifest["files"].append({"path": "model.gguf", "size": path.stat().st_size, "sha256": bundle.sha256(path)})
        self.save()
        with self.assertRaisesRegex(ValueError, "Models"):
            bundle.validate(self.root)

    def test_statically_linked_cudart_needs_no_runtime_dll(self):
        (self.root / "ggml-cuda.dll").write_bytes(pe_bytes(["cublas64_13.dll", "nvcuda.dll"]))
        (self.root / "cudart64_13.dll").unlink()
        self.assertNotIn("cudart64_13.dll", bundle.closure_check(self.root))

    def test_missing_imported_cudart_is_rejected(self):
        (self.root / "cudart64_13.dll").unlink()
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            bundle.closure_check(self.root)

    def test_source_archive_provenance_must_match(self):
        path = self.root / f"source/llama.cpp-{bundle.COMMIT}-runtime-source.tar"
        path.write_bytes(b"changed source archive")
        for entry in self.manifest["files"]:
            if entry["path"] == path.relative_to(self.root).as_posix():
                entry["size"], entry["sha256"] = path.stat().st_size, bundle.sha256(path)
        self.save()
        with self.assertRaisesRegex(ValueError, "source archive hash"):
            bundle.validate(self.root)

    def test_assemble_static_fixture_and_exclude_model(self):
        from argparse import Namespace
        base = Path(self.temp.name)
        source, audit, build = base / "input-source", base / "audit", base / "fresh-build"
        cuda, redist, docs = base / "cuda", base / "redist", base / "license-snapshots"
        for folder in (source, audit, build / "bin", cuda / "bin/x64", redist / "x64/Microsoft.VC145.CRT", redist / "x64/Microsoft.VC145.OpenMP", docs):
            folder.mkdir(parents=True, exist_ok=True)
        ordinary = ("LICENSE", "licenses/LICENSE-jsonhpp", "vendor/cpp-httplib/LICENSE", "vendor/hash/sha256/LICENSE", "vendor/hash/xxhash/LICENSE", "vendor/hash/rotate-bits/LICENSE.md")
        excerpts = {
            "vendor/miniaudio/miniaudio.h": "This software is available as a choice of the following licenses.\nfixture license",
            "vendor/stb/stb_image.h": "This software is available under 2 licenses\nfixture license",
            "vendor/sheredom/subprocess.h": "This is free and unencumbered software\nfixture */",
            "vendor/hash/sha1/sha1.c": "SHA-1 in C\nfixture */",
        }
        for relative in ordinary + tuple(excerpts):
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(excerpts.get(relative, "synthetic license fixture, not production terms"), encoding="utf-8")
        for path in self.root.glob("*"):
            if path.suffix.lower() in (".exe", ".dll"):
                target = cuda / "bin/x64" if path.name.lower() in bundle.CUDA_FILES else redist / "x64/Microsoft.VC145.CRT" if path.name == "msvcp140.dll" else build / "bin"
                shutil.copy2(path, target / path.name)
        input_archive = audit / f"llama.cpp-{bundle.COMMIT}.tar"
        with tarfile.open(input_archive, "w") as archive:
            for name, payload in (("CMakeLists.txt", b"# synthetic source"), ("models/fixture.gguf", b"not actual model data")):
                entry = tarfile.TarInfo(name)
                entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))
        original_hash = bundle.sha256(input_archive)
        inputs = dict(self.manifest["identity"], sourceRoot=str(source), sourceArchiveSha256=original_hash, buildArgv=["cmake", "--build", str(build)])
        (audit / "build-inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
        (audit / "build-result.json").write_text(json.dumps({"exitCode": 0, "target": "llama-server", "syntheticFixture": True}), encoding="utf-8")
        (audit / "build_bundled_llama.ps1").write_text("# synthetic build script fixture", encoding="utf-8")
        names = ("NVIDIA-CUDA-EULA.html", "MSVC-redistribution.html", "MSVC-license.html", "MSVC-Community-2026-license.docx", "CCCL-LICENSE.txt")
        records = []
        for name in names:
            path = docs / name
            path.write_text("synthetic document fixture, not actual license", encoding="utf-8")
            records.append({"outputName": name, "path": name, "sha256": bundle.sha256(path), "url": "https://example.invalid/fixture"})
        (docs / "document-sources.json").write_text(json.dumps(records), encoding="utf-8")
        destination = base / "assembled"
        bundle.assemble(Namespace(source_root=source, audit_root=audit, build_root=build, destination=destination, cuda_root=cuda, msvc_redist=redist, license_root=docs))
        self.assertEqual(bundle.validate(destination)["identity"], self.manifest["identity"])
        provenance = json.loads((destination / "source/provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["sourceArchiveExclusions"][0]["path"], "models/fixture.gguf")
        self.assertEqual(bundle.sha256(input_archive), original_hash)
        with tarfile.open(destination / f"source/llama.cpp-{bundle.COMMIT}-runtime-source.tar") as archive:
            self.assertEqual(archive.getnames(), ["CMakeLists.txt"])

    def test_build_script_gates(self):
        script = (TOOL.parents[1] / "Build-Plugin.ps1").read_text(encoding="utf-8-sig")
        self.assertIn('[string]$LlamaRuntimeBundle = ""', script)
        self.assertIn('if (-not $LlamaRuntimeBundle)', script)
        self.assertIn('--bundle $LlamaRuntimeBundle --copy-to $llamaRuntimeTarget', script)
        self.assertIn('--validate $llamaRuntimeTarget', script)
        self.assertIn('[switch]$LocalCandidate', script)
        self.assertIn('[switch]$SkipArchive', script)
        self.assertIn('RuntimeLock is required', script)
        self.assertIn('VendorRoot is required', script)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -I -B tests/test_bundle_llama_runtime.py <new-test-work-root>")
    TEST_ROOT = Path(sys.argv.pop()).absolute()
    TEST_ROOT.mkdir(parents=True, exist_ok=False)
    unittest.main(verbosity=2)

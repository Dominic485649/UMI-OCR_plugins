"""Offline, fail-closed assembler/validator for the PaddleVL Windows llama runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import tarfile

COMMIT = "74bca3950054eed1795f525cb9cf06bfe796492d"
ARCHITECTURES = ["75", "80", "86", "89", "90", "120a", "121a"]
CUDA_FILES = {"cudart64_13.dll", "cublas64_13.dll", "cublaslt64_13.dll"}
# OS/driver components are prerequisites, never copied from the developer system.
SYSTEM_DLLS = set("""advapi32.dll avrt.dll bcrypt.dll cfgmgr32.dll comdlg32.dll
combase.dll crypt32.dll dbghelp.dll dnsapi.dll dwmapi.dll dxgi.dll gdi32.dll
imm32.dll iphlpapi.dll kernel32.dll kernelbase.dll mf.dll mfplat.dll mfreadwrite.dll
mfuuid.dll mmdevapi.dll mpr.dll msvcrt.dll mswsock.dll ncrypt.dll ntdll.dll
nvcuda.dll ole32.dll oleaut32.dll powrprof.dll propsys.dll psapi.dll rpcrt4.dll
secur32.dll setupapi.dll shcore.dll shell32.dll shlwapi.dll user32.dll userenv.dll
ucrtbase.dll version.dll winhttp.dll wininet.dll winmm.dll wintrust.dll ws2_32.dll
wtsapi32.dll""".split())
LICENSE_FILES = {
    "LICENSES/llama-MIT.txt", "LICENSES/nlohmann-json-MIT.txt",
    "LICENSES/cpp-httplib-MIT.txt", "LICENSES/miniaudio.txt", "LICENSES/stb_image.txt",
    "LICENSES/sheredom-subprocess.txt", "LICENSES/sha1.txt", "LICENSES/sha256.txt",
    "LICENSES/xxhash.txt", "LICENSES/rotate-bits.txt", "LICENSES/NVIDIA-CUDA-EULA.html",
    "LICENSES/MSVC-redistribution.html", "LICENSES/MSVC-license.html",
    "LICENSES/MSVC-Community-2026-license.docx", "LICENSES/CCCL-LICENSE.txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def no_links(path: Path) -> None:
    for item in (path, *path.parents):
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise ValueError(f"Symlink/junction is not allowed: {item}")
        if item.exists() and getattr(item.stat(), "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Reparse point is not allowed: {item}")


def safe_relative(text: str) -> str:
    if not isinstance(text, str) or not text or "\\" in text or ":" in text:
        raise ValueError(f"Invalid manifest path: {text!r}")
    path = PurePosixPath(text)
    if path.is_absolute() or any(p in ("", ".", "..") for p in text.split("/")):
        raise ValueError(f"Unsafe manifest path: {text!r}")
    if any(p.endswith((".", " ")) for p in path.parts):
        raise ValueError(f"Ambiguous Windows path: {text!r}")
    return text


def is_system(name: str) -> bool:
    return name in SYSTEM_DLLS or bool(re.fullmatch(r"(?:api|ext)-ms-win-[a-z0-9-]+\.dll", name))


def pe_imports(path: Path) -> list[str]:
    """Read normal and delay import descriptors, require x64 PE32+ binaries."""
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        def unpack(fmt: str, offset: int):
            if offset < 0 or offset + struct.calcsize(fmt) > len(data):
                raise ValueError(f"Truncated PE: {path}")
            return struct.unpack_from(fmt, data, offset)
        if data[:2] != b"MZ":
            raise ValueError(f"Not a PE file: {path}")
        pe, = unpack("<I", 0x3C)
        if data[pe:pe + 4] != b"PE\0\0":
            raise ValueError(f"Bad PE signature: {path}")
        machine, sections = unpack("<HH", pe + 4)
        optional_size, = unpack("<H", pe + 20)
        optional = pe + 24
        magic, = unpack("<H", optional)
        if machine != 0x8664 or magic != 0x20B or optional_size < 240:
            raise ValueError(f"Expected x64 PE32+: {path}")
        image_base, = unpack("<Q", optional + 24)
        section_offset = optional + optional_size
        ranges = []
        for index in range(sections):
            virtual_size, virtual, raw_size, raw = unpack("<IIII", section_offset + index * 40 + 8)
            ranges.append((virtual, virtual + max(virtual_size, raw_size), raw, raw_size))
        def rva_offset(rva: int) -> int:
            for low, high, raw, raw_size in ranges:
                if low <= rva < high and rva - low < raw_size:
                    return raw + rva - low
            raise ValueError(f"Unmapped PE RVA {rva:x}: {path}")
        def read_name(rva: int) -> str:
            start = rva_offset(rva)
            end = data.find(b"\0", start, min(len(data), start + 256))
            if end < 0:
                raise ValueError(f"Unterminated PE import: {path}")
            name = data[start:end].decode("ascii").lower()
            if not re.fullmatch(r"[a-z0-9_.-]+\.dll", name):
                raise ValueError(f"Unsafe PE import {name!r}: {path}")
            return name
        result = set()
        directories, = unpack("<I", optional + 108)
        for entry, width in ((1, 20), (13, 32)):
            if directories <= entry:
                continue
            rva, size = unpack("<II", optional + 112 + entry * 8)
            if not rva:
                continue
            start = rva_offset(rva)
            for index in range(min(size // width + 1, 4096)):
                values = unpack("<" + "I" * (width // 4), start + width * index)
                if not any(values):
                    break
                name_rva = values[3] if entry == 1 else values[1]
                if entry == 13 and not values[0] & 1:
                    name_rva -= image_base
                result.add(read_name(name_rva))
            else:
                raise ValueError(f"Unbounded PE imports: {path}")
        return sorted(result)


def closure_check(root: Path) -> dict[str, list[str]]:
    binaries = {p.name.lower(): p for p in root.iterdir() if p.suffix.lower() in (".exe", ".dll")}
    if "llama-server.exe" not in binaries:
        raise ValueError("llama-server.exe is missing")
    if set(p.name.lower() for p in root.rglob("*.exe")) != {"llama-server.exe"}:
        raise ValueError("Only the reviewed llama-server executable is allowed")
    if not {"cublas64_13.dll", "cublaslt64_13.dll"} <= binaries.keys():
        raise ValueError("CUDA cuBLAS/cuBLASLt closure is missing")
    graph = {name: pe_imports(path) for name, path in binaries.items()}
    for name, imports in graph.items():
        for dependency in imports:
            if dependency not in binaries and not is_system(dependency):
                raise ValueError(f"Unresolved dependency: {name} -> {dependency}")
    if any(name in SYSTEM_DLLS or name.startswith(("api-ms-", "ext-ms-")) for name in binaries):
        raise ValueError("Windows OS/NVIDIA driver DLL must not be bundled")
    reachable = {"llama-server.exe"}
    pending = list(reachable)
    while pending:
        for dependency in graph[pending.pop()]:
            if dependency in binaries and dependency not in reachable:
                reachable.add(dependency)
                pending.append(dependency)
    if reachable != binaries.keys():
        raise ValueError(f"Unneeded binaries in bundle: {sorted(binaries.keys() - reachable)}")
    return graph


def validate(root: Path) -> dict:
    root = root.absolute()
    no_links(root)
    manifest_path = root / "runtime-manifest.json"
    no_links(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schemaVersion") != 1:
        raise ValueError("Unsupported runtime manifest schema")
    identity = manifest.get("identity", {})
    if identity.get("commit") != COMMIT or identity.get("cudaToolkit") != "13.4.92":
        raise ValueError("Runtime commit/toolkit does not match the reviewed input")
    if identity.get("architectures") != ARCHITECTURES:
        raise ValueError("Reviewed CUDA architecture set is missing or changed")
    options = identity.get("buildOptions", {})
    if options.get("GGML_NATIVE") != "OFF" or options.get("GGML_CUDA") != "ON":
        raise ValueError("Runtime must use CUDA with GGML_NATIVE=OFF")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest files must be a nonempty list")
    listed = set()
    folded = set()
    for entry in entries:
        relative = safe_relative(entry["path"])
        if relative == "runtime-manifest.json" or relative.casefold() in folded:
            raise ValueError(f"Duplicate/self manifest entry: {relative}")
        listed.add(relative)
        folded.add(relative.casefold())
        path = root / relative
        no_links(path)
        if not path.is_file() or type(entry.get("size")) is not int or path.stat().st_size != entry["size"]:
            raise ValueError(f"Runtime file size/missing: {relative}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")) or sha256(path) != entry["sha256"]:
            raise ValueError(f"Runtime file hash mismatch: {relative}")
    actual = set()
    for path in root.rglob("*"):
        no_links(path)
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if path.suffix.lower() in (".gguf", ".mmproj"):
                raise ValueError("Models are not allowed in this runtime")
            if relative != "runtime-manifest.json":
                actual.add(relative)
    if listed != actual:
        raise ValueError(f"Manifest coverage mismatch: extra={sorted(actual-listed)}, missing={sorted(listed-actual)}")
    required = LICENSE_FILES | {"source/provenance.json", f"source/llama.cpp-{COMMIT}-runtime-source.tar", "source/build-inputs.json", "source/build_bundled_llama.ps1"}
    if not required <= listed:
        raise ValueError(f"Runtime licenses/source missing: {sorted(required-listed)}")
    for relative in listed:
        if Path(relative).suffix.lower() in (".exe", ".dll") and len(PurePosixPath(relative).parts) != 1:
            raise ValueError(f"Runtime binaries must be adjacent to llama-server.exe: {relative}")
    inputs = json.loads((root / "source/build-inputs.json").read_text(encoding="utf-8-sig"))
    provenance = json.loads((root / "source/provenance.json").read_text(encoding="utf-8-sig"))
    source_hash = sha256(root / f"source/llama.cpp-{COMMIT}-runtime-source.tar")
    if any(inputs.get(key) != identity[key] for key in ("commit", "cudaToolkit", "architectures", "buildOptions")):
        raise ValueError("Build/source identity differs from manifest")
    if source_hash != provenance.get("sourceArchiveSha256") or inputs.get("sourceArchiveSha256") != provenance.get("buildInputArchiveSha256"):
        raise ValueError("Corresponding source archive hash differs from build provenance")
    with tarfile.open(root / f"source/llama.cpp-{COMMIT}-runtime-source.tar", "r") as archive:
        for member in archive:
            safe_relative(member.name.rstrip("/"))
            if not (member.isfile() or member.isdir()) or member.name.lower().endswith((".gguf", ".mmproj")):
                raise ValueError(f"Forbidden model/link in corresponding source: {member.name}")
    closure_check(root)
    return manifest


def copy_bundle(root: Path, destination: Path) -> None:
    validate(root)
    destination = destination.absolute()
    no_links(destination)
    if destination.exists():
        raise ValueError(f"Bundle destination must be new (no overwrite): {destination}")
    if root.absolute() == destination or root.absolute() in destination.parents:
        raise ValueError("Destination cannot be inside input runtime")
    destination.mkdir(parents=True, exist_ok=False)
    for path in root.rglob("*"):
        target = destination / path.relative_to(root)
        if path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    validate(destination)


def assemble(args: argparse.Namespace) -> None:
    source, audit, build, destination = map(Path, (args.source_root, args.audit_root, args.build_root, args.destination))
    no_links(destination.absolute())
    if destination.exists():
        raise ValueError(f"Use a new runtime directory: {destination}")
    inputs = json.loads((audit / "build-inputs.json").read_text(encoding="utf-8-sig"))
    if inputs["commit"] != COMMIT or inputs["cudaToolkit"] != "13.4.92":
        raise ValueError("Unreviewed build input")
    result = json.loads((audit / "build-result.json").read_text(encoding="utf-8-sig"))
    if result.get("exitCode") != 0 or result.get("target") != "llama-server":
        raise ValueError("Successful fresh build record is required")
    argv = inputs.get("buildArgv", [])
    if len(argv) < 3 or argv[1] != "--build" or Path(argv[2]).resolve() != build.resolve():
        raise ValueError("Build output root differs from the recorded fresh build argv")
    if Path(inputs["sourceRoot"]).resolve() != source.resolve():
        raise ValueError("License source root differs from the fixed build input")
    search = [(build / "bin", "llama.cpp fresh build"),
              (Path(args.cuda_root) / "bin" / "x64", "CUDA Toolkit x64 redistributable"),
              (Path(args.msvc_redist) / "x64" / "Microsoft.VC145.CRT", "MSVC CRT redistributable"),
              (Path(args.msvc_redist) / "x64" / "Microsoft.VC145.OpenMP", "MSVC OpenMP redistributable")]
    candidates = {}
    for directory, origin in search:
        for path in directory.iterdir():
            if path.suffix.lower() not in (".exe", ".dll"):
                continue
            name = path.name.lower()
            if origin.startswith("CUDA") and name not in CUDA_FILES:
                continue
            if name in candidates:
                raise ValueError(f"Ambiguous runtime dependency: {name}")
            candidates[name] = (path, origin)
    selected = {}
    pending = ["llama-server.exe"]
    graph = {}
    while pending:
        name = pending.pop()
        if name in selected or is_system(name):
            continue
        if name not in candidates:
            raise ValueError(f"Missing dependency in reviewed roots: {name}")
        path, origin = candidates[name]
        imports = pe_imports(path)
        selected[name] = {"source": str(path), "origin": origin, "sha256": sha256(path)}
        graph[name] = imports
        pending.extend(imports)
    destination.mkdir(parents=True, exist_ok=False)
    for name, entry in selected.items():
        shutil.copy2(entry["source"], destination / Path(entry["source"]).name)
    license_root = destination / "LICENSES"
    license_root.mkdir()
    copied_licenses = {
        "llama-MIT.txt": "LICENSE", "nlohmann-json-MIT.txt": "licenses/LICENSE-jsonhpp",
        "cpp-httplib-MIT.txt": "vendor/cpp-httplib/LICENSE", "sha256.txt": "vendor/hash/sha256/LICENSE",
        "xxhash.txt": "vendor/hash/xxhash/LICENSE", "rotate-bits.txt": "vendor/hash/rotate-bits/LICENSE.md",
    }
    for target, relative in copied_licenses.items():
        shutil.copy2(source / relative, license_root / target)
    excerpts = {
        "miniaudio.txt": ("vendor/miniaudio/miniaudio.h", "This software is available as a choice of the following licenses.", None),
        "stb_image.txt": ("vendor/stb/stb_image.h", "This software is available under 2 licenses", None),
        "sheredom-subprocess.txt": ("vendor/sheredom/subprocess.h", "This is free and unencumbered software", "*/"),
        "sha1.txt": ("vendor/hash/sha1/sha1.c", "SHA-1 in C", "*/"),
    }
    for target, (relative, start, end) in excerpts.items():
        text = (source / relative).read_text(encoding="utf-8")
        offset = text.index(start)
        text = text[offset:]
        if end:
            text = text[:text.index(end)]
        (license_root / target).write_text(f"Source: {relative} at {COMMIT}\n\n{text}", encoding="utf-8")
    license_sources = json.loads((Path(args.license_root) / "document-sources.json").read_text(encoding="utf-8-sig"))
    for document in license_sources:
        name = safe_relative(document["outputName"])
        path = Path(args.license_root) / safe_relative(document["path"])
        if sha256(path) != document["sha256"]:
            raise ValueError(f"License document hash mismatch: {name}")
        shutil.copy2(path, license_root / name)
    source_target = destination / "source"
    source_target.mkdir()
    input_archive = audit / f"llama.cpp-{COMMIT}.tar"
    if sha256(input_archive) != inputs["sourceArchiveSha256"]:
        raise ValueError("Original build-input source snapshot has changed")
    runtime_archive = source_target / f"llama.cpp-{COMMIT}-runtime-source.tar"
    exclusions = []
    with tarfile.open(input_archive, "r") as original, tarfile.open(runtime_archive, "w", format=tarfile.PAX_FORMAT) as filtered:
        for member in original:
            safe_relative(member.name.rstrip("/"))
            if member.name.lower().endswith((".gguf", ".mmproj")):
                stream = original.extractfile(member)
                digest = hashlib.sha256()
                if stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                exclusions.append({"path": member.name, "size": member.size, "sha256": digest.hexdigest(), "reason": "Upstream vocabulary/model regression fixture, not needed to build llama-server; no model assets in product"})
                continue
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unexpected source archive member type: {member.name}")
            filtered.addfile(member, original.extractfile(member) if member.isfile() else None)
    for name in ("build-inputs.json", "build-result.json", "build_bundled_llama.ps1"):
        shutil.copy2(audit / name, source_target / name)
    shutil.copy2(Path(__file__), source_target / "bundle_llama_runtime.py")
    provenance = {
        "schemaVersion": 1, "upstream": "https://github.com/ggml-org/llama.cpp", "commit": COMMIT,
        "sourceArchiveSha256": sha256(runtime_archive), "buildInputArchiveSha256": inputs["sourceArchiveSha256"],
        "sourceArchiveScope": "Exact fixed-commit tracked file bytes, excluding upstream .gguf/.mmproj vocabulary/model fixtures; original full git archive retained unchanged in build audit.",
        "sourceArchiveExclusions": exclusions, "binaryOrigins": selected,
        "imports": graph,
        "cudaRuntimeLinkage": "Only DLLs reachable through PE normal/delay imports are bundled. The upstream nvcc link may embed cudart_static; cudart DLL is not added unless imported. Driver components remain external.",
        "externalPrerequisites": sorted({name for imports in graph.values() for name in imports if is_system(name)}),
        "licenseSources": license_sources,
        "driver": {"source": "https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html",
                   "cuda13CompatibilityFloorBranch": "R580", "cuda13_4CorrespondingBranch": "R615",
                   "policy": "R615 or later recommended; new CUDA 13.4 features/platforms require R615 or later. R580 is a CUDA 13.x minor-compatibility floor, not proof this seven-architecture/PTX runtime works there.",
                   "notBundled": True},
        "licensingScope": "Local candidate only. CUDA/MSVC are proprietary and remain subject to their own terms; valid developer licensing and downstream compliance are not presumed. No toolkit, driver or model is redistributed here.",
        "validationLimit": "Build/packaging evidence does not imply inference testing. Device discovery results recorded separately; other GPUs and old driver versions not physically tested.",
    }
    (source_target / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = [{"path": p.relative_to(destination).as_posix(), "size": p.stat().st_size, "sha256": sha256(p)} for p in sorted(destination.rglob("*")) if p.is_file()]
    manifest = {"schemaVersion": 1, "identity": {key: inputs[key] for key in ("commit", "cudaToolkit", "architectures", "buildOptions")}, "files": files}
    (destination / "runtime-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validate(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate", type=Path)
    modes.add_argument("--bundle", type=Path, help="Validate and copy an existing bundle (requires --copy-to)")
    modes.add_argument("--assemble", action="store_true")
    parser.add_argument("--copy-to", type=Path)
    for name in ("source-root", "audit-root", "build-root", "cuda-root", "msvc-redist", "license-root", "destination"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    try:
        if args.validate:
            result = validate(args.validate)
            print(json.dumps({"valid": True, "identity": result["identity"], "fileCount": len(result["files"])}, ensure_ascii=False))
        elif args.bundle:
            if not args.copy_to:
                parser.error("--bundle requires --copy-to")
            copy_bundle(args.bundle, args.copy_to)
            print(f"Verified runtime copied to {args.copy_to}")
        else:
            for name in ("source_root", "audit_root", "build_root", "cuda_root", "msvc_redist", "license_root", "destination"):
                if not getattr(args, name):
                    parser.error("--assemble requires --" + name.replace("_", "-"))
            assemble(args)
            print(f"Verified runtime assembled at {args.destination}")
    except (ValueError, OSError, KeyError, TypeError, struct.error, tarfile.TarError) as error:
        parser.exit(1, f"llama runtime rejected: {error}\n")


if __name__ == "__main__":
    main()

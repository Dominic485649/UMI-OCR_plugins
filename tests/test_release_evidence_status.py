from contextlib import ExitStack
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "build" / "tools" / "verify_release_evidence.py"
    spec = importlib.util.spec_from_file_location("release_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def allowed(conditions):
        # These unit cases isolate component statuses; publication additionally
        # requires separately verified source and current-payload gates.
        return module.public_release_allowed({
            "source_delivery": {"status": "PASS", "verified": True},
            "current_source_e2e": {"status": "PASS", "verified": True, "publication_authorized": True},
            **conditions,
        })

    assert allowed({"included": {"status": "PASS"}})
    assert allowed({"absent": {"status": "NOT_INCLUDED"}})
    assert not allowed({"cudnn": {"status": "PASS_WITH_REQUIREMENTS"}})
    assert not allowed({"model": {"status": "BLOCK"}})
    staging = ROOT / "staging"
    package = staging / "UmiOCR-data"
    assert module.runtime_lock_target(staging, package, package / "native-dlls.lock.json", "UmiOCR-data/runtime/python313.dll") == staging / "UmiOCR-data/runtime/python313.dll"
    assert module.runtime_lock_target(staging, package, package / "runtime.lock.json", "vendor/paddle/base/paddle.pyd") == package / "vendor/paddle/base/paddle.pyd"
    assert module.runtime_lock_target(staging, package, package / "runtime.lock.json", "../escape.dll") is None
    assert tuple(module.CPU_MODELS) == (
        "PP-LCNet_x1_0_doc_ori",
        "UVDoc",
        "PP-LCNet_x1_0_textline_ori",
        "PP-OCRv6_medium_det",
        "PP-OCRv6_medium_rec",
    )
    with tempfile.TemporaryDirectory() as temporary:
        vendor = Path(temporary)
        marker = vendor / "shapely-2.1.2.dist-info" / "DELVEWHEEL"
        marker.parent.mkdir()
        marker.write_text("Arguments: ['D:\\\\a\\\\_temp\\\\geos-3.13.1\\\\bin']", encoding="utf-8")
        assert module.cpu_geos_version(vendor) == "3.13.1"
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        vendor = staging / "vendor"
        info = vendor / "colorama-0.4.6.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            "Name: colorama\nVersion: 0.4.6\nHome-page: https://github.com/tartley/colorama\n"
            "Classifier: License :: OSI Approved :: BSD License\n",
            encoding="utf-8",
        )
        rows, _, _ = module.package_rows(vendor, staging)
        assert rows[0]["upstream_revision"] == "0.4.6"
        assert rows[0]["evidence_snapshot"]["artifact_path"].endswith("colorama-0.4.6.dist-info/METADATA")
        assert module.CPU_BINARY_LICENSES["colorama"] == "BSD-3-Clause"
        assert callable(module.verify_cpu_pdfium_bundle)
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        plugin = staging / "UmiOCR-data" / "plugins" / "win32_PaddleOCRv6_CUDA"
        plugin.mkdir(parents=True)
        runtime_lock = plugin / "runtime.lock.json"
        runtime_lock.write_text('{"nativeBinaryCount": 1}', encoding="utf-8")
        models = {
            "models": [
                {"model_name": "PP-LCNet_x1_0_doc_ori_infer"},
                {"model_name": "UVDoc_grid_infer"},
                {"model_name": "PP-LCNet_x1_0_textline_ori_infer"},
                {"model_name": "PP-OCRv6_medium_det_infer"},
                {"model_name": "PP-OCRv6_medium_rec_infer"},
            ]
        }
        record = {
            "test_type": "real_gpu_ocr_workflows_e2e",
            "status": "PASS",
            "device": "gpu:0",
            "architecture": {
                "type": "separate-worker-stdio-ipc",
                "host_loaded_proprietary_modules": [],
                "ipc_payload": "JSON-lines path/base64 -> JSON OCR result",
            },
            "plugin_runtime_lock_sha256": module.sha256(runtime_lock),
            "model_initialization": [
                "PP-LCNet_x1_0_doc_ori",
                "UVDoc",
                "PP-LCNet_x1_0_textline_ori",
                "PP-OCRv6_medium_det",
                "PP-OCRv6_medium_rec",
            ],
            "ffmpeg_backend_dll_count": 0,
            "cusolvermg_dll_count": 0,
            "exercised_paths": ["batch OCR", "document PDF OCR", "CUDA-only failure paths"],
        }
        evidence = plugin / "gpu-e2e-evidence.json"
        source = staging / "source" / "UmiOCR-data" / "plugins" / plugin.name / evidence.name
        source.parent.mkdir(parents=True)
        evidence.write_text(json.dumps(record), encoding="utf-8")
        source.write_text(json.dumps(record), encoding="utf-8")
        assert module.verify_gpu_e2e(staging, plugin, models)["status"] == "PASS"
        source.unlink()
        (staging / "SOURCE-CODE.md").write_text(
            "# local candidate\n"
            "- Local Git commit: 9deac935099d742d627ecaf15af7f2b2f7b708a1\n"
            "- Local tag: v2.8.0\n"
            "- Upstream repository (reference only): https://github.com/Dominic485649/Umi-OCR\n"
            "- Public source URL: **PENDING**\n",
            encoding="utf-8",
        )
        assert module.verify_gpu_e2e(staging, plugin, models, ROOT)["status"] == "PASS"
        record["device"] = "cpu"
        evidence.write_text(json.dumps(record), encoding="utf-8")
        assert module.verify_gpu_e2e(staging, plugin, models, ROOT)["status"] == "BLOCK"
        # New release evidence must bind adapter source, not just the native
        # runtime lock.  A changed Python adapter invalidates old inference.
        manifest = plugin / "umi-plugin.json"
        manifest.write_text('{"version":"3.0.0"}', encoding="utf-8")
        adapter = plugin / "api.py"
        adapter.write_text("# tested adapter\n", encoding="utf-8")
        record["device"] = "gpu:0"
        evidence.write_text(json.dumps(record), encoding="utf-8")
        assert module.verify_gpu_e2e(staging, plugin, models, ROOT)["status"] == "BLOCK"
        record["plugin_source_sha256"] = {"api.py": module.sha256(adapter)}
        record["workflow_test"] = {"SHA256": module.sha256(ROOT / "tests/test_ocr_workflows.py")}
        evidence.write_text(json.dumps(record), encoding="utf-8")
        assert module.verify_gpu_e2e(staging, plugin, models, ROOT)["status"] == "PASS"
        adapter.write_text("# changed after the test\n", encoding="utf-8")
        assert module.verify_gpu_e2e(staging, plugin, models, ROOT)["status"] == "BLOCK"
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        repo = base / "repo"
        staging = base / "public-staging"
        sibling = base / "Umi-OCR-Plugin" / "win32_PaddleOCRv6_CPU"
        sibling.mkdir(parents=True)
        (repo / "UmiOCR-data/plugins/win32_PaddleOCRVL_1_6").mkdir(parents=True)
        (staging / "UmiOCR-data/plugins").mkdir(parents=True)
        result = module.verify_cpu_and_document(repo, artifact_staging=staging)
        assert result["cpu_ocr"]["status"] == "NOT_INCLUDED"
        assert result["document_models"]["status"] == "NOT_INCLUDED"
        # A development source tree must not influence actual archive scope.
        assert allowed({"cpu": result["cpu_ocr"], "document": result["document_models"]})
        (staging / "UmiOCR-data/plugins/win32_PaddleOCRVL_1_6").mkdir()
        result = module.verify_cpu_and_document(repo, artifact_staging=staging)
        assert result["document_models"]["status"] == "BLOCK"
        (staging / "UmiOCR-data/plugins" / module.CPU_PLUGIN_ID).mkdir()
        audited = {"cpu_ocr": {"status": "BLOCK", "blockers": ["forced failure"]},
                   "runtime": {}, "models": {}, "opencv": {}, "ffmpeg": {},
                   "mpl": {}, "copyleft": {}, "e2e": {}, "source": {}}
        with patch.object(module, "verify_cpu_artifact", return_value=audited) as verify_cpu:
            result = module.verify_cpu_and_document(repo, artifact_staging=staging)
            verify_cpu.assert_called_once_with(staging, repo, [])
        assert result["cpu_ocr"]["status"] == "BLOCK"
        assert not allowed({"cpu": result["cpu_ocr"]})
    print("Release evidence status gate regression passed.")


class StartupEvidenceStatusTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "tests/test_release_gate_inputs.py"
        spec = importlib.util.spec_from_file_location("status_startup_fixtures", path)
        fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        self.context = fixtures.StartupGateTests("test_complete_scoped_acceptance_keeps_ocr_not_run_and_public_paths_private")
        self.context.setUp()
        self.addCleanup(self.context.doCleanups)
        self.fixture = self.context.fixture
        self.current = self.context.verify()
        spec = importlib.util.spec_from_file_location("startup_status_verifier", ROOT / "build/tools/verify_release_evidence.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.conditions = {"source_delivery": {"status": "PASS", "verified": True}, "current_source_e2e": self.current,
                           "gpu_worker_startup": self.current["gpu_worker_startup"],
                           "gpu_ocr_e2e": {"status": "NOT_RUN", "validation_scope": "cuda-startup-hotfix", "binding": self.fixture.binding},
                           "cudnn": {"status": "PASS"}, "runtime_vendor": {"status": "PASS"}}

    def test_only_explicit_current_scope_permits_named_ocr_not_run(self):
        self.assertTrue(self.module.public_release_allowed(self.conditions))
        for field, value in (("validation_scope", "full"), ("publication_authorized", False), ("verified", False), ("release_version", "3.0.3"), ("artifact_kind", "base")):
            changed = copy.deepcopy(self.conditions)
            changed["current_source_e2e"][field] = value
            self.assertFalse(self.module.public_release_allowed(changed), field)
        for status in ("PASS", "BLOCK", "NOT_INCLUDED"):
            changed = copy.deepcopy(self.conditions)
            changed["gpu_ocr_e2e"]["status"] = status
            self.assertFalse(self.module.public_release_allowed(changed), status)

    def test_other_not_run_block_or_requirements_are_never_waived(self):
        for name in ("cudnn", "runtime_vendor", "gpu_worker_startup", "source_delivery"):
            for status in ("NOT_RUN", "BLOCK", "PASS_WITH_REQUIREMENTS"):
                changed = copy.deepcopy(self.conditions)
                changed[name]["status"] = status
                self.assertFalse(self.module.public_release_allowed(changed), (name, status))
        changed = copy.deepcopy(self.conditions)
        changed["unknown_extra"] = {"status": "NOT_RUN"}
        self.assertFalse(self.module.public_release_allowed(changed))

    def test_raw_startup_pass_does_not_become_current_ocr_pass(self):
        result = self.module.verify_gpu_e2e(self.fixture.staging, self.fixture.plugin, {}, startup_evidence=self.current["gpu_worker_startup"])
        self.assertEqual(result["status"], "NOT_RUN")
        result = self.module.verify_gpu_e2e(self.fixture.staging, self.fixture.plugin, {}, startup_evidence={"status": "PASS"})
        self.assertEqual(result["status"], "BLOCK")
        self.assertFalse(self.module._material_functionality(self.fixture.staging, {"status": "PASS"})[0])
        self.assertFalse(self.module._material_functionality(self.fixture.staging)[0])  # stale 3.0.3 source hashes

    def test_cudnn_historical_functionality_is_scoped_and_does_not_erase_other_blocks(self):
        vendor = self.fixture.plugin / "vendor"
        info = vendor / "nvidia_cudnn_cu12-9.0.dist-info"
        self.context.write(vendor / "nvidia/cudnn/bin/cudnn64_9.dll", b"SYNTHETIC cuDNN binary - not executable")
        self.context.write(info / "licenses/License.txt", "NVIDIA proprietary license fixture")
        snapshot = self.fixture.metadata / self.module.CUDNN_SLA_SNAPSHOT
        self.context.write(snapshot, "NVIDIA runtime files .so and .dll material additional functionality only be accessed by your application")
        self.context.write(self.fixture.metadata / "LICENSES/THIRD_PARTY_NOTICES.md", "NVIDIA proprietary notices retained")
        startup_result = copy.deepcopy(self.current["gpu_worker_startup"])
        # Isolate the consuming component check; the producer's official tar
        # hashes are separately enforced by synthetic matrix tests.
        startup_result["baselines"]["historical_functionality"]["official_archive_sha256"] = "be8bd1f8d20181500370cc5a64cb867514478a45bc0f897c63a3ebde96ce4485"
        with patch.object(self.module, "CUDNN_SLA_SNAPSHOT_SHA256", self.module.sha256(snapshot)):
            result = self.module.verify_cudnn(self.fixture.staging, vendor, startup_result)
            self.assertEqual(result["status"], "PASS", result)
            self.assertTrue(result["distribution_requirements_satisfied"]["material_additional_functionality"])
            self.context.write(self.fixture.metadata / "LICENSES/THIRD_PARTY_NOTICES.md", "notice lacks required attribution")
            result = self.module.verify_cudnn(self.fixture.staging, vendor, startup_result)
            self.assertEqual(result["status"], "BLOCK")
            self.assertFalse(result["distribution_requirements_satisfied"]["nvidia_proprietary_notices_intact"])
        startup_result["baselines"]["unchanged_ocr"]["verified"] = False
        self.assertFalse(self.module._material_functionality(self.fixture.staging, startup_result)[0])

    def test_native_entry_only_uses_verified_exact_launcher_and_keeps_all_process_checks(self):
        self.context.write(self.fixture.plugin / "api.py", "# subprocess.Popen json.dumps JSON-lines\n")
        self.context.write(self.fixture.plugin / "worker.py", "# import paddle\n# from paddleocr import PaddleOCR\n# import cv2\n# run_path run_bytes json.loads\n")
        result = self.module.verify_agpl(self.fixture.staging, self.fixture.root, self.fixture.plugin, self.current["gpu_worker_startup"])
        self.assertTrue(result["integration_architecture"]["checks"]["manifest_process_boundary"])
        self.assertEqual(result["status"], "PASS", result)
        for evidence in (None, {"status": "PASS"}):
            result = self.module.verify_agpl(self.fixture.staging, self.fixture.root, self.fixture.plugin, evidence)
            self.assertFalse(result["integration_architecture"]["checks"]["manifest_process_boundary"])
            self.assertEqual(result["status"], "BLOCK")
        self.context.write(self.fixture.plugin / "api.py", "# subprocess.Popen json.dumps JSON-lines\nimport paddle\n")
        result = self.module.verify_agpl(self.fixture.staging, self.fixture.root, self.fixture.plugin, self.current["gpu_worker_startup"])
        self.assertEqual(result["status"], "BLOCK")
        self.assertFalse(result["integration_architecture"]["checks"]["host_has_no_proprietary_runtime_imports"])

    def test_startup_report_can_pass_diagnostics_before_actual_publication_consent(self):
        path = self.fixture.audit / "not-approved.json"
        self.context.write(path, {"schema_version": 1, "artifact_kind": "gpu", "release_version": "3.0.4", "publication_authorized": False,
                                  "current_source_e2e": self.context.record})
        with self.fixture.patches(), patch.object(self.module.release_gate_inputs, "startup_evidence_module", return_value=self.context.startup):
            result = self.module.requested_startup_evidence(path, self.fixture.staging, "gpu")
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["public_release_allowed"])
        self.conditions["current_source_e2e"]["publication_authorized"] = False
        self.assertFalse(self.module.public_release_allowed(self.conditions))

    def test_complete_verifier_writes_only_scoped_derived_reports_and_preserves_release_blocks(self):
        self.context.write(self.fixture.root / "build/sources-windows-cu126.lock.json", {})
        path = self.fixture.audit / "current-gates.json"
        document = {"schema_version": 1, "artifact_kind": "gpu", "release_version": "3.0.4", "publication_authorized": True,
                    "source_delivery": {"status": "PASS", "repositories": [{"role": "cuda", "url": self.module.release_gate_inputs.PUBLIC_REPOSITORIES["cuda"],
                        "tag": "v3.0.4", "commit": "a" * 40, "checkout": str(self.fixture.root)}]}, "current_source_e2e": self.context.record}
        self.context.write(path, document)
        source_before = self.module.release_gate_inputs._files(self.fixture.metadata / "source")
        with ExitStack() as stack:
            stack.enter_context(self.fixture.patches())
            stack.enter_context(patch.object(self.module.release_gate_inputs, "startup_evidence_module", return_value=self.context.startup))
            stack.enter_context(patch.object(self.module.release_gate_inputs, "public_tag", return_value={"refs/tags/v3.0.4": "a" * 40}))
            stack.enter_context(patch.object(self.module.release_gate_inputs, "git_source_matches"))
            stack.enter_context(patch.object(self.module, "ensure_apache_license", return_value="LICENSES/Apache-2.0.txt"))
            for name in ("verify_runtime_vendor", "verify_models", "verify_nvidia", "verify_cudnn", "verify_opencv", "verify_mpl", "verify_agpl"):
                stack.enter_context(patch.object(self.module, name, return_value={"status": "PASS"}))
            stack.enter_context(patch.object(self.module, "verify_ffmpeg", return_value={"status": "NOT_INCLUDED"}))
            report = self.module.verify(self.fixture.staging, self.fixture.root, "gpu", None, release_gates=path)
            self.assertTrue(report["public_prerelease_allowed"], report)
            self.assertEqual(report["gates"]["gpu_ocr_e2e"], "NOT_RUN")
            self.assertEqual(report["gates"]["gpu_worker_startup"], "PASS")
            self.assertEqual(report["gates"]["current_source_e2e"], "ACCEPTED_WITH_LIMITATIONS")
            self.assertTrue((self.fixture.metadata / "LICENSES/gpu-startup-evidence.json").is_file())
            self.assertFalse((self.fixture.staging / "LICENSES/gpu-startup-evidence.json").exists())
            document["publication_authorized"] = False
            self.context.write(path, document)
            report = self.module.verify(self.fixture.staging, self.fixture.root, "gpu", None, release_gates=path)
            self.assertEqual(report["gates"]["gpu_worker_startup"], "PASS")
            self.assertEqual(report["gates"]["gpu_ocr_e2e"], "NOT_RUN")
            self.assertEqual(report["gates"]["current_source_e2e"], "BLOCK")
            self.assertFalse(report["public_prerelease_allowed"])
            self.assertTrue(report["local_candidate"])
        self.assertEqual(self.module.release_gate_inputs._files(self.fixture.metadata / "source"), source_before)

    def test_full_mode_cannot_use_unvalidated_startup_to_skip_ocr(self):
        changed = copy.deepcopy(self.conditions)
        changed["current_source_e2e"].pop("validation_scope")
        self.assertFalse(self.module.public_release_allowed(changed))
        changed = copy.deepcopy(self.conditions)
        changed["gpu_worker_startup"] = {"status": "PASS", "verified": True}
        self.assertFalse(self.module.public_release_allowed(changed))
        changed = copy.deepcopy(self.conditions)
        changed["current_source_e2e"]["user_acceptance"]["release_notes_approved"] = False
        self.assertFalse(self.module.public_release_allowed(changed))


if __name__ == "__main__":
    main()
    unittest.main(verbosity=2)

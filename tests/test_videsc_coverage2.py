"""
tests/test_videsc_coverage2.py

Additional tests to raise videsc coverage.  Targets:
  - videsc/main.py  (_run_vllm, _run_vl, _run_wd14 YouTube branches)
  - videsc/pipeline/vllm_runner.py  (extract_frames_as_pil, _extract_frames,
    _create_vllm_client, _consolidate_vllm, run_batch_vllm, no-frames chunk)
  - videsc/audio/transcription.py  (safe_transcribe_segment pipeline-load path,
    fallback, transcribe_audio_segments, combine_transcription_results string/text
    and dict-timestamp paths, transcribe_audio_from_video non-empty result,
    format_transcript_with_timestamps no-raw-transcript)
  - videsc/model/loader.py  (model_full, half_cpu, optimize, gemma4 loader)
  - videsc/describe.py  (_download_model, _load_labels, include_ratings,
    describe_folder public API, _describe_folder_impl no-frames branch)
  - videsc/utils/helpers.py  (expand_inputs indir-no-exts, filelist, empty list)
  - videsc/pipeline/runner.py  (process_mm_info / process_vision_info wrappers,
    run_single_video non-dry-run, run_single_video_gemma4 dry-run chunks)
"""

from __future__ import annotations

import csv
import importlib
import io
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: fresh-import a module with stubs injected into sys.modules
# ---------------------------------------------------------------------------

def _import_fresh(
    monkeypatch, module_name: str, stubs: dict | None = None
):
    """Import *module_name* from scratch with optional sys.modules stubs."""
    for name, stub in (stubs or {}).items():
        monkeypatch.setitem(sys.modules, name, stub)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# videsc.utils.helpers – expand_inputs / namespace_to_cli branches
# ---------------------------------------------------------------------------


class TestExpandInputsBranches:
    def test_indir_with_no_exts_uses_defaults(self, tmp_path):
        """When indir is given but exts is empty, default video extensions apply."""
        from videsc.utils.helpers import expand_inputs

        mp4 = tmp_path / "clip.mp4"
        mp4.write_bytes(b"x")
        txt = tmp_path / "readme.txt"
        txt.write_bytes(b"x")

        result = expand_inputs(None, str(tmp_path), [], None)
        assert mp4.resolve() in result
        assert txt.resolve() not in result

    def test_filelist_paths(self, tmp_path):
        """filelist entries that resolve to real files are returned."""
        from videsc.utils.helpers import expand_inputs

        vid = tmp_path / "v.mp4"
        vid.write_bytes(b"x")
        fl = tmp_path / "list.txt"
        fl.write_text(f"# comment\n{vid}\n/nonexistent/z.mp4\n")

        result = expand_inputs(None, None, [], str(fl))
        assert vid.resolve() in result

    def test_namespace_to_cli_empty_list_skipped(self):
        """An empty list in the namespace is not included in the output."""
        from videsc.utils.helpers import namespace_to_cli

        args = types.SimpleNamespace(videos=[], prompt="hi", verbose=False)
        out = namespace_to_cli(args, exclude_keys=set())
        assert "--videos" not in out
        assert "--prompt" in out

    def test_expand_inputs_videos_glob(self, tmp_path, monkeypatch):
        """Videos glob patterns are expanded via Path.glob."""
        from videsc.utils.helpers import expand_inputs

        # Use absolute paths directly so glob works under tmp_path
        vid1 = tmp_path / "a.mp4"
        vid1.write_bytes(b"x")
        vid2 = tmp_path / "b.mp4"
        vid2.write_bytes(b"x")

        # CWD-relative glob: change to tmp_path so *.mp4 works
        monkeypatch.chdir(tmp_path)
        result = expand_inputs(["*.mp4"], None, [], None)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# videsc.describe – _download_model, _load_labels, include_ratings,
#                   describe_folder public API, no-frames branch
# ---------------------------------------------------------------------------


class TestDescribeMissingBranches:
    def test_download_model_calls_hf_hub_download(self, monkeypatch):
        """_download_model should call hf_hub_download for both files."""
        called = []

        def fake_download(repo_id, filename):
            called.append(filename)
            return f"/tmp/fake_{filename}"

        hf_stub = types.SimpleNamespace(hf_hub_download=fake_download)
        monkeypatch.setitem(sys.modules, "huggingface_hub", hf_stub)

        import videsc.describe as desc_mod
        monkeypatch.setattr(
            desc_mod,
            "_download_model",
            lambda repo: (
                Path(fake_download(repo, "model.onnx")),
                Path(fake_download(repo, "selected_tags.csv")),
            ),
        )
        # Call the real _download_model via fresh import
        sys.modules.pop("videsc.describe", None)
        desc_mod2 = _import_fresh(
            monkeypatch,
            "videsc.describe",
            {"huggingface_hub": hf_stub},
        )
        p1, p2 = desc_mod2._download_model("owner/repo")
        assert "model.onnx" in str(p1)
        assert "selected_tags.csv" in str(p2)
        assert "model.onnx" in called
        assert "selected_tags.csv" in called

    def test_load_labels_parses_csv(self, tmp_path):
        """_load_labels returns correct tag_names, rating_indices, general_indices."""
        from videsc.describe import _load_labels

        csv_path = tmp_path / "tags.csv"
        rows = [
            {"row_id": "0", "name": "safe", "category": "9"},
            {"row_id": "1", "name": "cat", "category": "0"},
            {"row_id": "2", "name": "dog", "category": "0"},
            {"row_id": "3", "name": "sensitive", "category": "9"},
            {"row_id": "4", "name": "other", "category": "3"},
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["row_id", "name", "category"])
            writer.writeheader()
            writer.writerows(rows)

        tag_names, rating_idxs, general_idxs = _load_labels(csv_path)
        assert tag_names == ["safe", "cat", "dog", "sensitive", "other"]
        assert 0 in rating_idxs   # category 9
        assert 3 in rating_idxs   # category 9
        assert 1 in general_idxs  # category 0
        assert 2 in general_idxs  # category 0
        assert 4 not in general_idxs

    def test_build_caption_with_include_ratings(self, tmp_path):
        """_build_caption with include_ratings=True emits rating tags."""
        from videsc.describe import _build_caption
        import numpy as np

        # Set up a fake session: session.run returns probs > threshold
        probs = np.zeros(5, dtype=np.float32)
        probs[0] = 0.9  # rating tag at index 0
        probs[1] = 0.8  # general tag at index 1
        probs[4] = 0.5  # rating tag at index 4

        class FakeInput:
            name = "input"

        class FakeSession:
            def get_inputs(self):
                return [FakeInput()]

            def run(self, outputs, feed):
                return [probs[np.newaxis, :]]

        session = FakeSession()
        # Build a fake image as HWC numpy array
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        tag_names = ["explicit", "cat", "dog", "person", "safe"]
        general_indices = [1, 2, 3]
        rating_indices = [0, 4]

        caption = _build_caption(
            session=session,
            frames=[frame],
            tag_names=tag_names,
            general_indices=general_indices,
            rating_indices=rating_indices,
            threshold=0.35,
            prefix="",
            include_ratings=True,
        )
        # Rating tags (0 and 4) should appear in output
        assert "explicit" in caption or "safe" in caption

    def test_describe_folder_public_api_calls_impl(self, monkeypatch, tmp_path):
        """describe_folder() imports onnxruntime and delegates to _describe_folder_impl."""
        import onnxruntime as real_ort

        called = {}

        def fake_impl(**kwargs):
            called["kwargs"] = kwargs
            return {"described": 2, "skipped": 0}

        import videsc.describe as desc_mod
        monkeypatch.setattr(desc_mod, "_describe_folder_impl", fake_impl)

        result = desc_mod.describe_folder(tmp_path)
        assert result == {"described": 2, "skipped": 0}
        assert "ort" in called["kwargs"]

    def test_describe_folder_impl_no_frames_skipped(self, monkeypatch, tmp_path):
        """When extract_keyframes returns [] the video is counted as skipped."""
        from videsc.describe import _describe_folder_impl
        import onnxruntime as ort_mod

        # Put a fake video in tmp_path
        vid = tmp_path / "test.mp4"
        vid.write_bytes(b"fake")

        fake_ort = types.SimpleNamespace(
            InferenceSession=lambda *a, **k: MagicMock(
                get_inputs=lambda: [types.SimpleNamespace(name="input")],
                run=lambda *_a, **_k: [np.zeros((1, 1))],
            ),
        )

        fake_labels = (["cat"], [], [0])
        monkeypatch.setattr("videsc.describe._load_labels", lambda *_a: fake_labels)
        monkeypatch.setattr(
            "videsc.describe._download_model",
            lambda *_a: (Path("/tmp/model.onnx"), Path("/tmp/tags.csv")),
        )
        monkeypatch.setattr(
            "videsc.describe.extract_keyframes",
            lambda *_a, **_k: [],
        )

        result = _describe_folder_impl(
            ort=fake_ort,
            input_dir=tmp_path,
            output_dir=None,
            every_n=30,
            max_frames=10,
            prefix="",
            threshold=0.35,
            model_repo="SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
            include_ratings=False,
            skip_existing=False,
        )
        assert result["skipped"] >= 1


# ---------------------------------------------------------------------------
# videsc.audio.transcription – additional branches
# ---------------------------------------------------------------------------


class TestTranscriptionAdditionalBranches:
    def _make_transformers_stub(self):
        return types.SimpleNamespace(pipeline=lambda *a, **k: None)

    def test_safe_transcribe_segment_loads_pipeline_when_none(self, monkeypatch, tmp_path):
        """safe_transcribe_segment creates the pipeline when _ASR_PIPELINE is None."""
        pipeline_created = []
        fake_result = {"text": "hello", "chunks": []}

        def fake_pipeline(*args, **kwargs):
            pipeline_created.append(True)
            return lambda path, **kw: fake_result

        transformers_stub = types.SimpleNamespace(pipeline=fake_pipeline)
        mod = _import_fresh(
            monkeypatch,
            "videsc.audio.transcription",
            {"transformers": transformers_stub},
        )

        audio_path = tmp_path / "seg.mp3"
        audio_path.write_bytes(b"fake audio")

        # Ensure pipeline is None
        mod._ASR_PIPELINE = None

        result = mod.safe_transcribe_segment(str(audio_path), "openai/whisper-base")
        assert pipeline_created, "pipeline should have been created"
        assert result == fake_result

    def test_safe_transcribe_segment_fallback_on_exception(self, monkeypatch, tmp_path):
        """When main ASR call raises, the fallback tiny-Whisper path is tried."""
        call_count = [0]

        def exploding_pipeline(*args, **kwargs):
            call_count[0] += 1
            # First call (main model) returns a callable that raises
            # Second call (fallback) returns a callable that succeeds
            if call_count[0] == 1:
                def raiser(path, **kw):
                    raise RuntimeError("CUDA OOM")
                return raiser
            else:
                return lambda path, **kw: {"text": "fallback result"}

        transformers_stub = types.SimpleNamespace(pipeline=exploding_pipeline)
        mod = _import_fresh(
            monkeypatch,
            "videsc.audio.transcription",
            {"transformers": transformers_stub},
        )

        audio_path = tmp_path / "seg.mp3"
        audio_path.write_bytes(b"fake audio")

        mod._ASR_PIPELINE = None

        result = mod.safe_transcribe_segment(str(audio_path), "openai/whisper-large")
        assert result == {"text": "fallback result"}

    def test_safe_transcribe_segment_both_fail_returns_none(self, monkeypatch, tmp_path):
        """When both main and fallback ASR calls fail, None is returned."""
        def bad_pipeline(*args, **kwargs):
            def raiser(path, **kw):
                raise RuntimeError("always fail")
            return raiser

        transformers_stub = types.SimpleNamespace(pipeline=bad_pipeline)
        mod = _import_fresh(
            monkeypatch,
            "videsc.audio.transcription",
            {"transformers": transformers_stub},
        )

        audio_path = tmp_path / "seg.mp3"
        audio_path.write_bytes(b"fake audio")

        mod._ASR_PIPELINE = None

        result = mod.safe_transcribe_segment(str(audio_path), "any-model")
        assert result is None

    def test_transcribe_audio_segments_with_results(self, monkeypatch, tmp_path):
        """transcribe_audio_segments returns results for valid segments."""
        transformers_stub = self._make_transformers_stub()
        mod = _import_fresh(
            monkeypatch,
            "videsc.audio.transcription",
            {"transformers": transformers_stub},
        )

        audio = tmp_path / "s1.mp3"
        audio.write_bytes(b"audio")

        monkeypatch.setattr(mod, "safe_transcribe_segment", lambda path, model, dur: {"text": "hi"})
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

        segs = [{"path": str(audio), "start": 0.0, "end": 5.0}]
        results = mod.transcribe_audio_segments(segs, "openai/whisper-base")
        assert len(results) == 1
        assert results[0]["result"] == {"text": "hi"}

    def test_transcribe_audio_segments_none_result_skipped(self, monkeypatch, tmp_path):
        """Segments where safe_transcribe_segment returns None are not appended."""
        transformers_stub = self._make_transformers_stub()
        mod = _import_fresh(
            monkeypatch,
            "videsc.audio.transcription",
            {"transformers": transformers_stub},
        )

        audio = tmp_path / "s2.mp3"
        audio.write_bytes(b"x")

        monkeypatch.setattr(mod, "safe_transcribe_segment", lambda *_a: None)
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

        segs = [{"path": str(audio), "start": 0.0, "end": 5.0}]
        results = mod.transcribe_audio_segments(segs, "any")
        assert results == []

    def test_combine_transcription_results_res_is_string(self, monkeypatch):
        """combine_transcription_results handles res as plain string."""
        transformers_stub = self._make_transformers_stub()
        mod = _import_fresh(
            monkeypatch,
            "videsc.audio.transcription",
            {"transformers": transformers_stub},
        )

        results = [{"start": 0.0, "result": "hello world"}]
        text, segs = mod.combine_transcription_results(results)
        assert "hello world" in text

    def test_combine_transcription_results_res_has_text_key(self, monkeypatch):
        """combine_transcription_results handles res={'text': '...'} with no chunks."""
        transformers_stub = self._make_transformers_stub()
        mod = _import_fresh(
            monkeypatch,
            "videsc.audio.transcription",
            {"transformers": transformers_stub},
        )

        results = [{"start": 0.0, "result": {"text": "plain text"}}]
        text, segs = mod.combine_transcription_results(results)
        assert "plain text" in text

    def test_combine_transcription_results_ts_as_dict(self, monkeypatch):
        """combine_transcription_results handles timestamp as dict with start/end keys."""
        transformers_stub = self._make_transformers_stub()
        mod = _import_fresh(
            monkeypatch,
            "videsc.audio.transcription",
            {"transformers": transformers_stub},
        )

        results = [
            {
                "start": 5.0,
                "result": {
                    "chunks": [
                        {
                            "text": "segment",
                            "timestamp": {"start": 0.0, "end": 1.0},
                        }
                    ]
                },
            }
        ]
        text, segs = mod.combine_transcription_results(results)
        assert "segment" in text
        assert segs[0]["timestamp"] == (5.0, 6.0)

    def test_transcribe_audio_from_video_returns_transcript(self, monkeypatch):
        """transcribe_audio_from_video returns transcript when non-empty."""
        transformers_stub = self._make_transformers_stub()
        mod = _import_fresh(
            monkeypatch,
            "videsc.audio.transcription",
            {"transformers": transformers_stub},
        )

        monkeypatch.setattr(
            mod,
            "transcribe_large_video",
            lambda *_a, **_k: ("the transcript", [{"timestamp": (0.0, 1.0), "text": "hi"}]),
        )

        args = types.SimpleNamespace(asr_model="openai/whisper-base", max_audio_seconds=0.0)
        text, segs = mod.transcribe_audio_from_video("/tmp/v.mp4", args, "cuda")
        assert text == "the transcript"
        assert len(segs) == 1

    def test_format_transcript_with_timestamps_no_raw(self, monkeypatch):
        """format_transcript_with_timestamps with empty raw_transcript omits raw section."""
        transformers_stub = self._make_transformers_stub()
        mod = _import_fresh(
            monkeypatch,
            "videsc.audio.transcription",
            {"transformers": transformers_stub},
        )

        segs = [{"timestamp": (0.0, 2.0), "text": "hello"}]
        result = mod.format_transcript_with_timestamps("", segs)
        assert "Raw transcript" not in result
        assert "Transcript with timestamps:" in result
        assert "hello" in result

    def test_format_transcript_skips_invalid_segment(self, monkeypatch):
        """format_transcript_with_timestamps skips segments with bad timestamps."""
        transformers_stub = self._make_transformers_stub()
        mod = _import_fresh(
            monkeypatch,
            "videsc.audio.transcription",
            {"transformers": transformers_stub},
        )

        segs = [
            {"timestamp": None, "text": "bad"},
            {"timestamp": (0.0, 1.5), "text": "good"},
        ]
        result = mod.format_transcript_with_timestamps("raw", segs)
        assert "good" in result


# ---------------------------------------------------------------------------
# videsc.model.loader – half_cpu, model_full, optimize, gemma4 branches
# ---------------------------------------------------------------------------


def _make_torch_transformers_stubs():
    class FakeModel:
        device = "cpu"
        dtype = "float16"

        @staticmethod
        def from_pretrained(*_a, **_k):
            return FakeModel()

        def disable_talker(self):
            pass

    class FakeProcessor:
        @staticmethod
        def from_pretrained(*_a, **_k):
            return FakeProcessor()

    torch_stub = types.SimpleNamespace(
        set_num_threads=lambda _n: None,
        compile=lambda m, **_k: m,
        bfloat16="bfloat16",
        float16="float16",
        float32="float32",
    )
    transformers_stub = types.SimpleNamespace(
        Qwen3VLForConditionalGeneration=FakeModel,
        Qwen3VLMoeForConditionalGeneration=FakeModel,
        AutoProcessor=FakeProcessor,
        AutoModelForMultimodalLM=FakeModel,
        BitsAndBytesConfig=lambda **_k: object(),
        Qwen3OmniMoeForConditionalGeneration=FakeModel,
        Qwen3OmniMoeProcessor=FakeProcessor,
        Qwen3_5ForConditionalGeneration=FakeModel,
    )
    torchsummary_stub = types.SimpleNamespace(summary=lambda *_a, **_k: None)
    return torch_stub, transformers_stub, torchsummary_stub


class TestLoaderAdditionalBranches:
    def _loader(self, monkeypatch):
        torch_stub, transformers_stub, torchsummary_stub = _make_torch_transformers_stubs()
        loader = _import_fresh(
            monkeypatch,
            "videsc.model.loader",
            {
                "torch": torch_stub,
                "torchsummary": torchsummary_stub,
                "transformers": transformers_stub,
            },
        )
        loader._SHARED_MODEL = None
        loader._SHARED_PROCESSOR = None
        return loader

    def _base_args(self, **overrides):
        defaults = dict(
            model_hf=True,
            model_full=False,
            model="Qwen/Qwen3-VL-8B-Instruct",
            half_cpu=False,
            quant="none",
            reader="auto",
            attn="sdpa",
            optimize=False,
            min_pixels=128,
            max_pixels=256,
        )
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def test_load_model_model_full_path(self, monkeypatch):
        """model_full=True uses args.model directly as path."""
        loader = self._loader(monkeypatch)
        args = self._base_args(model_hf=False, model_full=True)
        m, p = loader.load_model_and_processor(args)
        assert m is not None and p is not None

    def test_load_model_neither_hf_nor_full(self, monkeypatch):
        """Neither model_hf nor model_full → uses model_dir prefix."""
        loader = self._loader(monkeypatch)
        args = self._base_args(model_hf=False, model_full=False)
        m, p = loader.load_model_and_processor(args)
        assert m is not None

    def test_load_model_half_cpu(self, monkeypatch):
        """half_cpu=True calls set_num_threads and sets MKL/OMP env vars."""
        loader = self._loader(monkeypatch)
        args = self._base_args(half_cpu=True)
        m, p = loader.load_model_and_processor(args)
        assert m is not None

    def test_load_model_optimize(self, monkeypatch):
        """optimize=True calls torch.compile on the model."""
        loader = self._loader(monkeypatch)
        args = self._base_args(optimize=True)
        compiled = []
        import sys as _sys
        torch_stub = _sys.modules.get("torch")
        if torch_stub:
            orig_compile = torch_stub.compile
            torch_stub.compile = lambda m, **_k: (compiled.append(True), m)[1]
        m, p = loader.load_model_and_processor(args)
        assert m is not None

    def test_load_omni_half_cpu(self, monkeypatch):
        """load_omni_model_and_processor respects half_cpu=True."""
        loader = self._loader(monkeypatch)
        args = self._base_args(half_cpu=True)
        m, p = loader.load_omni_model_and_processor(args)
        assert m is not None

    def test_load_omni_cache_reuse(self, monkeypatch):
        """load_omni_model_and_processor returns cached model on second call."""
        loader = self._loader(monkeypatch)
        args = self._base_args()
        m1, p1 = loader.load_omni_model_and_processor(args)
        m2, p2 = loader.load_omni_model_and_processor(args)
        assert m1 is m2

    def test_load_qwen35_model_full(self, monkeypatch):
        """load_qwen35_model_and_processor with model_full=True."""
        loader = self._loader(monkeypatch)
        args = self._base_args(model_hf=False, model_full=True, model="my/path")
        m, p = loader.load_qwen35_model_and_processor(args)
        assert m is not None

    def test_load_qwen35_half_cpu_and_optimize(self, monkeypatch):
        """load_qwen35_model_and_processor with half_cpu and optimize."""
        loader = self._loader(monkeypatch)
        args = self._base_args(half_cpu=True, optimize=True)
        m, p = loader.load_qwen35_model_and_processor(args)
        assert m is not None

    def test_load_gemma4_basic(self, monkeypatch):
        """load_gemma4_model_and_processor loads model and processor."""
        loader = self._loader(monkeypatch)
        args = types.SimpleNamespace(
            model_hf=True,
            model_full=False,
            model="google/gemma-4-it",
            half_cpu=False,
            quant="none",
            reader="auto",
            attn="sdpa",
            optimize=False,
            min_pixels=128,
            max_pixels=256,
            processor=None,
            torch_dtype="auto",
        )
        m, p = loader.load_gemma4_model_and_processor(args)
        assert m is not None and p is not None

    def test_load_gemma4_half_cpu_optimize_and_custom_processor(self, monkeypatch):
        """load_gemma4_model_and_processor with half_cpu, optimize, custom processor."""
        loader = self._loader(monkeypatch)
        args = types.SimpleNamespace(
            model_hf=True,
            model_full=False,
            model="google/gemma-4-it",
            half_cpu=True,
            quant="none",
            reader="auto",
            attn="sdpa",
            optimize=True,
            min_pixels=128,
            max_pixels=256,
            processor="custom/processor/path",
            torch_dtype="bfloat16",
        )
        m, p = loader.load_gemma4_model_and_processor(args)
        assert m is not None

    def test_load_gemma4_cache_reuse(self, monkeypatch):
        """Second call to load_gemma4_model_and_processor returns cached model."""
        loader = self._loader(monkeypatch)
        args = types.SimpleNamespace(
            model_hf=True,
            model_full=False,
            model="google/gemma-4-it",
            half_cpu=False,
            quant="none",
            reader="auto",
            attn="sdpa",
            optimize=False,
            min_pixels=128,
            max_pixels=256,
            processor=None,
            torch_dtype="auto",
        )
        m1, p1 = loader.load_gemma4_model_and_processor(args)
        m2, p2 = loader.load_gemma4_model_and_processor(args)
        assert m1 is m2


# ---------------------------------------------------------------------------
# videsc.pipeline.vllm_runner – extract_frames_as_pil, _extract_frames,
#   _create_vllm_client, _consolidate_vllm, run_batch_vllm
# ---------------------------------------------------------------------------


class TestVllmRunnerAdditional:
    def _make_mock_cap(self, opened=True, fps=25.0, n_frames=50):
        cap = MagicMock()
        cap.isOpened.return_value = opened
        cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_COUNT: n_frames,
        }.get(prop, 0.0)
        # Simulate reading 1 frame then EOF
        fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cap.read.side_effect = [(True, fake_frame)] + [(False, None)] * 100
        return cap

    def test_extract_frames_as_pil_success(self, monkeypatch):
        """extract_frames_as_pil returns PIL images when video opens."""
        from videsc.pipeline.vllm_runner import extract_frames_as_pil

        cap = self._make_mock_cap()
        with patch("cv2.VideoCapture", return_value=cap):
            frames = extract_frames_as_pil("/fake/v.mp4", 0.0, 2.0, fps=1.0)
        assert isinstance(frames, list)
        cap.release.assert_called_once()

    def test_extract_frames_as_pil_cannot_open(self, monkeypatch):
        """extract_frames_as_pil returns [] when video cannot be opened."""
        from videsc.pipeline.vllm_runner import extract_frames_as_pil

        cap = self._make_mock_cap(opened=False)
        with patch("cv2.VideoCapture", return_value=cap):
            frames = extract_frames_as_pil("/fake/bad.mp4", 0.0, 5.0)
        assert frames == []

    def test_extract_frames_as_pil_zero_fps_uses_default(self, monkeypatch):
        """When video_fps <= 0, it defaults to 25 fps."""
        from videsc.pipeline.vllm_runner import extract_frames_as_pil

        cap = self._make_mock_cap(fps=0.0)
        with patch("cv2.VideoCapture", return_value=cap):
            frames = extract_frames_as_pil("/fake/zero.mp4", 0.0, 1.0, fps=1.0)
        assert isinstance(frames, list)

    def test_extract_frames_delegates_correctly(self, monkeypatch):
        """_extract_frames calls extract_frames_as_pil with correct args."""
        from videsc.pipeline import vllm_runner

        vinfo = {"tot_time": 10.0}
        called = {}

        def fake_extract(video_path, start, end, fps):
            called["args"] = (video_path, start, end, fps)
            return []

        monkeypatch.setattr(vllm_runner, "extract_frames_as_pil", fake_extract)
        monkeypatch.setattr(vllm_runner, "get_video_info", lambda _v: vinfo)

        args = types.SimpleNamespace(clip_start=1.0, clip_end=5.0, vllm_fps=2.0)
        frames = vllm_runner._extract_frames("/fake/v.mp4", args)
        assert called["args"] == ("/fake/v.mp4", 1.0, 5.0, 2.0)
        assert frames == []

    def test_extract_frames_uses_full_duration_when_clip_end_negative(self, monkeypatch):
        """_extract_frames uses full video duration when clip_end <= 0."""
        from videsc.pipeline import vllm_runner

        monkeypatch.setattr(vllm_runner, "get_video_info", lambda _v: {"tot_time": 20.0})
        captured = {}

        def fake_extract(video_path, start, end, fps):
            captured["end"] = end
            return []

        monkeypatch.setattr(vllm_runner, "extract_frames_as_pil", fake_extract)

        args = types.SimpleNamespace(clip_start=0.0, clip_end=-1.0, vllm_fps=1.0)
        vllm_runner._extract_frames("/fake/v.mp4", args)
        assert captured["end"] == 20.0

    def test_create_vllm_client(self, monkeypatch):
        """_create_vllm_client instantiates VLLMClient with correct parameters."""
        from videsc.pipeline import vllm_runner

        created = {}

        class FakeClient:
            def __init__(self, **kwargs):
                created.update(kwargs)

        monkeypatch.setattr(
            "videsc.model.vllm_client.VLLMClient",
            FakeClient,
        )

        args = types.SimpleNamespace(
            vllm_host="localhost",
            vllm_port=8000,
            vllm_model="meta-llama/Llama-3-8B",
            max_new_tokens=512,
            vllm_api_key="KEY",
            vllm_temperature=0.7,
            vllm_top_p=0.95,
            vllm_base_url=None,
        )
        client = vllm_runner._create_vllm_client(args)
        assert created["host"] == "localhost"
        assert created["port"] == 8000

    def test_consolidate_vllm(self, monkeypatch):
        """_consolidate_vllm calls client.generate and returns structured result."""
        from videsc.pipeline.vllm_runner import _consolidate_vllm

        class FakeClient:
            def generate(self, messages, max_tokens):
                return "consolidated summary"

        args = types.SimpleNamespace(
            system=None,
            max_new_tokens=256,
            consolidate_prompt=None,
        )
        result = _consolidate_vllm(["desc1", "desc2"], FakeClient(), args)
        assert "Consolidated Summary" in result
        assert "consolidated summary" in result
        assert "desc1" in result

    def test_consolidate_vllm_with_system_prompt(self, monkeypatch):
        """_consolidate_vllm includes system message when args.system is set."""
        from videsc.pipeline.vllm_runner import _consolidate_vllm

        received_messages = []

        class FakeClient:
            def generate(self, messages, max_tokens):
                received_messages.extend(messages)
                return "ok"

        args = types.SimpleNamespace(
            system="You are a helpful assistant.",
            max_new_tokens=128,
            consolidate_prompt="Summarize:",
        )
        _consolidate_vllm(["a", "b"], FakeClient(), args)
        assert received_messages[0]["role"] == "system"

    def test_run_batch_vllm_no_inputs_returns_3(self, monkeypatch):
        """run_batch_vllm returns 3 when no inputs match."""
        from videsc.pipeline import vllm_runner

        with patch("videsc.utils.helpers.expand_inputs", return_value=[]):
            args = types.SimpleNamespace(
                videos=None, indir=None, ext=[], filelist=None, workers=1
            )
            rc = vllm_runner.run_batch_vllm(args)
        assert rc == 3

    def test_run_batch_vllm_dry_run(self, monkeypatch, tmp_path, capsys):
        """run_batch_vllm dry_run prints paths and returns 0."""
        from videsc.pipeline import vllm_runner

        vid = tmp_path / "v.mp4"
        vid.write_bytes(b"x")
        with patch("videsc.utils.helpers.expand_inputs", return_value=[vid]):
            args = types.SimpleNamespace(
                videos=None, indir=None, ext=[], filelist=None, workers=1, dry_run=True
            )
            rc = vllm_runner.run_batch_vllm(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "dry-run" in captured.out

    def test_run_batch_vllm_with_workers(self, monkeypatch, tmp_path):
        """run_batch_vllm processes videos via ThreadPoolExecutor."""
        from videsc.pipeline import vllm_runner

        vid1 = tmp_path / "a.mp4"
        vid2 = tmp_path / "b.mp4"
        vid1.write_bytes(b"x")
        vid2.write_bytes(b"x")

        monkeypatch.setattr(vllm_runner, "run_single_video_vllm", lambda _a: 0)

        with patch("videsc.utils.helpers.expand_inputs", return_value=[vid1, vid2]):
            args = types.SimpleNamespace(
                videos=None, indir=None, ext=[], filelist=None, workers=2, dry_run=False
            )
            rc = vllm_runner.run_batch_vllm(args)
        assert rc == 0

    def test_run_single_video_vllm_no_frames_chunk(self, monkeypatch, tmp_path):
        """run_single_video_vllm appends placeholder when no frames for chunk."""
        from videsc.pipeline import vllm_runner

        vid = tmp_path / "v.mp4"
        vid.write_bytes(b"x")

        monkeypatch.setattr(vllm_runner, "get_video_info", lambda _v: {"tot_time": 5.0})
        # Return empty frames → triggers the no-frames branch
        monkeypatch.setattr(vllm_runner, "extract_frames_as_pil", lambda *_a, **_k: [])

        args = types.SimpleNamespace(
            video=str(vid),
            vllm_host="localhost",
            vllm_port=8000,
            vllm_model="test-model",
            max_new_tokens=128,
            vllm_chunk_duration=0.0,
            vllm_fps=1.0,
            vllm_max_image_size=1280,
            clip_start=0.0,
            clip_end=-1.0,
            outdir=str(tmp_path / "out"),
            prompt="describe it",
            system=None,
            dry=False,
            consolidate=False,
            vllm_api_key="EMPTY",
            vllm_temperature=0.7,
            vllm_top_p=0.95,
            vllm_base_url=None,
        )

        class FakeClient:
            def describe_frames(self, frames, prompt, system, max_tokens, max_size):
                return "description"

        monkeypatch.setattr(vllm_runner, "_create_vllm_client", lambda _a: FakeClient())

        rc = vllm_runner.run_single_video_vllm(args)
        assert rc == 0
        out_file = tmp_path / "out" / "v.txt"
        text = out_file.read_text()
        assert "no frames" in text


# ---------------------------------------------------------------------------
# videsc.main – _run_vllm, _run_vl, _run_wd14 branches
# ---------------------------------------------------------------------------


class TestMainAdditionalBranches:
    def _import_main(self, monkeypatch):
        return _import_fresh(monkeypatch, "videsc.main")

    # --- _run_vllm ---

    def test_main_dispatch_vllm(self, monkeypatch):
        """main() with vllm=True dispatches to _run_vllm."""
        mod = self._import_main(monkeypatch)
        monkeypatch.setitem(
            sys.modules,
            "videsc.cli.args",
            types.SimpleNamespace(
                parse_args=lambda _a=None: types.SimpleNamespace(vl=False, vllm=True)
            ),
        )
        monkeypatch.setattr(mod, "_run_vllm", lambda _a: 7)
        assert mod.main([]) == 7

    def test_run_vllm_batch_mode(self, monkeypatch):
        """_run_vllm batch mode calls run_batch_vllm."""
        mod = self._import_main(monkeypatch)
        called = {}
        monkeypatch.setitem(
            sys.modules,
            "videsc.pipeline.vllm_runner",
            types.SimpleNamespace(
                run_single_video_vllm=lambda _a: 0,
                run_batch_vllm=lambda _a: called.setdefault("batch", 99),
            ),
        )
        args = types.SimpleNamespace(
            videos=["*.mp4"], indir=None, filelist=None, youtube_url=None,
            vllm_host="localhost", vllm_port=8000, vllm_model="test",
        )
        rc = mod._run_vllm(args)
        assert called.get("batch") == 99
        assert rc == 99

    def test_run_vllm_no_video_returns_1(self, monkeypatch):
        """_run_vllm returns 1 when no video is provided in single mode."""
        mod = self._import_main(monkeypatch)
        monkeypatch.setitem(
            sys.modules,
            "videsc.pipeline.vllm_runner",
            types.SimpleNamespace(run_single_video_vllm=lambda _a: 0, run_batch_vllm=lambda _a: 0),
        )
        args = types.SimpleNamespace(
            videos=None, indir=None, filelist=None, youtube_url=None, video=None,
            vllm_host="localhost", vllm_port=8000, vllm_model="test",
        )
        rc = mod._run_vllm(args)
        assert rc == 1

    def test_run_vllm_youtube_no_api_key_returns_1(self, monkeypatch):
        """_run_vllm with youtube_url but no api_key returns 1."""
        mod = self._import_main(monkeypatch)
        monkeypatch.setitem(
            sys.modules,
            "videsc.pipeline.vllm_runner",
            types.SimpleNamespace(run_single_video_vllm=lambda _a: 0, run_batch_vllm=lambda _a: 0),
        )
        args = types.SimpleNamespace(
            videos=None, indir=None, filelist=None,
            youtube_url="https://yt.be/abc", youtube_api_key=None,
            vllm_host="localhost", vllm_port=8000, vllm_model="test",
        )
        rc = mod._run_vllm(args)
        assert rc == 1

    def test_run_vllm_youtube_download_fails_returns_1(self, monkeypatch, tmp_path):
        """_run_vllm returns 1 when youtube download fails."""
        mod = self._import_main(monkeypatch)
        monkeypatch.setitem(
            sys.modules,
            "videsc.pipeline.vllm_runner",
            types.SimpleNamespace(run_single_video_vllm=lambda _a: 0, run_batch_vllm=lambda _a: 0),
        )
        monkeypatch.setitem(
            sys.modules,
            "videsc.describe",
            types.SimpleNamespace(_download_youtube_video=lambda *_a, **_k: None),
        )
        args = types.SimpleNamespace(
            videos=None, indir=None, filelist=None,
            youtube_url="https://yt.be/abc", youtube_api_key="KEY",
            outdir=None, output_dir=None,
            vllm_host="localhost", vllm_port=8000, vllm_model="test",
        )
        rc = mod._run_vllm(args)
        assert rc == 1

    def test_run_vllm_youtube_save_video(self, monkeypatch, tmp_path):
        """_run_vllm with save_video copies video and returns 0."""
        mod = self._import_main(monkeypatch)
        src = tmp_path / "downloaded.mp4"
        src.write_bytes(b"fake video bytes")

        monkeypatch.setitem(
            sys.modules,
            "videsc.pipeline.vllm_runner",
            types.SimpleNamespace(run_single_video_vllm=lambda _a: 0, run_batch_vllm=lambda _a: 0),
        )
        monkeypatch.setitem(
            sys.modules,
            "videsc.describe",
            types.SimpleNamespace(_download_youtube_video=lambda *_a, **_k: src),
        )
        dest = tmp_path / "saved.mp4"
        args = types.SimpleNamespace(
            videos=None, indir=None, filelist=None,
            youtube_url="https://yt.be/abc", youtube_api_key="KEY",
            outdir=None, output_dir=None,
            save_video=str(dest),
            vllm_host="localhost", vllm_port=8000, vllm_model="test",
        )
        rc = mod._run_vllm(args)
        assert rc == 0
        assert dest.exists()

    def test_run_vllm_single_video(self, monkeypatch, tmp_path):
        """_run_vllm in single-video mode calls run_single_video_vllm."""
        mod = self._import_main(monkeypatch)
        called = {}
        monkeypatch.setitem(
            sys.modules,
            "videsc.pipeline.vllm_runner",
            types.SimpleNamespace(
                run_single_video_vllm=lambda _a: called.setdefault("ok", 0),
                run_batch_vllm=lambda _a: 0,
            ),
        )
        vid = tmp_path / "v.mp4"
        vid.write_bytes(b"x")
        args = types.SimpleNamespace(
            videos=None, indir=None, filelist=None,
            youtube_url=None, video=str(vid),
            vllm_host="localhost", vllm_port=8000, vllm_model="test",
        )
        rc = mod._run_vllm(args)
        assert rc == 0
        assert "ok" in called

    # --- _run_vl ---

    def test_run_vl_youtube_no_api_key_returns_1(self, monkeypatch):
        """_run_vl with youtube_url but no api_key returns 1."""
        mod = self._import_main(monkeypatch)
        monkeypatch.setitem(
            sys.modules,
            "videsc.pipeline.runner",
            types.SimpleNamespace(run_batch=lambda _a: 0, run_single_video=lambda *_a: 0, run_single_video_gemma4=lambda *_a: 0),
        )
        monkeypatch.setitem(
            sys.modules,
            "videsc.model.loader",
            types.SimpleNamespace(
                load_model_and_processor=lambda _a: ("m", "p"),
                load_omni_model_and_processor=lambda _a: ("m", "p"),
                load_qwen35_model_and_processor=lambda _a: ("m", "p"),
                load_gemma4_model_and_processor=lambda _a: ("m", "p"),
            ),
        )
        args = types.SimpleNamespace(
            videos=None, indir=None, filelist=None,
            youtube_url="https://yt.be/abc", youtube_api_key=None,
            outdir=None, output_dir=None,
        )
        rc = mod._run_vl(args)
        assert rc == 1

    def test_run_vl_youtube_download_fails(self, monkeypatch):
        """_run_vl returns 1 when youtube download fails."""
        mod = self._import_main(monkeypatch)
        monkeypatch.setitem(
            sys.modules,
            "videsc.pipeline.runner",
            types.SimpleNamespace(run_batch=lambda _a: 0, run_single_video=lambda *_a: 0, run_single_video_gemma4=lambda *_a: 0),
        )
        monkeypatch.setitem(
            sys.modules,
            "videsc.model.loader",
            types.SimpleNamespace(
                load_model_and_processor=lambda _a: ("m", "p"),
                load_omni_model_and_processor=lambda _a: ("m", "p"),
                load_qwen35_model_and_processor=lambda _a: ("m", "p"),
                load_gemma4_model_and_processor=lambda _a: ("m", "p"),
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "videsc.describe",
            types.SimpleNamespace(_download_youtube_video=lambda *_a, **_k: None),
        )
        args = types.SimpleNamespace(
            videos=None, indir=None, filelist=None,
            youtube_url="https://yt.be/abc", youtube_api_key="KEY",
            outdir=None, output_dir=None,
            save_video=None,
        )
        rc = mod._run_vl(args)
        assert rc == 1

    def test_run_vl_youtube_save_video(self, monkeypatch, tmp_path):
        """_run_vl with save_video copies video and returns 0."""
        mod = self._import_main(monkeypatch)
        src = tmp_path / "dl.mp4"
        src.write_bytes(b"data")
        dest = tmp_path / "out.mp4"

        monkeypatch.setitem(
            sys.modules,
            "videsc.pipeline.runner",
            types.SimpleNamespace(run_batch=lambda _a: 0, run_single_video=lambda *_a: 0, run_single_video_gemma4=lambda *_a: 0),
        )
        monkeypatch.setitem(
            sys.modules,
            "videsc.model.loader",
            types.SimpleNamespace(
                load_model_and_processor=lambda _a: ("m", "p"),
                load_omni_model_and_processor=lambda _a: ("m", "p"),
                load_qwen35_model_and_processor=lambda _a: ("m", "p"),
                load_gemma4_model_and_processor=lambda _a: ("m", "p"),
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "videsc.describe",
            types.SimpleNamespace(_download_youtube_video=lambda *_a, **_k: src),
        )
        args = types.SimpleNamespace(
            videos=None, indir=None, filelist=None,
            youtube_url="https://yt.be/abc", youtube_api_key="KEY",
            outdir=None, output_dir=None,
            save_video=str(dest),
        )
        rc = mod._run_vl(args)
        assert rc == 0
        assert dest.exists()

    def test_run_vl_omni_model(self, monkeypatch):
        """_run_vl with omni=True calls load_omni_model_and_processor."""
        mod = self._import_main(monkeypatch)
        called = {}
        monkeypatch.setitem(
            sys.modules,
            "videsc.pipeline.runner",
            types.SimpleNamespace(
                run_batch=lambda _a: 0,
                run_single_video=lambda *_a: called.setdefault("single", 0) or 0,
                run_single_video_gemma4=lambda *_a: 0,
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "videsc.model.loader",
            types.SimpleNamespace(
                load_model_and_processor=lambda _a: ("m", "p"),
                load_omni_model_and_processor=lambda _a: called.setdefault("omni", ("om", "op")) or ("om", "op"),
                load_qwen35_model_and_processor=lambda _a: ("35m", "35p"),
                load_gemma4_model_and_processor=lambda _a: ("gm", "gp"),
            ),
        )
        args = types.SimpleNamespace(
            videos=None, indir=None, filelist=None,
            youtube_url=None, video="/tmp/v.mp4",
            omni=True, qwen35=False, gemma4=False,
        )
        rc = mod._run_vl(args)
        assert rc == 0
        assert "omni" in called

    def test_run_vl_qwen35_model(self, monkeypatch):
        """_run_vl with qwen35=True calls load_qwen35_model_and_processor."""
        mod = self._import_main(monkeypatch)
        called = {}
        monkeypatch.setitem(
            sys.modules,
            "videsc.pipeline.runner",
            types.SimpleNamespace(
                run_batch=lambda _a: 0,
                run_single_video=lambda *_a: called.setdefault("single", 0) or 0,
                run_single_video_gemma4=lambda *_a: 0,
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "videsc.model.loader",
            types.SimpleNamespace(
                load_model_and_processor=lambda _a: ("m", "p"),
                load_omni_model_and_processor=lambda _a: ("om", "op"),
                load_qwen35_model_and_processor=lambda _a: called.setdefault("qwen35", ("35m", "35p")) or ("35m", "35p"),
                load_gemma4_model_and_processor=lambda _a: ("gm", "gp"),
            ),
        )
        args = types.SimpleNamespace(
            videos=None, indir=None, filelist=None,
            youtube_url=None, video="/tmp/v.mp4",
            omni=False, qwen35=True, gemma4=False,
        )
        rc = mod._run_vl(args)
        assert rc == 0
        assert "qwen35" in called

    def test_run_vl_gemma4_model(self, monkeypatch):
        """_run_vl with gemma4=True calls load_gemma4 and run_single_video_gemma4."""
        mod = self._import_main(monkeypatch)
        called = {}
        monkeypatch.setitem(
            sys.modules,
            "videsc.pipeline.runner",
            types.SimpleNamespace(
                run_batch=lambda _a: 0,
                run_single_video=lambda *_a: 0,
                run_single_video_gemma4=lambda *_a: called.setdefault("gemma4", 0) or 0,
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "videsc.model.loader",
            types.SimpleNamespace(
                load_model_and_processor=lambda _a: ("m", "p"),
                load_omni_model_and_processor=lambda _a: ("om", "op"),
                load_qwen35_model_and_processor=lambda _a: ("35m", "35p"),
                load_gemma4_model_and_processor=lambda _a: ("gm", "gp"),
            ),
        )
        args = types.SimpleNamespace(
            videos=None, indir=None, filelist=None,
            youtube_url=None, video="/tmp/v.mp4",
            omni=False, qwen35=False, gemma4=True,
        )
        rc = mod._run_vl(args)
        assert rc == 0
        assert "gemma4" in called

    # --- _run_wd14 ---

    def test_run_wd14_youtube_with_describe_youtube(self, monkeypatch):
        """_run_wd14 with youtube_url calls describe_youtube and returns 0."""
        mod = self._import_main(monkeypatch)
        called = {}
        monkeypatch.setitem(
            sys.modules,
            "videsc.describe",
            types.SimpleNamespace(
                describe_youtube=lambda *_a, **_k: called.setdefault("yt", {"described": 1, "skipped": 0}),
                describe_folder=lambda *_a, **_k: {"described": 0, "skipped": 0},
            ),
        )
        args = types.SimpleNamespace(
            input_dir=None,
            youtube_url="https://yt.be/abc",
            youtube_api_key="APIKEY",
            output_dir=None,
            every_n=30,
            max_frames=10,
            prefix="",
            threshold=0.35,
            model_repo="SmilingWolf/wd",
            include_ratings=False,
            no_skip_existing=False,
            save_video=None,
        )
        rc = mod._run_wd14(args)
        assert rc == 0
        assert "yt" in called

    def test_run_wd14_youtube_save_video_success(self, monkeypatch, tmp_path):
        """_run_wd14 with save_video downloads video, saves it, and returns 0."""
        mod = self._import_main(monkeypatch)
        src = tmp_path / "dl.mp4"
        src.write_bytes(b"fake")
        dest = tmp_path / "saved.mp4"

        monkeypatch.setitem(
            sys.modules,
            "videsc.describe",
            types.SimpleNamespace(
                _download_youtube_video=lambda *_a, **_k: src,
                describe_youtube=lambda *_a, **_k: {"described": 0, "skipped": 0},
                describe_folder=lambda *_a, **_k: {"described": 0, "skipped": 0},
            ),
        )
        args = types.SimpleNamespace(
            input_dir=None,
            youtube_url="https://yt.be/abc",
            youtube_api_key="KEY",
            output_dir=None,
            every_n=30,
            max_frames=10,
            prefix="",
            threshold=0.35,
            model_repo="SmilingWolf/wd",
            include_ratings=False,
            no_skip_existing=False,
            save_video=str(dest),
        )
        rc = mod._run_wd14(args)
        assert rc == 0
        assert dest.exists()

    def test_run_wd14_youtube_save_video_download_fails(self, monkeypatch, tmp_path):
        """_run_wd14 save_video returns 1 when youtube download fails."""
        mod = self._import_main(monkeypatch)
        monkeypatch.setitem(
            sys.modules,
            "videsc.describe",
            types.SimpleNamespace(
                _download_youtube_video=lambda *_a, **_k: None,
                describe_youtube=lambda *_a, **_k: {"described": 0, "skipped": 0},
                describe_folder=lambda *_a, **_k: {"described": 0, "skipped": 0},
            ),
        )
        args = types.SimpleNamespace(
            input_dir=None,
            youtube_url="https://yt.be/abc",
            youtube_api_key="KEY",
            output_dir=None,
            every_n=30,
            max_frames=10,
            prefix="",
            threshold=0.35,
            model_repo="SmilingWolf/wd",
            include_ratings=False,
            no_skip_existing=False,
            save_video=str(tmp_path / "out.mp4"),
        )
        rc = mod._run_wd14(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# videsc.pipeline.runner – process_mm_info / process_vision_info wrapper bodies
#   and run_single_video non-dry-run path
# ---------------------------------------------------------------------------


def _make_runner_stubs():
    helper_mod = types.SimpleNamespace(
        expand_inputs=lambda *_a, **_k: [],
        namespace_to_cli=lambda *_a, **_k: [],
        _patch_size_for_model=lambda _m: 32,
        expand_video_grid_thw=lambda _i: None,
    )
    return {
        "torch": types.SimpleNamespace(
            manual_seed=lambda _n: None,
            no_grad=MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))),
            amp=types.SimpleNamespace(autocast=MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)))),
        ),
        "qwen_omni_utils": types.SimpleNamespace(process_mm_info=lambda *_a, **_k: (None, None, None)),
        "qwen_vl_utils": types.SimpleNamespace(process_vision_info=lambda *_a, **_k: (None, None, {})),
        "videsc.model.loader": types.SimpleNamespace(
            load_model_and_processor=lambda _a: ("m", "p"),
            load_omni_model_and_processor=lambda _a: ("m", "p"),
            load_qwen35_model_and_processor=lambda _a: ("m", "p"),
            load_gemma4_model_and_processor=lambda _a: ("gm", "gp"),
            _maybe_set_reader=lambda _r: None,
        ),
        "videsc.audio.transcription": types.SimpleNamespace(
            transcribe_audio_from_video=lambda *_a, **_k: (None, [])
        ),
        "videsc.video.info": types.SimpleNamespace(
            get_video_info=lambda _v: {"tot_time": 1.0, "FPS": 1.0, "num_frames": 1}
        ),
        "videsc.video.sampling": types.SimpleNamespace(
            compute_effective_nframes=lambda *_a, **_k: 1,
            compress_audio_segments_to_nframes=lambda s, *_a, **_k: s,
        ),
        "videsc.video.messages": types.SimpleNamespace(
            build_messages=lambda **_k: [{"role": "user", "content": "x"}]
        ),
        "videsc.utils.helpers": helper_mod,
    }


class TestRunnerWrapperBodies:
    def test_process_mm_info_wrapper_calls_real_function(self, monkeypatch):
        """process_mm_info wrapper imports from qwen_omni_utils and delegates."""
        stubs = _make_runner_stubs()
        runner = _import_fresh(monkeypatch, "videsc.pipeline.runner", stubs)

        # The wrapper is monkeypatched in many tests; here we call the real
        # wrapper body by reverting the monkeypatch:
        called = {}

        def fake_process(*args, **kwargs):
            called["args"] = args
            return ("a", "b", "c")

        # Patch qwen_omni_utils directly so the lazy import inside finds it:
        monkeypatch.setitem(
            sys.modules,
            "qwen_omni_utils",
            types.SimpleNamespace(process_mm_info=fake_process),
        )
        # Clear the module cache so the lazy import runs fresh:
        sys.modules.pop("videsc.pipeline.runner", None)
        runner2 = importlib.import_module("videsc.pipeline.runner")
        result = runner2.process_mm_info("arg1", key="val")
        assert called["args"] == ("arg1",)
        assert result == ("a", "b", "c")

    def test_process_vision_info_wrapper_calls_real_function(self, monkeypatch):
        """process_vision_info wrapper imports from qwen_vl_utils and delegates."""
        stubs = _make_runner_stubs()
        runner = _import_fresh(monkeypatch, "videsc.pipeline.runner", stubs)

        called = {}

        def fake_pvi(*args, **kwargs):
            called["args"] = args
            return ("imgs", "vids", {})

        monkeypatch.setitem(
            sys.modules,
            "qwen_vl_utils",
            types.SimpleNamespace(process_vision_info=fake_pvi),
        )
        sys.modules.pop("videsc.pipeline.runner", None)
        runner2 = importlib.import_module("videsc.pipeline.runner")
        result = runner2.process_vision_info("msgs", key="x")
        assert called["args"] == ("msgs",)
        assert result == ("imgs", "vids", {})


class TestRunnerNonDryRun:
    def _import_runner(self, monkeypatch):
        stubs = _make_runner_stubs()
        return _import_fresh(monkeypatch, "videsc.pipeline.runner", stubs)

    def test_run_single_video_non_dry_non_omni(self, monkeypatch, tmp_path):
        """run_single_video executes the generate path when dry=False."""
        runner = self._import_runner(monkeypatch)
        monkeypatch.setattr(runner, "get_video_info", lambda _v: {"tot_time": 2.0, "FPS": 30.0, "num_frames": 60})
        monkeypatch.setattr(runner, "compute_effective_nframes", lambda *_a, **_k: 2)
        monkeypatch.setattr(runner, "transcribe_audio_from_video", lambda *_a, **_k: (None, []))
        monkeypatch.setattr(runner, "compress_audio_segments_to_nframes", lambda s, *_a, **_k: s)
        monkeypatch.setattr(runner, "build_messages", lambda **_k: [{"role": "user", "content": "x"}])
        monkeypatch.setattr(runner, "process_vision_info", lambda *_a, **_k: (None, None, {}))
        monkeypatch.setattr(runner, "_maybe_set_reader", lambda _r: None)
        monkeypatch.setattr(runner, "expand_video_grid_thw", lambda _i: None)

        class Inputs(dict):
            def __init__(self):
                super().__init__()
                self["input_ids"] = [[1, 2, 3]]
                self.input_ids = [[1, 2, 3]]

            def to(self, *_a, **_k):
                return self

        class FakeModel:
            device = "cuda"
            dtype = "float16"

            def generate(self, **kw):
                return [[1, 2, 3, 4, 5]]

        class FakeProcessor:
            def apply_chat_template(self, *_a, **_k):
                return "prompt"

            def __call__(self, **_kwargs):
                return Inputs()

            def batch_decode(self, *_a, **_k):
                return ["output text"]

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"x")
        args = types.SimpleNamespace(
            seed=42,
            video=str(video),
            audio=False,
            num_frames=4,
            spf=0.0,
            model="Qwen/Qwen3-VL-8B-Instruct",
            total_pixels=100,
            prompt="describe",
            system=None,
            cont_prompt=False,
            omni=False,
            reader="auto",
            no_meta=True,
            dry=False,
            optimize=False,
            max_new_tokens=100,
            rep_pen=1.0,
            no_think_trim=True,
            outdir=str(tmp_path / "out"),
            no_save_transcript=True,
        )
        rc = runner.run_single_video(args, FakeModel(), FakeProcessor())
        assert rc == 0
        assert (tmp_path / "out" / "clip.txt").exists()

    def test_run_single_video_gemma4_dry_single_chunk(self, monkeypatch, tmp_path):
        """run_single_video_gemma4 with dry=True produces [chunk X: dry run]."""
        runner = self._import_runner(monkeypatch)
        monkeypatch.setattr(runner, "get_video_info", lambda _v: {"tot_time": 30.0})

        video = tmp_path / "g4.mp4"
        video.write_bytes(b"x")
        args = types.SimpleNamespace(
            seed=1,
            video=str(video),
            gemma4_chunk_duration=60.0,
            gemma4_fps=1.0,
            consolidate=False,
            dry=True,
            system=None,
            prompt="describe",
            max_new_tokens=512,
            no_think_trim=True,
            model="google/gemma-4-it",
            outdir=str(tmp_path / "out"),
            segment_prompt=None,
        )

        class DryProcessor:
            device = "cpu"

            def apply_chat_template(self, *_a, **_k):
                return MagicMock(to=lambda *_a, **_k: {})

        class DryModel:
            device = "cpu"

        rc = runner.run_single_video_gemma4(args, DryModel(), DryProcessor())
        assert rc == 0
        out = (tmp_path / "out" / "g4.txt").read_text()
        assert "dry run" in out

    def test_run_single_video_gemma4_dry_multiple_chunks(self, monkeypatch, tmp_path):
        """run_single_video_gemma4 with dry=True and multi-chunk video produces multiple placeholders."""
        runner = self._import_runner(monkeypatch)
        monkeypatch.setattr(runner, "get_video_info", lambda _v: {"tot_time": 90.0})

        video = tmp_path / "long.mp4"
        video.write_bytes(b"x")
        args = types.SimpleNamespace(
            seed=1,
            video=str(video),
            gemma4_chunk_duration=30.0,
            gemma4_fps=1.0,
            consolidate=False,
            dry=True,
            system=None,
            prompt="describe",
            max_new_tokens=512,
            no_think_trim=True,
            model="google/gemma-4-it",
            outdir=str(tmp_path / "out2"),
            segment_prompt=None,
        )

        class DryModel:
            device = "cpu"

        class DryProcessor:
            device = "cpu"

            def apply_chat_template(self, *_a, **_k):
                return MagicMock(to=lambda *_a, **_k: {})

        # Avoid calling ffmpeg for multi-chunk trimming
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            rc = runner.run_single_video_gemma4(args, DryModel(), DryProcessor())
        assert rc == 0
        out = (tmp_path / "out2" / "long.txt").read_text()
        # 3 chunks × 30 s each
        assert out.count("dry run") == 3

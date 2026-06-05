from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest


def _import_fresh(monkeypatch, module_name: str, stubs: dict[str, object] | None = None):
    for name, stub in (stubs or {}).items():
        monkeypatch.setitem(sys.modules, name, stub)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_root_main_shim_imports_portrait_prep_main(monkeypatch):
    called = {"n": 0}

    def fake_main():
        called["n"] += 1

    stub = types.SimpleNamespace(main=fake_main)
    module = _import_fresh(monkeypatch, "main", {"portrait_prep.cli": stub})
    module.main()
    assert called["n"] == 1


class TestPortraitPrepCli:
    def test_parse_args_defaults(self):
        mod = importlib.import_module("portrait_prep.cli")
        args = mod.parse_args(["--input-dir", "/tmp/in"])
        assert args.output_dir is None
        assert args.threshold == 0.35
        assert args.steps == ["convert", "crop", "caption", "augment", "cpcap"]

    def test_run_convert_calls_convert_folder(self, monkeypatch, tmp_path):
        mod = importlib.import_module("portrait_prep.cli")
        called = {}

        def fake_convert_folder(inp, out, skip_existing):
            called["args"] = (inp, out, skip_existing)
            return 3, 4

        stub = types.SimpleNamespace(convert_folder=fake_convert_folder)
        monkeypatch.setitem(sys.modules, "portrait_prep.convert", stub)

        args = types.SimpleNamespace(
            input_dir=tmp_path / "in",
            output_dir=tmp_path / "out",
            no_skip_existing=False,
        )
        mod.run_convert(args)
        assert called["args"] == (args.input_dir, args.output_dir, True)

    def test_run_convert_exits_when_output_missing(self, monkeypatch, tmp_path):
        mod = importlib.import_module("portrait_prep.cli")
        monkeypatch.setitem(
            sys.modules,
            "portrait_prep.convert",
            types.SimpleNamespace(convert_folder=lambda *_a, **_k: (0, 0)),
        )
        args = types.SimpleNamespace(input_dir=tmp_path / "in", output_dir=None, no_skip_existing=False)
        with pytest.raises(SystemExit):
            mod.run_convert(args)

    def test_main_routes_pipeline_steps(self, monkeypatch, tmp_path):
        mod = importlib.import_module("portrait_prep.cli")
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        args = types.SimpleNamespace(
            input_dir=in_dir,
            output_dir=out_dir,
            steps=["convert", "crop", "caption", "augment", "cpcap"],
            source_dir=None,
        )
        calls: list[str] = []

        monkeypatch.setattr(mod, "parse_args", lambda _argv=None: args)
        monkeypatch.setattr(mod, "run_convert", lambda _a: calls.append("convert"))
        monkeypatch.setattr(mod, "run_crop", lambda _a, _i: calls.append("crop"))
        monkeypatch.setattr(mod, "run_caption", lambda _a, _i: calls.append("caption"))
        monkeypatch.setattr(mod, "run_augment", lambda _a, _i: calls.append("augment"))
        monkeypatch.setattr(mod, "run_cpcap", lambda _a, _i: calls.append("cpcap"))

        mod.main([])
        assert calls == ["convert", "crop", "caption", "augment", "cpcap"]
        assert args.source_dir == in_dir.resolve()


class TestVicropCli:
    def test_parse_args_defaults(self, tmp_path):
        mod = importlib.import_module("vicrop.cli")
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        input_file = tmp_path / "clip.mp4"
        input_file.touch()
        output_dir = tmp_path / "out"

        dir_args = mod.parse_args(["--input", str(input_dir), "--output-dir", str(output_dir)])
        file_args = mod.parse_args(["--input", str(input_file), "--output-dir", str(output_dir)])

        assert dir_args.input == input_dir
        assert dir_args.every_n == 30
        assert dir_args.detection_model == "hog"

        assert file_args.input == input_file
        assert file_args.every_n == 30
        assert file_args.detection_model == "hog"
    def test_main_calls_crop_folder(self, monkeypatch, tmp_path):
        mod = importlib.import_module("vicrop.cli")
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        args = types.SimpleNamespace(
            input=in_dir,
            output_dir=tmp_path / "out",
            every_n=10,
            margin_ratio=0.5,
            crop_size=256,
            detection_model="cnn",
            no_classify=False,
            tolerance=0.55,
            no_skip_existing=True,
            ref_thresh=0.8,
            classified_path=None,
            classified_max=10,
            output_type="photo",
            segment_length=30,
        )
        called = {}

        def fake_crop_folder(*a, **kw):
            called["args"] = a
            called["kwargs"] = kw
            return {"videos_processed": 1, "frames_processed": 2, "faces": 3, "persons": 1, "ref_photos": 0}

        monkeypatch.setattr(mod, "parse_args", lambda _argv=None: args)
        crop_stub = types.SimpleNamespace(
            crop_folder=fake_crop_folder,
            crop_video=None,
            SUPPORTED_VIDEO_EXTS={".mp4", ".mov"},
        )
        monkeypatch.setitem(sys.modules, "vicrop.crop", crop_stub)

        import face_ops
        from unittest.mock import MagicMock
        monkeypatch.setattr(face_ops, "backend_for_model", lambda *a, **kw: MagicMock())
        mod.main([])
        assert called["kwargs"]["skip_existing"] is False
        assert called["kwargs"]["classify"] is True

    def test_main_calls_crop_video_for_single_file(self, monkeypatch, tmp_path):
        mod = importlib.import_module("vicrop.cli")
        video_file = tmp_path / "clip.mp4"
        video_file.write_bytes(b"fake")
        args = types.SimpleNamespace(
            input=video_file,
            output_dir=tmp_path / "out",
            every_n=10,
            margin_ratio=0.5,
            crop_size=256,
            detection_model="hog",
            no_classify=False,
            tolerance=0.6,
            no_skip_existing=False,
            ref_thresh=0.8,
            classified_path=None,
            classified_max=10,
            output_type="photo",
            segment_length=30,
        )
        called = {}

        def fake_crop_video(*a, **kw):
            called["path"] = a[0]
            called["kwargs"] = kw
            return {"frames_processed": 2, "faces": 1, "persons": 1, "ref_photos": 0}

        monkeypatch.setattr(mod, "parse_args", lambda _argv=None: args)
        crop_stub = types.SimpleNamespace(
            crop_folder=None,
            crop_video=fake_crop_video,
            SUPPORTED_VIDEO_EXTS={".mp4", ".mov"},
        )
        monkeypatch.setitem(sys.modules, "vicrop.crop", crop_stub)

        import face_ops
        from unittest.mock import MagicMock
        monkeypatch.setattr(face_ops, "backend_for_model", lambda *a, **kw: MagicMock())
        mod.main([])
        assert called["path"] == video_file
        assert called["kwargs"]["classify"] is True

    def test_main_rejects_unsupported_file_extension(self, monkeypatch, tmp_path):
        mod = importlib.import_module("vicrop.cli")
        bad_file = tmp_path / "image.jpg"
        bad_file.write_bytes(b"fake")
        args = types.SimpleNamespace(
            input=bad_file,
            output_dir=tmp_path / "out",
            every_n=30,
            margin_ratio=0.4,
            crop_size=1024,
            detection_model="hog",
            no_classify=False,
            tolerance=0.6,
            no_skip_existing=False,
            ref_thresh=0.8,
            classified_path=None,
            classified_max=10,
        )
        monkeypatch.setattr(mod, "parse_args", lambda _argv=None: args)
        crop_stub = types.SimpleNamespace(
            crop_folder=None,
            crop_video=None,
            SUPPORTED_VIDEO_EXTS={".mp4", ".mov"},
        )
        monkeypatch.setitem(sys.modules, "vicrop.crop", crop_stub)

        import face_ops
        from unittest.mock import MagicMock
        monkeypatch.setattr(face_ops, "backend_for_model", lambda *a, **kw: MagicMock())
        with pytest.raises(SystemExit):
            mod.main([])


class TestTranscription:
    def test_combine_and_format(self, monkeypatch):
        transformers_stub = types.SimpleNamespace(pipeline=lambda *a, **k: None)
        mod = _import_fresh(monkeypatch, "videsc.audio.transcription", {"transformers": transformers_stub})

        text, segs = mod.combine_transcription_results(
            [
                {
                    "start": 10.0,
                    "result": {
                        "chunks": [
                            {"text": "hello", "timestamp": (0.0, 1.0)},
                            {"text": "world", "timestamp": (1.0, 2.0)},
                        ]
                    },
                }
            ]
        )
        assert text == "hello world"
        assert segs[0]["timestamp"] == (10.0, 11.0)
        rendered = mod.format_transcript_with_timestamps(text, segs)
        assert "Transcript with timestamps:" in rendered
        assert "0:00:10.000 --> 0:00:11.000" in rendered

    def test_transcribe_audio_from_video_empty_model(self, monkeypatch):
        transformers_stub = types.SimpleNamespace(pipeline=lambda *a, **k: None)
        mod = _import_fresh(monkeypatch, "videsc.audio.transcription", {"transformers": transformers_stub})
        args = types.SimpleNamespace(asr_model="   ", max_audio_seconds=30)
        out = mod.transcribe_audio_from_video("/tmp/a.mp4", args, "cuda")
        assert out == (None, [])

    def test_safe_transcribe_segment_missing_file(self, monkeypatch):
        transformers_stub = types.SimpleNamespace(pipeline=lambda *a, **k: None)
        mod = _import_fresh(monkeypatch, "videsc.audio.transcription", {"transformers": transformers_stub})
        assert mod.safe_transcribe_segment("/not/found.mp3", "model") is None

    def test_transcribe_large_video_success(self, monkeypatch, tmp_path):
        transformers_stub = types.SimpleNamespace(pipeline=lambda *a, **k: None)
        mod = _import_fresh(monkeypatch, "videsc.audio.transcription", {"transformers": transformers_stub})
        segdir = tmp_path / "seg"
        segdir.mkdir()
        monkeypatch.setattr(mod.tempfile, "mkdtemp", lambda prefix="": str(segdir))

        def fake_run(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return types.SimpleNamespace(stdout="120.0\n")
            if cmd[0] == "ffmpeg":
                Path(cmd[-1]).write_text("audio")
                return types.SimpleNamespace(stdout="")
            raise AssertionError("unexpected subprocess command")

        monkeypatch.setattr(mod.subprocess, "run", fake_run)
        monkeypatch.setattr(
            mod,
            "transcribe_audio_segments",
            lambda segs, model_name, segment_duration: [
                {"start": s["start"], "result": {"text": f"seg-{i}"}} for i, s in enumerate(segs)
            ],
        )

        text, segs = mod.transcribe_large_video(str(tmp_path / "video.mp4"), "asr-model")
        assert "seg-0" in text
        assert isinstance(segs, list)

    def test_transcribe_large_video_handles_subprocess_error(self, monkeypatch):
        transformers_stub = types.SimpleNamespace(pipeline=lambda *a, **k: None)
        mod = _import_fresh(monkeypatch, "videsc.audio.transcription", {"transformers": transformers_stub})

        def boom(*_a, **_k):
            raise mod.subprocess.CalledProcessError(1, "ffprobe")

        monkeypatch.setattr(mod.subprocess, "run", boom)
        assert mod.transcribe_large_video("/tmp/a.mp4", "asr-model") == ("", [])


class TestLoader:
    def test_quant_config_and_reader_env(self, monkeypatch):
        calls = {"bnb": []}

        class BitsAndBytesConfig:
            def __init__(self, **kwargs):
                calls["bnb"].append(kwargs)
                self.kwargs = kwargs

        class FakeModelClass:
            @staticmethod
            def from_pretrained(*_a, **_kw):
                return types.SimpleNamespace(disable_talker=lambda: None)

        class FakeProcessorClass:
            @staticmethod
            def from_pretrained(*_a, **_kw):
                return object()

        torch_stub = types.SimpleNamespace(set_num_threads=lambda _n: None, compile=lambda m, **_k: m)
        transformers_stub = types.SimpleNamespace(
            Qwen3VLForConditionalGeneration=FakeModelClass,
            Qwen3VLMoeForConditionalGeneration=FakeModelClass,
            AutoProcessor=FakeProcessorClass,
            AutoModelForMultimodalLM=FakeModelClass,
            BitsAndBytesConfig=BitsAndBytesConfig,
            Qwen3OmniMoeForConditionalGeneration=FakeModelClass,
            Qwen3OmniMoeProcessor=FakeProcessorClass,
            Qwen3_5ForConditionalGeneration=FakeModelClass,
        )
        loader = _import_fresh(
            monkeypatch,
            "videsc.model.loader",
            {
                "torch": torch_stub,
                "torchsummary": types.SimpleNamespace(summary=lambda *_a, **_k: None),
                "transformers": transformers_stub,
            },
        )

        assert loader._quant_config("none") is None
        assert loader._quant_config("8bit").kwargs == {"load_in_8bit": True}
        assert loader._quant_config("4bit").kwargs == {"load_in_4bit": True}

        os.environ.pop("FORCE_QWENVL_VIDEO_READER", None)
        loader._maybe_set_reader("decord")
        assert os.environ["FORCE_QWENVL_VIDEO_READER"] == "decord"

    def test_load_model_and_processor_cache(self, monkeypatch):
        calls = {"model": 0, "processor": 0}

        class FakeModelClass:
            @staticmethod
            def from_pretrained(*_a, **_kw):
                calls["model"] += 1
                return object()

        class FakeProcessorClass:
            @staticmethod
            def from_pretrained(*_a, **_kw):
                calls["processor"] += 1
                return object()

        torch_stub = types.SimpleNamespace(set_num_threads=lambda _n: None, compile=lambda m, **_k: m)
        transformers_stub = types.SimpleNamespace(
            Qwen3VLForConditionalGeneration=FakeModelClass,
            Qwen3VLMoeForConditionalGeneration=FakeModelClass,
            AutoProcessor=FakeProcessorClass,
            AutoModelForMultimodalLM=FakeModelClass,
            BitsAndBytesConfig=lambda **_k: object(),
            Qwen3OmniMoeForConditionalGeneration=FakeModelClass,
            Qwen3OmniMoeProcessor=FakeProcessorClass,
            Qwen3_5ForConditionalGeneration=FakeModelClass,
        )
        loader = _import_fresh(
            monkeypatch,
            "videsc.model.loader",
            {
                "torch": torch_stub,
                "torchsummary": types.SimpleNamespace(summary=lambda *_a, **_k: None),
                "transformers": transformers_stub,
            },
        )
        loader._SHARED_MODEL = None
        loader._SHARED_PROCESSOR = None
        args = types.SimpleNamespace(
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
        loader.load_model_and_processor(args)
        loader.load_model_and_processor(args)
        assert calls == {"model": 1, "processor": 1}

    def test_load_omni_and_qwen35_paths(self, monkeypatch):
        class FakeModelClass:
            @staticmethod
            def from_pretrained(*_a, **_kw):
                return types.SimpleNamespace(disable_talker=lambda: None)

        class FakeProcessorClass:
            @staticmethod
            def from_pretrained(*_a, **_kw):
                return object()

        torch_stub = types.SimpleNamespace(set_num_threads=lambda _n: None, compile=lambda m, **_k: m)
        transformers_stub = types.SimpleNamespace(
            Qwen3VLForConditionalGeneration=FakeModelClass,
            Qwen3VLMoeForConditionalGeneration=FakeModelClass,
            AutoProcessor=FakeProcessorClass,
            AutoModelForMultimodalLM=FakeModelClass,
            BitsAndBytesConfig=lambda **_k: object(),
            Qwen3OmniMoeForConditionalGeneration=FakeModelClass,
            Qwen3OmniMoeProcessor=FakeProcessorClass,
            Qwen3_5ForConditionalGeneration=FakeModelClass,
        )
        loader = _import_fresh(
            monkeypatch,
            "videsc.model.loader",
            {
                "torch": torch_stub,
                "torchsummary": types.SimpleNamespace(summary=lambda *_a, **_k: None),
                "transformers": transformers_stub,
            },
        )
        args = types.SimpleNamespace(
            model_hf=True,
            model_full=False,
            model="Qwen/Qwen3.5-4B",
            half_cpu=False,
            quant="none",
            reader="auto",
            attn="sdpa",
            optimize=False,
            min_pixels=128,
            max_pixels=256,
        )
        loader._SHARED_MODEL = None
        loader._SHARED_PROCESSOR = None
        m1, p1 = loader.load_omni_model_and_processor(args)
        assert m1 is not None and p1 is not None
        loader._SHARED_MODEL = None
        loader._SHARED_PROCESSOR = None
        m2, p2 = loader.load_qwen35_model_and_processor(args)
        assert m2 is not None and p2 is not None


class TestVidescMainRuntime:
    def _import_main(self, monkeypatch):
        return _import_fresh(monkeypatch, "videsc.main")

    def test_main_dispatch(self, monkeypatch):
        mod = self._import_main(monkeypatch)
        monkeypatch.setitem(sys.modules, "videsc.cli.args", types.SimpleNamespace(parse_args=lambda _a=None: types.SimpleNamespace(vl=True, vllm=False)))
        monkeypatch.setattr(mod, "_run_vl", lambda _a: 9)
        monkeypatch.setattr(mod, "_run_wd14", lambda _a: 3)
        assert mod.main([]) == 9

    def test_run_wd14_basic_paths(self, monkeypatch, tmp_path):
        mod = self._import_main(monkeypatch)
        bad = types.SimpleNamespace(input_dir=None, youtube_url=None)
        assert mod._run_wd14(bad) == 1

        args = types.SimpleNamespace(input_dir=tmp_path, youtube_url=None, output_dir=None, every_n=1, max_frames=2, prefix="", threshold=0.2, model_repo="m", include_ratings=False, no_skip_existing=False)
        called = {}
        monkeypatch.setitem(
            sys.modules,
            "videsc.describe",
            types.SimpleNamespace(describe_folder=lambda *_a, **_k: called.setdefault("ok", {"described": 1, "skipped": 0})),
        )
        assert mod._run_wd14(args) == 0
        assert called["ok"]["described"] == 1

    def test_run_vl_batch_and_single(self, monkeypatch):
        mod = self._import_main(monkeypatch)
        dummy_args = types.SimpleNamespace(videos=["*.mp4"], indir=None, filelist=None, youtube_url=None)
        monkeypatch.setitem(
            sys.modules,
            "videsc.pipeline.runner",
            types.SimpleNamespace(run_batch=lambda _a: 5, run_single_video=lambda *_a: 0, run_single_video_gemma4=lambda *_a: 0),
        )
        monkeypatch.setitem(
            sys.modules,
            "videsc.model.loader",
            types.SimpleNamespace(
                load_model_and_processor=lambda _a: ("m", "p"),
                load_omni_model_and_processor=lambda _a: ("om", "op"),
                load_qwen35_model_and_processor=lambda _a: ("35m", "35p"),
                load_gemma4_model_and_processor=lambda _a: ("g4m", "g4p"),
            ),
        )
        assert mod._run_vl(dummy_args) == 5

        single = types.SimpleNamespace(
            videos=None, indir=None, filelist=None, youtube_url=None, omni=False, qwen35=False, gemma4=False
        )
        assert mod._run_vl(single) == 0

    def test_run_wd14_youtube_validation(self, monkeypatch):
        mod = self._import_main(monkeypatch)
        args = types.SimpleNamespace(input_dir=None, youtube_url="http://yt", youtube_api_key=None)
        assert mod._run_wd14(args) == 1

    def test_run_vl_youtube_requires_api_key(self, monkeypatch):
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
                load_gemma4_model_and_processor=lambda _a: ("g4m", "g4p"),
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "videsc.describe",
            types.SimpleNamespace(_download_youtube_video=lambda *_a, **_k: None),
        )
        args = types.SimpleNamespace(
            videos=None,
            indir=None,
            filelist=None,
            youtube_url="http://yt",
            youtube_api_key=None,
            outdir=None,
            output_dir=None,
        )
        assert mod._run_vl(args) == 1


class TestRunner:
    def _import_runner(self, monkeypatch):
        helper_mod = types.SimpleNamespace(
            expand_inputs=lambda *_a, **_k: [],
            namespace_to_cli=lambda *_a, **_k: [],
            _patch_size_for_model=lambda _m: 32,
            expand_video_grid_thw=lambda _i: None,
        )
        stubs = {
            "torch": types.SimpleNamespace(manual_seed=lambda _n: None),
            "qwen_omni_utils": types.SimpleNamespace(process_mm_info=lambda *_a, **_k: (None, None, None)),
            "qwen_vl_utils": types.SimpleNamespace(process_vision_info=lambda *_a, **_k: (None, None, {})),
            "videsc.model.loader": types.SimpleNamespace(
                load_model_and_processor=lambda _a: ("m", "p"),
                load_omni_model_and_processor=lambda _a: ("m", "p"),
                load_qwen35_model_and_processor=lambda _a: ("m", "p"),
                load_gemma4_model_and_processor=lambda _a: ("g4m", "g4p"),
                _maybe_set_reader=lambda _r: None,
            ),
            "videsc.audio.transcription": types.SimpleNamespace(transcribe_audio_from_video=lambda *_a, **_k: (None, [])),
            "videsc.video.info": types.SimpleNamespace(get_video_info=lambda _v: {"tot_time": 1.0, "FPS": 1.0, "num_frames": 1}),
            "videsc.video.sampling": types.SimpleNamespace(
                compute_effective_nframes=lambda *_a, **_k: 1,
                compress_audio_segments_to_nframes=lambda s, *_a, **_k: s,
            ),
            "videsc.video.messages": types.SimpleNamespace(build_messages=lambda **_k: []),
            "videsc.utils.helpers": helper_mod,
        }
        return _import_fresh(monkeypatch, "videsc.pipeline.runner", stubs)

    def test_run_batch_dispatch(self, monkeypatch):
        runner = self._import_runner(monkeypatch)
        monkeypatch.setattr(runner, "run_batch_subprocess", lambda _a: 11)
        monkeypatch.setattr(runner, "run_batch_threads", lambda _a: 22)
        assert runner.run_batch(types.SimpleNamespace(batch_mode="subprocess")) == 11
        assert runner.run_batch(types.SimpleNamespace(batch_mode="threads")) == 22

    def test_run_batch_subprocess_no_inputs(self, monkeypatch):
        runner = self._import_runner(monkeypatch)
        monkeypatch.setattr(runner, "expand_inputs", lambda *_a, **_k: [])
        args = types.SimpleNamespace(videos=None, indir=None, ext=[], filelist=None, workers=1, sleep=0.0, dry_run=True)
        assert runner.run_batch_subprocess(args) == 3

    def test_run_batch_subprocess_dry_run(self, monkeypatch, tmp_path):
        runner = self._import_runner(monkeypatch)
        vid = tmp_path / "clip.mp4"
        vid.write_text("x")
        monkeypatch.setattr(runner, "expand_inputs", lambda *_a, **_k: [vid])
        monkeypatch.setattr(runner, "namespace_to_cli", lambda *_a, **_k: [])
        args = types.SimpleNamespace(
            videos=None,
            indir=None,
            ext=[],
            filelist=None,
            workers=1,
            sleep=0.0,
            dry_run=True,
            prompt="hi",
        )
        assert runner.run_batch_subprocess(args) == 0

    def test_run_batch_threads_no_inputs_and_dry_run(self, monkeypatch, tmp_path):
        runner = self._import_runner(monkeypatch)
        monkeypatch.setattr(runner, "expand_inputs", lambda *_a, **_k: [])
        args = types.SimpleNamespace(videos=None, indir=None, ext=[], filelist=None, dry_run=False, workers=1, omni=False, qwen35=False, gemma4=False)
        assert runner.run_batch_threads(args) == 3

        vid = tmp_path / "v.mp4"
        vid.write_text("x")
        monkeypatch.setattr(runner, "expand_inputs", lambda *_a, **_k: [vid])
        args2 = types.SimpleNamespace(videos=None, indir=None, ext=[], filelist=None, dry_run=True, workers=1, omni=False, qwen35=False, gemma4=False)
        assert runner.run_batch_threads(args2) == 0

    def test_run_batch_threads_executes_futures(self, monkeypatch, tmp_path):
        runner = self._import_runner(monkeypatch)
        v1 = tmp_path / "a.mp4"
        v2 = tmp_path / "b.mp4"
        v1.write_text("x")
        v2.write_text("x")
        monkeypatch.setattr(runner, "expand_inputs", lambda *_a, **_k: [v1, v2])
        monkeypatch.setattr(runner, "run_single_video", lambda *_a, **_k: 0)
        args = types.SimpleNamespace(videos=None, indir=None, ext=[], filelist=None, dry_run=False, workers=2, omni=False, qwen35=False, gemma4=False)
        assert runner.run_batch_threads(args) == 0

    def test_run_batch_subprocess_executes_processes(self, monkeypatch, tmp_path):
        runner = self._import_runner(monkeypatch)
        vid = tmp_path / "clip.mp4"
        vid.write_text("x")
        monkeypatch.setattr(runner, "expand_inputs", lambda *_a, **_k: [vid])
        monkeypatch.setattr(runner, "namespace_to_cli", lambda *_a, **_k: [])
        monkeypatch.setattr(runner.time, "sleep", lambda _s: None)

        class P:
            def __init__(self):
                self.pid = 123
                self._n = 0

            def poll(self):
                self._n += 1
                return 0 if self._n > 1 else None

        monkeypatch.setattr(runner.subprocess, "Popen", lambda _cmd: P())
        args = types.SimpleNamespace(
            videos=None,
            indir=None,
            ext=[],
            filelist=None,
            workers=1,
            sleep=0.0,
            dry_run=False,
            prompt=None,
        )
        assert runner.run_batch_subprocess(args) == 0

    def test_run_single_video_non_omni_dry_and_transcript(self, monkeypatch, tmp_path):
        runner = self._import_runner(monkeypatch)
        monkeypatch.setattr(runner, "get_video_info", lambda _v: {"tot_time": 2.0, "FPS": 30.0, "num_frames": 60})
        monkeypatch.setattr(runner, "compute_effective_nframes", lambda *_a, **_k: 2)
        monkeypatch.setattr(
            runner,
            "transcribe_audio_from_video",
            lambda *_a, **_k: ("hello audio", [{"timestamp": (0.0, 1.0), "text": "hello"}]),
        )
        monkeypatch.setattr(runner, "compress_audio_segments_to_nframes", lambda segs, *_a, **_k: segs)
        monkeypatch.setattr(runner, "build_messages", lambda **_k: [{"role": "user", "content": "x"}])
        monkeypatch.setattr(runner, "process_vision_info", lambda *_a, **_k: (None, None, {}))

        class Inputs(dict):
            def __init__(self):
                super().__init__()
                self.input_ids = [[1, 2]]

            def to(self, *_a, **_k):
                return self

        class Processor:
            def apply_chat_template(self, *_a, **_k):
                return "prompt"

            def __call__(self, **_kwargs):
                return Inputs()

            def batch_decode(self, *_a, **_k):
                return ["out"]

        video = tmp_path / "a.mp4"
        video.write_text("x")
        args = types.SimpleNamespace(
            seed=1,
            video=str(video),
            audio=True,
            num_frames=4,
            spf=1.0,
            model="Qwen/Qwen3-VL-8B-Instruct",
            total_pixels=10,
            prompt="p",
            system="s",
            cont_prompt=False,
            omni=False,
            reader="auto",
            no_meta=False,
            dry=True,
            optimize=False,
            max_new_tokens=10,
            rep_pen=1.0,
            no_think_trim=False,
            outdir=str(tmp_path / "out"),
            no_save_transcript=False,
        )
        model = types.SimpleNamespace(device="cuda", dtype="float16")
        rc = runner.run_single_video(args, model, Processor())
        assert rc == 0
        assert (tmp_path / "out" / "a.txt").exists()
        assert (tmp_path / "out" / "a.transcript.txt").exists()

    def test_run_single_video_omni_dry(self, monkeypatch, tmp_path):
        runner = self._import_runner(monkeypatch)
        monkeypatch.setattr(runner, "get_video_info", lambda _v: {"tot_time": 2.0, "FPS": 30.0, "num_frames": 60})
        monkeypatch.setattr(runner, "compute_effective_nframes", lambda *_a, **_k: 2)
        monkeypatch.setattr(runner, "build_messages", lambda **_k: [{"role": "user", "content": "x"}])
        monkeypatch.setattr(runner, "process_mm_info", lambda *_a, **_k: (None, None, None))

        class Processor:
            def apply_chat_template(self, *_a, **_k):
                return "prompt"

            def __call__(self, **_kwargs):
                return types.SimpleNamespace()

            def batch_decode(self, *_a, **_k):
                return ["out"]

        video = tmp_path / "o.mp4"
        video.write_text("x")
        args = types.SimpleNamespace(
            seed=1,
            video=str(video),
            audio=False,
            num_frames=4,
            spf=1.0,
            model="Qwen/Qwen3-Omni",
            total_pixels=10,
            prompt="p",
            system="s",
            cont_prompt=False,
            omni=True,
            reader="auto",
            no_meta=False,
            dry=True,
            optimize=False,
            max_new_tokens=10,
            rep_pen=1.0,
            no_think_trim=False,
            outdir=str(tmp_path / "out2"),
            no_save_transcript=True,
        )
        model = types.SimpleNamespace(device="cuda", dtype="float16")
        rc = runner.run_single_video(args, model, Processor())
        assert rc == 0
        assert (tmp_path / "out2" / "o.txt").exists()

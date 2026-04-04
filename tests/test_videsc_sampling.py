"""
tests/test_videsc_sampling.py

Unit tests for videsc.video.sampling (ported from otherroch/videsc).
"""

from videsc.video.sampling import compute_effective_nframes, compress_audio_segments_to_nframes


def _make_vinfo(fps=30.0, num_frames=900, tot_time=30.0):
    return {"FPS": fps, "num_frames": num_frames, "tot_time": tot_time}


class TestComputeEffectiveNframes:
    def test_no_spf_returns_requested(self):
        vinfo = _make_vinfo(fps=30.0, num_frames=900)
        result = compute_effective_nframes(vinfo, requested_nframes=100, spf=0.0)
        assert result == 100

    def test_spf_divides_frames(self):
        # 900 frames / (4.0 spf * 30 fps) = 900/120 = 7.5 → 7
        vinfo = _make_vinfo(fps=30.0, num_frames=900)
        result = compute_effective_nframes(vinfo, requested_nframes=256, spf=4.0)
        assert result == 7

    def test_capped_at_768(self):
        vinfo = _make_vinfo(fps=30.0, num_frames=90000)
        result = compute_effective_nframes(vinfo, requested_nframes=10000, spf=0.0)
        assert result == 768

    def test_minimum_one_frame(self):
        # Very high spf should still give at least 1 frame
        vinfo = _make_vinfo(fps=30.0, num_frames=30)
        result = compute_effective_nframes(vinfo, requested_nframes=256, spf=100.0)
        assert result >= 1


class TestCompressAudioSegmentsToNframes:
    def _make_segment(self, start, end, text):
        return {"timestamp": (start, end), "text": text}

    def test_empty_segments_returned_unchanged(self):
        result = compress_audio_segments_to_nframes([], nframes=10, video_duration=30.0)
        assert result == []

    def test_output_length_equals_nframes(self):
        segs = [self._make_segment(0, 5, "hello"), self._make_segment(5, 10, "world")]
        result = compress_audio_segments_to_nframes(segs, nframes=4, video_duration=10.0)
        assert len(result) == 4

    def test_text_assigned_to_correct_bucket(self):
        # Single segment covers the full duration; all buckets should have text
        segs = [self._make_segment(0.0, 10.0, "spoken")]
        result = compress_audio_segments_to_nframes(segs, nframes=2, video_duration=10.0)
        assert all(r["text"] for r in result)

    def test_empty_buckets_have_empty_text(self):
        # Segment only in first half — second bucket should be empty
        segs = [self._make_segment(0.0, 4.9, "early")]
        result = compress_audio_segments_to_nframes(segs, nframes=2, video_duration=10.0)
        assert result[0]["text"] == "early"
        assert result[1]["text"] == ""

    def test_timestamps_cover_full_duration(self):
        segs = [self._make_segment(0, 10, "x")]
        result = compress_audio_segments_to_nframes(segs, nframes=3, video_duration=10.0)
        assert result[0]["timestamp"][0] == 0.0
        assert result[-1]["timestamp"][1] == 10.0

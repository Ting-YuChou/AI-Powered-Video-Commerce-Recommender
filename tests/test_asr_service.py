from types import SimpleNamespace

from asr_service.api import build_transcription_payload, group_alignment_items


def _item(text, start, end):
    return SimpleNamespace(text=text, start_time=start, end_time=end)


def test_group_alignment_items_splits_on_pause_and_duration():
    segments = group_alignment_items(
        [
            _item("新品", 0.0, 1.0),
            _item("手機", 1.1, 2.0),
            _item("特價", 3.0, 4.0),
            _item("today", 4.1, 8.5),
        ],
        pause_seconds=0.8,
        max_duration_seconds=5.0,
        max_chars=96,
        max_segments=64,
    )

    assert segments == [
        {"text": "新品手機", "start_seconds": 0.0, "end_seconds": 2.0},
        {"text": "特價", "start_seconds": 3.0, "end_seconds": 4.0},
        {"text": "today", "start_seconds": 4.1, "end_seconds": 8.5},
    ]


def test_group_alignment_items_merges_to_segment_cap_without_losing_coverage():
    segments = group_alignment_items(
        [_item(str(index), float(index), float(index) + 0.2) for index in range(8)],
        pause_seconds=0.1,
        max_duration_seconds=1.0,
        max_chars=1,
        max_segments=3,
    )

    assert len(segments) == 3
    assert segments[0]["start_seconds"] == 0.0
    assert segments[-1]["end_seconds"] == 7.2
    assert "".join(segment["text"] for segment in segments) == "01234567"


def test_transcription_payload_preserves_text_when_alignment_is_missing():
    payload = build_transcription_payload(
        SimpleNamespace(text="新品手機", language="Chinese", time_stamps=None),
        model="Qwen/Qwen3-ASR-0.6B",
    )

    assert payload["text"] == "新品手機"
    assert payload["alignment_status"] == "degraded"
    assert payload["segments"] == []


def test_transcription_payload_serializes_aligned_segments():
    payload = build_transcription_payload(
        SimpleNamespace(
            text="new phone",
            language="English",
            time_stamps=SimpleNamespace(
                items=[_item("new", 0.0, 0.4), _item("phone", 0.5, 1.0)]
            ),
        ),
        model="Qwen/Qwen3-ASR-0.6B",
    )

    assert payload["alignment_status"] == "completed"
    assert payload["segments"] == [
        {"text": "new phone", "start_seconds": 0.0, "end_seconds": 1.0}
    ]

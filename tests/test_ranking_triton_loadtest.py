from video_commerce.loadtest_ranking_ab import build_run_plan


def test_ranking_ab_plan_interleaves_backends_and_preserves_fixed_conditions(tmp_path):
    plan = build_run_plan(
        legacy_url="http://legacy:8003",
        triton_url="http://triton:8003",
        rates=[500, 1000],
        duration="5m",
        repetitions=3,
        output_dir=tmp_path,
    )

    assert len(plan) == 12
    assert [run.backend for run in plan[:4]] == [
        "legacy",
        "triton",
        "triton",
        "legacy",
    ]
    assert all(run.duration == "5m" for run in plan)
    assert {run.rate for run in plan} == {500, 1000}
    assert len({run.output_path for run in plan}) == len(plan)

from scripts.model_release import build_parser


def test_model_release_cli_requires_auditable_promotion_arguments():
    args = build_parser().parse_args(
        [
            "promote",
            "--version",
            "ranking-v2",
            "--expected-generation",
            "3",
            "--actor",
            "release-manager",
            "--reason",
            "offline gate passed",
        ]
    )

    assert args.command == "promote"
    assert args.expected_generation == 3
    assert args.actor == "release-manager"


def test_model_release_cli_supports_status_evaluate_rollback_and_bootstrap():
    parser = build_parser()

    assert parser.parse_args(["status", "--json"]).as_json is True
    assert parser.parse_args(["evaluate", "--version", "v1"]).version == "v1"
    assert (
        parser.parse_args(
            [
                "rollback",
                "--version",
                "v1",
                "--expected-generation",
                "4",
                "--actor",
                "operator",
                "--reason",
                "regression",
            ]
        ).command
        == "rollback"
    )
    assert (
        parser.parse_args(
            [
                "bootstrap-active",
                "--version",
                "v1",
                "--actor",
                "operator",
                "--reason",
                "initial verified release",
            ]
        ).command
        == "bootstrap-active"
    )


def test_model_release_cli_accepts_two_tower_family():
    args = build_parser().parse_args(
        ["--model-name", "two_tower_retrieval", "status", "--json"]
    )

    assert args.model_name == "two_tower_retrieval"
    assert args.command == "status"
    assert args.as_json is True


def test_model_release_cli_supports_rollout_status_and_wait():
    parser = build_parser()
    status = parser.parse_args(["rollout-status", "--generation", "7", "--json"])
    wait = parser.parse_args(
        [
            "wait-rollout",
            "--generation",
            "7",
            "--expected-service-count",
            "ranking-runner=2",
            "--expected-service-count",
            "ranking-service=1",
            "--timeout",
            "300",
        ]
    )

    assert status.generation == 7
    assert status.as_json is True
    assert wait.expected_service_count == ["ranking-runner=2", "ranking-service=1"]

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from hyrox.action_names import HYROX_ACTION_NAMES
from hyrox.registry import create_action_analyzer
from src.biomechanics.session_writer import SessionConfig, SessionWriter
from src.output_schema import artifact_metadata
from webui import create_app


def main() -> int:
    checks: dict[str, object] = {
        **artifact_metadata("no_camera_smoke_report"),
        "camera_opened": False,
    }
    for action_name in HYROX_ACTION_NAMES:
        create_action_analyzer(action_name)
    checks["hyrox_analyzers"] = len(HYROX_ACTION_NAMES)

    with tempfile.TemporaryDirectory(prefix="pose_smoke_") as temporary:
        writer = SessionWriter(Path(temporary))
        writer.start(
            SessionConfig(
                camera_index=-1,
                width=640,
                height=480,
                mirror=False,
                smoothing=0.0,
                model_name="smoke-test",
                plot_on_save=False,
            ),
            session_id="smoke",
        )
        session_dir = writer.stop()
        checks["session_output"] = bool(
            session_dir is not None
            and (session_dir / "metadata.json").is_file()
        )

    response = create_app().test_client().get("/healthz")
    checks["web_health"] = response.status_code == 200
    checks["ready"] = all(
        (
            checks["hyrox_analyzers"] == len(HYROX_ACTION_NAMES),
            checks["session_output"],
            checks["web_health"],
        )
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if checks["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

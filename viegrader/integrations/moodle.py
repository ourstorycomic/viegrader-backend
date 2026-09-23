from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional


@dataclass
class MoodleConfig:
    base_url: str
    token: str
    timeout: int = 30
    verify_ssl: bool = True

    @classmethod
    def from_env(cls) -> "MoodleConfig":
        url, token = os.environ.get("MOODLE_BASE_URL", ""), os.environ.get("MOODLE_TOKEN", "")
        if not url or not token:
            raise ValueError("Cần khai báo MOODLE_BASE_URL và MOODLE_TOKEN.")
        return cls(url, token)


class MoodleClient:
    """REST client nhỏ, không ghi điểm nếu chưa truyền ``approved=True``."""

    def __init__(self, cfg: MoodleConfig):
        self.cfg = cfg

    def call(self, function: str, params: Optional[Dict[str, Any]] = None) -> Any:
        try:
            import requests
        except ImportError as exc:
            raise ImportError("Cài viegrader[moodle] để kết nối Moodle.") from exc
        payload = {
            "wstoken": self.cfg.token, "wsfunction": function,
            "moodlewsrestformat": "json", **(params or {}),
        }
        response = requests.post(
            f"{self.cfg.base_url.rstrip('/')}/webservice/rest/server.php",
            data=payload, timeout=self.cfg.timeout, verify=self.cfg.verify_ssl,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and ("exception" in data or "errorcode" in data):
            raise RuntimeError(f"Moodle API lỗi: {data.get('message', data)}")
        return data

    def get_assignments(self, course_id: int) -> Any:
        return self.call("mod_assign_get_assignments", {"courseids[0]": course_id})

    def get_submissions(self, assignment_id: int) -> Any:
        return self.call("mod_assign_get_submissions", {"assignmentids[0]": assignment_id})

    def save_grade(
        self, assignment_id: int, user_id: int, grade: float, feedback: str = "",
        approved: bool = False, attempt_number: int = -1,
    ) -> Any:
        if not approved:
            raise PermissionError("Điểm phải được giảng viên phê duyệt trước khi gửi Moodle.")
        params = {
            "assignmentid": assignment_id, "userid": user_id, "grade": grade,
            "attemptnumber": attempt_number, "addattempt": 0, "workflowstate": "released",
            "applytoall": 0,
            "plugindata[assignfeedbackcomments_editor][text]": feedback,
            "plugindata[assignfeedbackcomments_editor][format]": 1,
        }
        return self.call("mod_assign_save_grade", params)

"""Exercise upload -> VieGrader service -> instructor approval -> Moodle simulation."""

import base64
import time

import pandas as pd
from fastapi.testclient import TestClient


def test_portal_workflow(tmp_path, monkeypatch):
    from viegrader import teacher_portal as portal

    monkeypatch.setattr(portal, "ROOT", tmp_path / "portal")
    monkeypatch.setattr(portal, "RUBRICS", tmp_path / "rubrics")
    monkeypatch.setattr(portal, "BASELINE", tmp_path / "model.pkl")
    monkeypatch.setenv("VIEGRADER_PORTAL_USER", "giangvien")
    monkeypatch.setenv("VIEGRADER_PORTAL_PASSWORD", "secret-test")
    monkeypatch.setenv("VIEGRADER_MOODLE_MODE", "simulate")
    portal.RUBRICS.mkdir()
    (portal.RUBRICS / "rubric_de_1.yaml").write_text("criteria: []\n", encoding="utf-8")
    portal.BASELINE.write_bytes(b"fixture")

    def fake_grade(model_path, input_path, rubric_dir, output_dir, **kwargs):
        frame = pd.read_csv(input_path)
        assert frame.loc[0, "prompt_text"] == "Đề thi Toán rời rạc 1"
        output_dir.mkdir(parents=True)
        pd.DataFrame([{"essay_id": frame.loc[0, "essay_id"], "final_total": 6.5,
                       "flags": "LOW_SECTION_COVERAGE"}]).to_csv(output_dir / "scores.csv", index=False)
        return {"n_scored": 1}

    monkeypatch.setattr(portal, "grade_with_rubric", fake_grade)
    auth = "Basic " + base64.b64encode(b"giangvien:secret-test").decode()
    with TestClient(portal.app) as client:
        assert client.get("/api/v1/exams").status_code == 401
        headers = {"Authorization": auth}
        exam = client.post("/api/v1/exams", headers=headers, data={"exam_id": "De_1"}, files={
            "question": ("de.txt", "Đề thi Toán rời rạc 1".encode("utf-8")),
            "answer": ("dapan.txt", b"Dap an day du cho de thi 1"),
        })
        assert exam.status_code == 200, exam.text
        exam_id = exam.json()["id"]
        job = client.post("/api/v1/jobs", headers=headers, data={"exam_ref": exam_id, "mode": "hybrid"},
                          files=[("files", ("bai_1.txt", b"Bai lam sinh vien co cac buoc giai thich va ket luan."))])
        assert job.status_code == 200, job.text
        job_id = job.json()["job_id"]
        for _ in range(100):
            result = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
            if result["job"]["state"] != "queued" and result["job"]["state"] != "running":
                break
            time.sleep(.01)
        assert result["job"]["state"] == "ready", result["job"]["error"]
        essay = result["essays"][0]
        assert essay["proposed"] == 6.5 and essay["review_needed"] == 1
        key = essay["id"]
        review = client.get(f"/api/v1/essays/{key}", headers=headers)
        assert review.status_code == 200
        assert "Bai lam sinh vien" in review.json()["essay"]["text"]
        assert review.json()["resources"]["question"] == "Đề thi Toán rời rạc 1"
        assert review.json()["resources"]["rubric_sha256"]
        queue = client.get(f"/api/v1/jobs/{job_id}/review-queue", headers=headers)
        assert queue.status_code == 200 and queue.json()[0]["id"] == key
        assert client.post(f"/api/v1/essays/{key}/moodle-push", headers=headers, json={"assignment_id": 7}).status_code == 403
        assert client.post(f"/api/v1/essays/{key}/approve", headers=headers,
                           json={"score": 6.3, "moodle_user_id": 12}).status_code == 422
        approved = client.post(f"/api/v1/essays/{key}/approve", headers=headers,
                               json={"score": 6.75, "moodle_user_id": 12,
                                     "comment": "Đúng hướng giải; cần trình bày rõ bước quy nạp."})
        assert approved.status_code == 200, approved.text
        assert approved.json()["reviewed_by"] == "giangvien"
        assert approved.json()["review_status"] == "approved"
        assert "quy nạp" in approved.json()["teacher_comment"]
        delivered = client.post(f"/api/v1/essays/{key}/moodle-push", headers=headers,
                                json={"assignment_id": 7})
        assert delivered.status_code == 200 and delivered.json()["mode"] == "simulate"
        assert delivered.json()["grade"] == 6.75
        assert client.post(f"/api/v1/essays/{key}/moodle-push", headers=headers,
                           json={"assignment_id": 7}).status_code == 409
        assert "6.75" in client.get(f"/api/v1/jobs/{job_id}/results.csv", headers=headers).text
        summary = client.get("/api/v1/reports/summary", headers=headers).json()
        assert summary["totals"]["essays"] == 1 and summary["totals"]["approved"] == 1
        audit = client.get("/api/v1/audit", headers=headers).json()
        assert audit[0]["actor"] == "giangvien" and audit[0]["action"] == "approve"

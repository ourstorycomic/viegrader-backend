from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd


def create_app(grader_loader: Callable[[], object], moodle_client=None):
    """Tạo FastAPI app; trì hoãn nạp model đến request đầu tiên."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError("Cài viegrader[api] để chạy dịch vụ.") from exc

    class EssayRequest(BaseModel):
        essay_id: str
        text: str = Field(min_length=1)
        prompt_id: str = ""
        prompt_text: str = ""

    class MoodlePushRequest(BaseModel):
        assignment_id: int
        user_id: int
        grade: float
        feedback: str = ""
        approved: bool = False

    app = FastAPI(title="VieGrader API", version="0.5.1")
    state: Dict[str, Any] = {"grader": None}

    def grader():
        if state["grader"] is None:
            state["grader"] = grader_loader()
        return state["grader"]

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": state["grader"] is not None}

    @app.post("/v1/score")
    def score(item: EssayRequest):
        try:
            frame = pd.DataFrame([item.model_dump()])
            return grader().score_to_frame(frame).iloc[0].to_dict()
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/batch-score")
    def batch_score(items: List[EssayRequest]):
        try:
            frame = pd.DataFrame([item.model_dump() for item in items])
            return grader().score_to_frame(frame).to_dict(orient="records")
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/moodle/push-grade")
    def push_grade(item: MoodlePushRequest):
        if moodle_client is None:
            raise HTTPException(status_code=503, detail="Chưa cấu hình Moodle.")
        try:
            return moodle_client.save_grade(**item.model_dump())
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    return app

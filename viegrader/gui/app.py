from __future__ import annotations

import os

from .handlers import (
    ablation_gui, build_rag_gui, evaluate_gui, export_reports_gui, inspect_environment, moodle_push_gui,
    preview_table, score_gui, sus_gui, train_baseline, train_qlora_hf,
)


CSS = """
.vg-title {text-align:center; margin-bottom:0.25rem}
.vg-note {border-left:4px solid #2563eb; padding:0.75rem; background:#eff6ff}
.vg-warn {border-left:4px solid #d97706; padding:0.75rem; background:#fffbeb}
footer {display:none !important}
"""


def build_gui():
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError("Cài viegrader[gui] để chạy giao diện Gradio.") from exc

    qlora_length = int(os.getenv("VIEGRADER_QLORA_MAX_LENGTH", "1024"))
    qlora_batch = int(os.getenv("VIEGRADER_QLORA_BATCH_SIZE", "1"))
    qlora_accum = int(os.getenv("VIEGRADER_QLORA_GRAD_ACCUM", "16"))

    def baseline_action(data, gold, rubric, encoder, test_size, progress=gr.Progress()):
        return train_baseline(data, gold, rubric, encoder, test_size, progress)

    def qlora_action(data, gold, rubric, model, repo, private, epochs, length,
                     batch, accum, push, progress=gr.Progress()):
        return train_qlora_hf(
            data, gold, rubric, model, repo, private, epochs, length,
            batch, accum, push, progress,
        )

    with gr.Blocks(title="VieGrader 0.5 Ubuntu RTX") as demo:
        gr.Markdown("# VieGrader 0.5 — Ubuntu Server & RTX 5060 Ti", elem_classes="vg-title")
        gr.Markdown(
            "GUI nghiên cứu: làm sạch/gán nhãn bằng CLI, huấn luyện baseline hoặc "
            "Vistral‑7B QLoRA, RAG, chấm điểm, đánh giá, SUS và Moodle."
        )

        with gr.Tabs():
            with gr.Tab("Tổng quan"):
                gr.Markdown(
                    "<div class='vg-note'>Token Hugging Face và Moodle phải được lưu "
                    "trong Secrets/biến môi trường; giao diện không yêu cầu nhập token trực tiếp.</div>"
                )
                env_out = gr.Textbox(label="Môi trường", lines=8, interactive=False)
                env_btn = gr.Button("Kiểm tra GPU và Hugging Face", variant="primary")
                env_btn.click(inspect_environment, outputs=env_out)

            with gr.Tab("Dữ liệu"):
                data_preview_file = gr.File(label="CSV/XLSX cần xem", type="filepath")
                preview_limit = gr.Slider(5, 100, value=20, step=5, label="Số dòng xem trước")
                preview_btn = gr.Button("Đọc dữ liệu")
                preview_status = gr.Textbox(label="Thông tin")
                preview_df = gr.Dataframe(label="Xem trước", interactive=False)
                preview_btn.click(preview_table, [data_preview_file, preview_limit], [preview_df, preview_status])

            with gr.Tab("Baseline TF-IDF/PhoBERT"):
                gr.Markdown("Huấn luyện baseline làm đối chứng cho ablation A và B.")
                with gr.Row():
                    baseline_data = gr.File(label="Dữ liệu sạch", type="filepath")
                    baseline_gold = gr.File(label="Nhãn vàng", type="filepath")
                    baseline_rubric = gr.File(label="Rubric YAML (tùy chọn)", type="filepath")
                with gr.Row():
                    baseline_encoder = gr.Dropdown(["tfidf", "phobert"], value="tfidf", label="Encoder")
                    baseline_test = gr.Slider(0.1, 0.4, value=0.2, step=0.05, label="Tỷ lệ test")
                baseline_btn = gr.Button("Huấn luyện baseline", variant="primary")
                baseline_log = gr.Textbox(label="Báo cáo", lines=18)
                with gr.Row():
                    baseline_model = gr.File(label="Mô hình .pkl")
                    baseline_report = gr.File(label="Báo cáo .txt")
                baseline_btn.click(
                    baseline_action,
                    [baseline_data, baseline_gold, baseline_rubric, baseline_encoder, baseline_test],
                    [baseline_log, baseline_model, baseline_report],
                )

            with gr.Tab("QLoRA & Hugging Face"):
                gr.Markdown(
                    "<div class='vg-warn'>Vistral‑7B QLoRA cần GPU CUDA. Trên Hugging "
                    "Face Spaces hãy chọn GPU hardware và tạo Secret <b>HF_TOKEN</b>.</div>"
                )
                with gr.Row():
                    q_data = gr.File(label="Dữ liệu sạch", type="filepath")
                    q_gold = gr.File(label="Nhãn vàng", type="filepath")
                    q_rubric = gr.File(label="Rubric YAML (tùy chọn)", type="filepath")
                with gr.Row():
                    q_model = gr.Textbox(value="Viet-Mistral/Vistral-7B-Chat", label="Base model trên Hub")
                    q_repo = gr.Textbox(placeholder="username/viegrader-vistral-qlora", label="Model repo_id")
                with gr.Row():
                    q_private = gr.Checkbox(value=True, label="Repository riêng tư")
                    q_push = gr.Checkbox(value=True, label="Đẩy adapter lên Hub sau train")
                    q_epochs = gr.Slider(1, 10, value=3, step=1, label="Epoch")
                with gr.Row():
                    q_length = gr.Dropdown([512, 768, 1024, 1280, 1536], value=qlora_length,
                                           label="Max sequence length (preset 16GB)")
                    q_batch = gr.Dropdown([1, 2, 4], value=qlora_batch, label="Batch size")
                    q_accum = gr.Dropdown([1, 2, 4, 8, 16, 32], value=qlora_accum,
                                          label="Gradient accumulation")
                q_btn = gr.Button("Bắt đầu huấn luyện QLoRA", variant="primary")
                q_log = gr.Textbox(label="Nhật ký kết quả", lines=12)
                with gr.Row():
                    q_adapter = gr.File(label="Tải adapter ZIP")
                    q_url = gr.Textbox(label="Hugging Face commit URL", interactive=False)
                q_btn.click(
                    qlora_action,
                    [q_data, q_gold, q_rubric, q_model, q_repo, q_private,
                     q_epochs, q_length, q_batch, q_accum, q_push],
                    [q_log, q_adapter, q_url],
                )

            with gr.Tab("RAG"):
                rag_file = gr.File(label="Tài liệu CSV/XLSX có cột text", type="filepath")
                rag_btn = gr.Button("Lập chỉ mục RAG", variant="primary")
                rag_status = gr.Textbox(label="Kết quả")
                rag_output = gr.File(label="rag_documents.json")
                rag_btn.click(build_rag_gui, rag_file, [rag_status, rag_output])

            with gr.Tab("Chấm điểm"):
                with gr.Row():
                    score_model = gr.File(label="Baseline model .pkl", type="filepath")
                    score_data = gr.File(label="Bài cần chấm", type="filepath")
                score_btn = gr.Button("Chấm bài", variant="primary")
                score_status = gr.Textbox(label="Trạng thái")
                score_preview = gr.Dataframe(label="Kết quả", interactive=False)
                score_output = gr.File(label="Tải kết quả CSV")
                score_btn.click(score_gui, [score_model, score_data], [score_preview, score_output, score_status])

            with gr.Tab("Đánh giá"):
                with gr.Row():
                    eval_pred = gr.File(label="Kết quả dự đoán", type="filepath")
                    eval_gold = gr.File(label="Nhãn vàng test", type="filepath")
                    eval_rubric = gr.File(label="Rubric YAML", type="filepath")
                eval_hh = gr.Number(label="QWK người–người (tùy chọn)")
                eval_btn = gr.Button("Đánh giá", variant="primary")
                eval_log = gr.Textbox(label="QWK/MAE/SMD", lines=18)
                eval_file = gr.File(label="Báo cáo")
                eval_btn.click(evaluate_gui, [eval_pred, eval_gold, eval_rubric, eval_hh], [eval_log, eval_file])

            with gr.Tab("Ablation A–H"):
                gr.Markdown("Xuất ma trận cấu hình chuẩn để chạy cùng một tập test và seed.")
                abl_btn = gr.Button("Tạo cấu hình A–H")
                abl_table = gr.Dataframe(label="Ma trận ablation", interactive=False)
                abl_file = gr.File(label="Tải cấu hình CSV")
                abl_btn.click(ablation_gui, outputs=[abl_table, abl_file])

            with gr.Tab("SUS"):
                sus_file = gr.File(label="Khảo sát có cột sus_1…sus_10", type="filepath")
                sus_btn = gr.Button("Phân tích SUS", variant="primary")
                sus_log = gr.Textbox(label="Kết quả", lines=10)
                sus_output = gr.File(label="Báo cáo JSON")
                sus_btn.click(sus_gui, sus_file, [sus_log, sus_output])

            with gr.Tab("Xuất báo cáo"):
                gr.Markdown(
                    "Tải các kết quả của từng giai đoạn. Hệ thống tự nhận diện theo tên tệp "
                    "và tổng hợp thành báo cáo nghiên cứu thống nhất."
                )
                report_files = gr.File(
                    label="Tệp kết quả CSV/XLSX/JSON/TXT/MD/YAML",
                    type="filepath", file_count="multiple",
                )
                with gr.Row():
                    report_project = gr.Textbox(value="VieGrader – NCKH 2026", label="Tên dự án")
                    report_author = gr.Textbox(label="Tác giả/nhóm nghiên cứu")
                report_formats = gr.CheckboxGroup(
                    ["xlsx", "docx", "pdf", "html", "json"],
                    value=["xlsx", "docx", "html", "json"],
                    label="Định dạng báo cáo",
                )
                report_btn = gr.Button("Tạo báo cáo các giai đoạn", variant="primary")
                report_status = gr.Textbox(label="Trạng thái")
                report_summary = gr.Dataframe(label="Tổng hợp giai đoạn", interactive=False)
                report_bundle = gr.File(label="Tải gói báo cáo ZIP")
                report_btn.click(
                    export_reports_gui,
                    [report_files, report_project, report_author, report_formats],
                    [report_summary, report_bundle, report_status],
                )

            with gr.Tab("Moodle"):
                gr.Markdown(
                    "Cần Secrets `MOODLE_BASE_URL`, `MOODLE_TOKEN`. Điểm chỉ được gửi "
                    "khi đánh dấu xác nhận của giảng viên."
                )
                with gr.Row():
                    moodle_assignment = gr.Number(label="Assignment ID", precision=0)
                    moodle_user = gr.Number(label="User ID", precision=0)
                    moodle_grade = gr.Number(label="Điểm")
                moodle_feedback = gr.Textbox(label="Phản hồi", lines=5)
                moodle_approved = gr.Checkbox(label="Giảng viên đã phê duyệt")
                moodle_btn = gr.Button("Gửi điểm lên Moodle", variant="stop")
                moodle_result = gr.Textbox(label="Kết quả", lines=8)
                moodle_audit = gr.File(label="Nhật ký Moodle đã ẩn danh")
                moodle_btn.click(
                    moodle_push_gui,
                    [moodle_assignment, moodle_user, moodle_grade, moodle_feedback, moodle_approved],
                    [moodle_result, moodle_audit],
                )

    return demo


def launch(**kwargs):
    app = build_gui()
    app.queue(default_concurrency_limit=1)
    kwargs.setdefault("css", CSS)
    app.launch(**kwargs)

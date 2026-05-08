"""Streamlit app for MedEvi — genetic association evidence overview."""

from __future__ import annotations

import io
import csv
import json
import threading
import time
from typing import Any

import streamlit as st

from graph import app as graph_app


def _safe_val(d: dict, key: str, default: Any = "") -> Any:
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _convert_to_csv(rows: list, columns: list) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_safe_val(row, k, "") for k in columns])
    return output.getvalue()


def _auto_generate_picos(snp: str, disease: str) -> dict:
    """Use LLM to auto-generate full query + PICOS from 2 simple inputs."""
    from prompts.picos_generation_prompt import PICOS_GENERATION_PROMPT
    from agents.llm_router import call_chat_model

    prompt = PICOS_GENERATION_PROMPT.format(snp=snp, disease=disease)
    try:
        text = call_chat_model(prompt, temperature=0.0)
        if text:
            data = json.loads(text)
            if isinstance(data, dict) and "query" in data:
                return {
                    "query": data.get("query", f"{snp} polymorphism and {disease}"),
                    "population": data.get("population", f"{disease} patients"),
                    "intervention": data.get("intervention", f"{snp} variant carriers"),
                    "comparison": data.get("comparison", "wild-type homozygous"),
                    "outcome": data.get("outcome", f"{disease} risk"),
                    "study_type": data.get("study_type", "observational studies"),
                }
    except (json.JSONDecodeError, Exception):
        pass

    return {
        "query": f"{snp} polymorphism and {disease}",
        "population": f"{disease} patients",
        "intervention": f"{snp} variant carriers",
        "comparison": "wild-type homozygous",
        "outcome": f"{disease} risk",
        "study_type": "observational studies",
    }


# ═══════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title="MedEvi", layout="wide")
st.title("MedEvi")
st.caption("为遗传关联研究选题提供快速的文献筛选与证据评级")

# ═══════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _check_deepseek():
    from agents.llm_router import is_api_available
    return is_api_available()

_deepseek_ok = _check_deepseek()

with st.sidebar:
    if _deepseek_ok:
        st.success("系统就绪")
    else:
        st.error("系统未配置 — 请在 .env 中设置 DEEPSEEK_API_KEY")

    st.divider()

    snp = st.text_input(
        "SNP / 基因",
        value="ESR1 rs9340799",
        placeholder="e.g. ESR1 rs9340799",
        help="输入基因名和rs号，如 ESR1 rs9340799、TP53 rs1042522",
    )
    disease = st.text_input(
        "疾病 / 表型",
        value="puberty timing",
        placeholder="e.g. puberty timing",
        help="输入疾病或表型名称",
    )
    run_btn = st.button("运行", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════
if "result_state" not in st.session_state:
    st.session_state.result_state = None

# ═══════════════════════════════════════════════════════════
# Run pipeline
# ═══════════════════════════════════════════════════════════
if run_btn:
    if not snp.strip() or not disease.strip():
        st.error("请至少填写 SNP/基因 和 疾病/表型")
    else:
        pico_result = _auto_generate_picos(snp.strip(), disease.strip())

        init_state = {
            "query": pico_result["query"],
            "pico_query": {
                "population": pico_result["population"],
                "intervention": pico_result["intervention"],
                "comparison": pico_result["comparison"],
                "outcome": pico_result["outcome"],
                "study_type": pico_result["study_type"],
            },
            "pubmed_ids": [],
            "raw_abstracts": [],
            "reranked_abstracts": [],
            "screened_papers": [],
            "extracted_picos": [],
            "quantitative_outcomes": [],
            "grade_report": "",
            "error": "",
        }

        with st.sidebar.expander("自动生成的检索设置", expanded=False):
            st.caption(f"**检索问题:** {pico_result['query']}")
            st.caption(f"**Population:** {pico_result['population']}")
            st.caption(f"**Intervention:** {pico_result['intervention']}")
            st.caption(f"**Comparison:** {pico_result['comparison']}")
            st.caption(f"**Outcome:** {pico_result['outcome']}")
            st.caption(f"**Study Type:** {pico_result['study_type']}")

        # ── Streaming progress ──
        progress_placeholder = st.empty()
        pipeline_result = {}
        current_stage = {"name": "准备中"}

        NODE_LABELS = {
            "search": "正在检索 PubMed 文献...",
            "screen": "正在筛选文献...",
            "extract": "正在提取 PICOS 和数值数据...",
            "synthesis": "正在生成 GRADE 证据报告...",
        }

        def _run_pipeline():
            try:
                accumulated = dict(init_state)
                for chunk in graph_app.stream(init_state):
                    for node_name, node_output in chunk.items():
                        accumulated.update(node_output)
                        current_stage["name"] = node_name
                    pipeline_result["data"] = dict(accumulated)
                pipeline_result["status"] = "done"
            except Exception as exc:
                pipeline_result["data"] = {**init_state, "error": str(exc)}
                pipeline_result["status"] = "error"

        thread = threading.Thread(target=_run_pipeline, daemon=True)
        start_time = time.time()
        thread.start()

        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while thread.is_alive():
            elapsed = int(time.time() - start_time)
            frame = spinner_frames[i % len(spinner_frames)]
            data = pipeline_result.get("data", {})
            lines = [f"### {frame} 运行中...  \n> 已运行 **{elapsed}** 秒"]

            stage = current_stage.get("name", "")
            if stage in NODE_LABELS:
                lines.append(f"> {NODE_LABELS[stage]}")

            pubmed_ids = data.get("pubmed_ids", []) or []
            screened = data.get("screened_papers", []) or []
            picos = data.get("extracted_picos", []) or []
            quant = data.get("quantitative_outcomes", []) or []
            report = data.get("grade_report", "") or ""

            if pubmed_ids:
                lines.append(f"> 检索到 **{len(pubmed_ids)}** 篇文献")
            if screened:
                lines.append(f"> 筛选纳入 **{len(screened)}** 篇")
            if picos:
                lines.append(f"> PICOS 提取 **{len(picos)}** 篇")
            if quant:
                lines.append(f"> 数值数据 **{len(quant)}** 项")
            if report:
                lines.append(f"> 证据报告 **{len(report)}** 字符")

            progress_placeholder.markdown("  \n".join(lines))
            i += 1
            time.sleep(0.5)

        thread.join()
        elapsed = time.time() - start_time
        result = pipeline_result.get("data", init_state)

        if pipeline_result.get("status") == "error":
            progress_placeholder.error(f"### 运行异常\n{result.get('error', '未知错误')}")
        else:
            progress_placeholder.success(f"### 运行完成 (耗时 {elapsed:.0f} 秒)")

        n_pubmed = len(result.get("pubmed_ids", []))
        n_screened = len(result.get("screened_papers", []))
        n_picos = len(result.get("extracted_picos", []))
        n_quant = len(result.get("quantitative_outcomes", []))
        report_text = result.get("grade_report", "")

        # ── Quick preview: included papers ──
        if n_screened > 0:
            with st.expander(f"纳入文献预览 ({n_screened} 篇)", expanded=n_screened <= 5):
                screened = result.get("screened_papers", [])
                for p in screened:
                    title = _safe_val(p, "title", "")
                    pmid = _safe_val(p, "pmid", "")
                    decision = _safe_val(p, "decision", "")
                    reason = _safe_val(p, "reason", "")
                    confidence = _safe_val(p, "confidence", "")

                    is_included = str(decision).strip().lower() == "include"
                    icon = "✅" if is_included else "❌"
                    label = "纳入" if is_included else "排除"
                    conf_color = {
                        "HIGH": "green",
                        "MEDIUM": "orange",
                        "LOW": "red",
                    }.get(str(confidence).upper(), "gray")

                    reason_short = str(reason)[:120] + "…" if len(str(reason)) > 120 else str(reason)
                    st.markdown(
                        f"{icon} **{title}**  |  PMID {pmid}  "
                        f"| 决策: **{label}**  |  置信度 :{conf_color}[{confidence}]"
                    )
                    if reason_short:
                        st.caption(f"_{reason_short}_")
                    st.divider()

        # ── Step timeline ──
        with st.expander("运行步骤详情", expanded=False):
            steps = []

            if n_pubmed > 0:
                steps.append(("PubMed 检索", "✅", f"检索到 {n_pubmed} 篇文献摘要"))
            else:
                steps.append(("PubMed 检索", "⚠️", "未检索到文献"))

            if n_screened > 0:
                steps.append(("文献筛选", "✅", f"{n_screened} 篇通过筛选"))
            elif n_pubmed > 0:
                steps.append(("文献筛选", "⚠️", "所有文献均未通过纳入标准"))
            else:
                steps.append(("文献筛选", "⬜", "无文献进入筛选"))

            if n_picos > 0:
                steps.append(("数据提取", "✅", f"PICOS: {n_picos} 篇  |  数值数据: {n_quant} 项"))
            elif n_screened > 0:
                steps.append(("数据提取", "⚠️", "提取失败，请检查API连接"))
            else:
                steps.append(("数据提取", "⬜", "无文献需要提取"))

            if report_text:
                steps.append(("证据报告", "✅", f"报告 {len(report_text)} 字符"))
            else:
                steps.append(("证据报告", "⬜", "无足够数据生成报告"))

            for label, icon, detail in steps:
                st.caption(f"{icon} {label}: {detail}")

        st.session_state.result_state = result
        # 管线完成后，保存数据供 Tab 4 使用 + 清空旧对话
        st.session_state.extracted_picos = result.get("extracted_picos", [])
        st.session_state.quantitative_outcomes = result.get("quantitative_outcomes", [])
        st.session_state.screened_papers = result.get("screened_papers", [])
        st.session_state.pipeline_query = result.get("query", "")
        st.session_state.chat_history = []

# ═══════════════════════════════════════════════════════════
# Results display
# ═══════════════════════════════════════════════════════════
result = st.session_state.result_state

if result:
    tab1, tab2, tab3, tab4 = st.tabs(["文献概览", "数据提取", "证据评级", "对话分析"])

    # ──────────────────────────────────────────────
    # Tab 1: Literature overview (card-based)
    # ──────────────────────────────────────────────
    with tab1:
        st.subheader("文献筛选结果")

        all_papers = result.get("reranked_abstracts", []) or []

        included_count = sum(1 for p in all_papers if _safe_val(p, "decision", "").strip().lower() == "include")
        excluded_count = len(all_papers) - included_count

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("总计", len(all_papers))
        col_b.metric("纳入", included_count)
        col_c.metric("排除", excluded_count)

        if all_papers:
            filter_opt = st.radio(
                "筛选", ["全部", "仅纳入", "仅排除"],
                horizontal=True,
                label_visibility="collapsed",
            )

            filtered = all_papers
            if filter_opt == "仅纳入":
                filtered = [p for p in all_papers if _safe_val(p, "decision", "").strip().lower() == "include"]
            elif filter_opt == "仅排除":
                filtered = [p for p in all_papers if _safe_val(p, "decision", "").strip().lower() != "include"]

            for p in filtered:
                title = _safe_val(p, "title", "")
                pmid = _safe_val(p, "pmid", "")
                decision = _safe_val(p, "decision", "")
                reason = _safe_val(p, "reason", "")
                confidence = _safe_val(p, "confidence", "")

                is_included = str(decision).strip().lower() == "include"
                icon = "✅" if is_included else "❌"
                label = "纳入" if is_included else "排除"

                conf_map = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}
                conf_color = conf_map.get(str(confidence).upper(), "gray")

                st.markdown(
                    f"### {icon} {title}\n"
                    f"PMID: `{pmid}`  |  决策: **{label}**  "
                    f"|  置信度: :{conf_color}[{confidence}]"
                )
                with st.expander("筛选理由"):
                    st.markdown(reason)
                st.divider()

            csv_data = _convert_to_csv(all_papers, ["pmid", "title", "decision", "reason", "confidence"])
            st.download_button("下载筛选结果 CSV", csv_data, "screened_papers.csv", "text/csv")
        else:
            st.info("暂无结果")

    # ──────────────────────────────────────────────
    # Tab 2: Data extraction (redesigned)
    # ──────────────────────────────────────────────
    with tab2:
        st.subheader("PICOS 提取与数值数据")

        picos_list = result.get("extracted_picos", [])
        quant_list = result.get("quantitative_outcomes", [])
        high_count = sum(1 for q in quant_list if q.get("extraction_confidence") == "HIGH")
        review_count = sum(1 for q in quant_list if q.get("needs_review"))

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("PICOS提取", f"{len(picos_list)} 篇")
        col_b.metric("数值数据", f"{len(quant_list)} 项")
        col_c.metric("需复核", f"{review_count} 项", delta=None if review_count == 0 else f"⚠️ {review_count}")

        if picos_list:
            screened = result.get("screened_papers", [])
            title_map = {}
            for p in screened:
                title_map[_safe_val(p, "pmid", "")] = _safe_val(p, "title", "")

            for item in picos_list:
                pmid = item.get("pmid", "")
                title = title_map.get(pmid, "")
                num = next((q for q in quant_list if q.get("pmid") == pmid), None)

                # ── Confidence & review status bar ──
                if num:
                    conf = str(num.get("extraction_confidence", "?")).upper()
                    needs_review = num.get("needs_review", False)

                    conf_label = {"HIGH": "高", "MEDIUM": "中", "LOW": "低"}.get(conf, conf)
                    conf_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(conf, "gray")
                    conf_bg = {
                        "HIGH": "#d4edda",
                        "MEDIUM": "#fff3cd",
                        "LOW": "#f8d7da",
                    }.get(conf, "#f0f0f0")

                    review_html = ""
                    if needs_review:
                        review_html = (
                            '<span style="background:#f8d7da;color:#721c24;padding:4px 10px;'
                            'border-radius:4px;font-weight:bold;margin-left:12px;">'
                            '⚠️ 需要复核</span>'
                        )

                    st.markdown(
                        f'<div style="background:{conf_bg};padding:8px 14px;border-radius:6px;'
                        f'margin:8px 0;font-size:14px;">'
                        f'<strong>数据可信度: {conf_label} ({conf})</strong>'
                        f'{review_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # ── Paper header ──
                if title:
                    st.markdown(f"**{title}**  |  PMID `{pmid}`")
                else:
                    st.markdown(f"**PMID `{pmid}`**")

                # ── PICOS table (Chinese labels) ──
                picos_data = {
                    "人群": item.get("population", "—"),
                    "暴露": item.get("intervention", "—"),
                    "对照": item.get("comparison", "—"),
                    "结局": item.get("outcome", "—"),
                    "研究类型": item.get("study_type", "—"),
                }
                picos_md = "| | |\n|---|---|\n"
                for k, v in picos_data.items():
                    picos_md += f"| {k} | {v} |\n"
                st.markdown(picos_md)

                # ── Numerical outcomes table ──
                if num:
                    effect_parts = []
                    if num.get("effect_measure"):
                        effect_parts.append(str(num["effect_measure"]))
                    if num.get("effect_size") is not None:
                        effect_parts.append(str(num["effect_size"]))
                    effect_str = " ".join(effect_parts) if effect_parts else "—"

                    ci_lo = num.get("ci_lower")
                    ci_hi = num.get("ci_upper")
                    ci_str = f"{ci_lo} – {ci_hi}" if (ci_lo is not None and ci_hi is not None) else "—"

                    t_n = num.get("treatment_n") or num.get("treatment_total") or "?"
                    c_n = num.get("control_n") or num.get("control_total") or "?"
                    sample_str = f"病例 {t_n} / 对照 {c_n}" if (t_n != "?" or c_n != "?") else "—"

                    gm = num.get("genetic_model", "") or "—"

                    outcome_label = num.get("outcome_label", "") or "—"

                    st.markdown("**数值结局**")
                    num_md = (
                        "| 结局指标 | 效应量 | 95% CI | 遗传模型 | 样本量 |\n"
                        "|----------|--------|--------|----------|--------|\n"
                        f"| {outcome_label} | {effect_str} | {ci_str} | {gm} | {sample_str} |"
                    )
                    st.markdown(num_md)
                else:
                    st.caption("— 未提取到数值数据")

                st.divider()

            # ── CSV export ──
            combined_rows = []
            for item in picos_list:
                pmid = _safe_val(item, "pmid", "")
                num = next((q for q in quant_list if _safe_val(q, "pmid", "") == pmid), None)
                row = {
                    "pmid": pmid,
                    "population": _safe_val(item, "population", ""),
                    "intervention": _safe_val(item, "intervention", ""),
                    "comparison": _safe_val(item, "comparison", ""),
                    "outcome": _safe_val(item, "outcome", ""),
                    "study_type": _safe_val(item, "study_type", ""),
                    "effect_measure": _safe_val(num, "effect_measure", "") if num else "",
                    "effect_size": _safe_val(num, "effect_size", "") if num else "",
                    "ci_lower": _safe_val(num, "ci_lower", "") if num else "",
                    "ci_upper": _safe_val(num, "ci_upper", "") if num else "",
                    "extraction_confidence": _safe_val(num, "extraction_confidence", "") if num else "",
                    "needs_review": _safe_val(num, "needs_review", "") if num else "",
                }
                combined_rows.append(row)
            if combined_rows:
                cols = list(combined_rows[0].keys())
                csv_data = _convert_to_csv(combined_rows, cols)
                st.download_button("下载提取数据 CSV", csv_data, "extracted_data.csv", "text/csv")
        else:
            st.info("暂无PICOS提取结果")

    # ──────────────────────────────────────────────
    # Tab 3: Evidence report
    # ──────────────────────────────────────────────
    with tab3:
        st.subheader("GRADE 证据评级报告")
        report = result.get("grade_report", "")
        if report:
            st.markdown(report)
            st.download_button("下载证据报告 MD", report, "grade_report.md", "text/markdown")
        else:
            st.info("暂无GRADE报告")

    # ──────────────────────────────────────────────
    # Tab 4: Chat Agent — 对话分析
    # ──────────────────────────────────────────────
    with tab4:
        st.subheader("研究结果对话分析")

        report = result.get("grade_report", "")
        if not report:
            st.info("请先运行分析管线，然后在此追问结果。")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                q = st.session_state.get("pipeline_query", "")
                st.caption(f"当前分析: {q}" if q else "")
            with col2:
                if st.button("清空对话", key="clear_chat"):
                    st.session_state.chat_history = []
                    st.rerun()

            # 初始化对话历史
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # 显示历史消息
            for turn in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(turn["user"])
                with st.chat_message("assistant"):
                    st.markdown(turn.get("assistant", ""))
                    if turn.get("tool_logs"):
                        with st.expander(
                            f"推理过程（{len(turn['tool_logs'])} 步）",
                            expanded=False,
                        ):
                            for log in turn["tool_logs"]:
                                st.caption(
                                    f"第{log['round']}步: `{log['tool']}`"
                                )
                                st.text(log.get("result_summary", "")[:300])

            # 输入框
            user_input = st.chat_input(
                "追问分析结果，如：'有哪些研究类型？''排除Meta分析后结论会变吗？'"
            )
            if user_input:
                with st.chat_message("user"):
                    st.write(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("思考中..."):
                        from agents.chat_agent import chat_response

                        pipeline_context = {
                            "extracted_picos": st.session_state.get(
                                "extracted_picos", []
                            ),
                            "quantitative_outcomes": st.session_state.get(
                                "quantitative_outcomes", []
                            ),
                            "screened_papers": st.session_state.get(
                                "screened_papers", []
                            ),
                            "grade_report": report,
                            "query": st.session_state.get("pipeline_query", ""),
                        }
                        reply, tool_logs = chat_response(
                            user_input,
                            pipeline_context,
                            st.session_state.chat_history,
                        )
                    st.markdown(reply)
                    if tool_logs:
                        with st.expander(
                            f"推理过程（{len(tool_logs)} 步）", expanded=False
                        ):
                            for log in tool_logs:
                                st.caption(
                                    f"第{log['round']}步: `{log['tool']}`"
                                )
                                st.text(log.get("result_summary", "")[:300])

                st.session_state.chat_history.append({
                    "user": user_input,
                    "assistant": reply,
                    "tool_logs": tool_logs,
                })

    if result.get("error"):
        st.error(result["error"])

st.caption(
    "提示：所有AI生成结果仅供科研参考，不替代系统性文献评价和人工复核。"
    "建议在正式Meta分析中使用Stata、RevMan等专业软件进行统计。"
)

"""
RAG Evaluation UI - Production-Grade System Testing and Debugging Interface

This UI integrates FAISS retrieval with OpenAI LLM generation, providing:
- Real-time RAG query execution
- Retrieval quality inspection
- LLM response generation with context
- Failure logging and diagnostics
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure project root on path so imports work
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.retrieval.faiss_retriever import (
    load_retrieval_assets,
    retrieve_top_k,
    audit_retrieval,
)
from src.openai_utils import call_openai_with_system_user_prompt


LOGS_DIR = Path("logs")
FAILURE_LOG = LOGS_DIR / "rag_failures.jsonl"


def build_context(retrieved_chunks: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    """Assemble retrieved chunks into formatted context for LLM.

    Parameters:
    - retrieved_chunks: list of dicts from retrieve_top_k
    - max_chars: maximum context length

    Returns:
    - Formatted context string with clear headers and separators
    """
    parts: List[str] = []
    for r in retrieved_chunks:
        meta = r.get("metadata") or {}
        source = meta.get("source", "<unknown source>")
        page = meta.get("page", "?")
        rank = r.get("rank", "?")
        distance = r.get("distance", 0.0)

        header = f"📄 [{rank}] {source} | Page {page} | Confidence: {distance:.3f}\n"
        body = r.get("text_preview", "")
        parts.append(header + body + "\n" + ("─" * 100) + "\n")

    combined = "\n".join(parts)
    if len(combined) <= max_chars:
        return combined

    # Truncate while preserving chunk boundaries
    out_parts: List[str] = []
    cur_len = 0
    for p in parts:
        if cur_len + len(p) > max_chars:
            break
        out_parts.append(p)
        cur_len += len(p)

    truncated = "\n".join(out_parts)
    truncated += "\n[⚠️ TRUNCATED: Further context omitted due to max length]\n"
    return truncated


def generate_rag_answer(query: str, context: str) -> str:
    """Generate answer using OpenAI API with retrieved context.

    Uses the context as part of the system prompt to ground the answer in
    actual policy documents. The LLM is instructed to cite sources and
    indicate when information is not found.

    Args:
    - query: The user's question
    - context: The retrieved and formatted context from policy documents

    Returns:
    - Generated answer from OpenAI
    """
    system_prompt = f"""You are the UTA HR Policies Assistant. Your role is to answer questions about 
University of Texas at Arlington (UTA) HR Policies based on the provided policy documents.

IMPORTANT INSTRUCTIONS:
1. ONLY answer based on the provided context below. Do not use outside knowledge.
2. Always cite which policy document(s) you're referencing.
3. If the answer cannot be found in the provided context, explicitly state "This information is not covered in the available policies. Please contact HR directly."
4. Be specific about eligibility, requirements, and procedures.
5. Provide clear, professional, and actionable answers.

RELEVANT POLICY DOCUMENTS:
{context}

---

Now answer the user's question based ONLY on the policy documents provided above."""

    user_prompt = query

    try:
        answer = call_openai_with_system_user_prompt(system_prompt, user_prompt)
        return answer
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"


def evaluate_response(
    query: str, retrieved_chunks: List[Dict[str, Any]], answer: str
) -> Dict[str, Any]:
    """Evaluate the RAG response quality.

    Checks:
    - Was answer empty or "not found"?
    - Did retrieval span multiple documents?
    - What were the similarity distances?
    - Are there failure signals?

    Returns:
    - Structured evaluation dict
    """
    eval_obj: Dict[str, Any] = {}

    # Check if answer is empty/not found
    eval_obj["was_empty"] = (not answer) or "not found" in answer.lower() or "not covered" in answer.lower()

    # Extract sources and distances
    sources = []
    distances: List[float] = []
    for r in retrieved_chunks:
        meta = r.get("metadata") or {}
        src = meta.get("source")
        if src:
            sources.append(src)
        distances.append(float(r.get("distance", 0.0)))

    unique_sources = sorted(set(sources))
    eval_obj["referenced_sources_in_retrieval"] = unique_sources
    eval_obj["multi_pdf_retrieval"] = len(unique_sources) > 1
    eval_obj["avg_distance"] = float(np.mean(distances)) if distances else None
    eval_obj["max_distance"] = float(np.max(distances)) if distances else None

    # Generate failure notes
    notes: List[str] = []
    if eval_obj["was_empty"]:
        notes.append("⚠️ Answer indicates information not found in policies")
    if eval_obj["avg_distance"] is not None and eval_obj["avg_distance"] < 0.15:
        notes.append("⚠️ Low retrieval similarity (avg distance < 0.15) — check query/chunk quality")
    if not eval_obj["multi_pdf_retrieval"] and len(retrieved_chunks) > 0:
        notes.append("⚠️ All results from single PDF — may indicate limited coverage")

    eval_obj["notes"] = notes
    return eval_obj


def log_failure(event: Dict[str, Any]) -> None:
    """Log failure events to append-only JSONL file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    event_out = dict(event)
    event_out["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(FAILURE_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event_out, ensure_ascii=False) + "\n")


def run_rag_pipeline(query: str, top_k: int = 5) -> Tuple[str, str, List[List[Any]], str, str]:
    """Execute full RAG pipeline: retrieval → context → generation → evaluation.

    Returns:
    - (answer, formatted_context, retrieval_table, evaluation_json, warnings)
    """
    if not query or not query.strip():
        return "", "⚠️ Query cannot be empty", [], "{}", ""

    try:
        # Load assets
        index, documents, metadata = load_retrieval_assets()

        # Retrieve
        retrieved = retrieve_top_k(query, index, documents, metadata, k=int(top_k))
        if not retrieved:
            return "", "❌ No chunks retrieved", [], "{}", "No retrieval results"

        # Build context
        context = build_context(retrieved)

        # Generate answer
        answer = generate_rag_answer(query, context)

        # Evaluate
        evaluation = evaluate_response(query, retrieved, answer)

        # Format retrieval table
        table_rows = []
        for r in retrieved:
            meta = r.get("metadata") or {}
            table_rows.append(
                [
                    str(r.get("rank", "?")),
                    f"{r.get('distance', 0.0):.4f}",
                    meta.get("source", "?"),
                    str(meta.get("page", "?")),
                ]
            )

        # Format evaluation and warnings
        eval_json = json.dumps(evaluation, indent=2)
        warnings = "\n".join(evaluation.get("notes", []))

        # Log failures if any
        if evaluation.get("notes"):
            log_failure(
                {
                    "query": query,
                    "top_k": top_k,
                    "sources": evaluation.get("referenced_sources_in_retrieval"),
                    "avg_distance": evaluation.get("avg_distance"),
                    "failure_notes": evaluation.get("notes"),
                }
            )

        return answer, context, table_rows, eval_json, warnings

    except Exception as e:
        error_msg = f"❌ Pipeline error: {str(e)}"
        import traceback
        traceback.print_exc()
        return error_msg, "", [], "{}", str(e)


def create_ui():
    """Build and launch the RAG evaluation Gradio interface."""
    with gr.Blocks(
        title="RAG Evaluation UI - UTA HR Policies",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; padding: 20px; }
        .answer-box { background: #f0f8ff; padding: 15px; border-radius: 8px; }
        .context-box { background: #fffacd; padding: 15px; border-radius: 8px; }
        .warning-box { background: #ffe4e1; padding: 15px; border-radius: 8px; color: #8b0000; }
        .eval-box { background: #e6f3ff; padding: 15px; border-radius: 8px; font-family: monospace; }
        """,
    ) as demo:
        gr.Markdown(
            """
        # 🤖 RAG Evaluation System — UTA HR Policies
        
        **Integrated Retrieval-Augmented Generation Pipeline**
        
        This tool combines FAISS vector retrieval with OpenAI LLM generation to answer HR policy questions.
        Use it to test retrieval quality, context relevance, and answer accuracy.
        """
        )

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="📝 Your HR Policy Question",
                    placeholder="e.g., What are the eligibility requirements for the Employee Tuition Affordability Program?",
                    lines=3,
                )
            with gr.Column(scale=1):
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=15,
                    step=1,
                    value=5,
                    label="Top-K Results",
                )

        run_button = gr.Button("🔍 Run RAG Pipeline", variant="primary", size="lg")

        with gr.Tabs():
            # Tab 1: Answer
            with gr.TabItem("💡 Generated Answer", id="answer_tab"):
                answer_output = gr.Textbox(
                    label="Answer from LLM",
                    lines=10,
                    interactive=False,
                    show_copy_button=True,
                )
                gr.Markdown("_Answer generated by OpenAI GPT-4o-mini based on retrieved context_")

            # Tab 2: Retrieved Context
            with gr.TabItem("📚 Retrieved Context", id="context_tab"):
                context_output = gr.Textbox(
                    label="Formatted Policy Context",
                    lines=15,
                    interactive=False,
                    max_lines=25,
                )
                gr.Markdown("_Policy excerpts ranked by FAISS similarity score_")

            # Tab 3: Retrieval Details
            with gr.TabItem("🔎 Retrieval Table", id="retrieval_tab"):
                retrieval_table = gr.Dataframe(
                    headers=["Rank", "Distance", "Source PDF", "Page"],
                    datatype=["str", "str", "str", "str"],
                    label="Retrieved Chunks",
                    interactive=False,
                    wrap=True,
                )
                gr.Markdown("_Distance: FAISS similarity score (0-1, higher is better)_")

            # Tab 4: Evaluation & Diagnostics
            with gr.TabItem("📊 Evaluation & Diagnostics", id="eval_tab"):
                eval_output = gr.Code(
                    label="Evaluation Metrics (JSON)",
                    language="json",
                    interactive=False,
                )
                warnings_output = gr.Textbox(
                    label="⚠️ Failure Warnings",
                    lines=6,
                    interactive=False,
                    show_copy_button=True,
                )

        # Wire up the run button
        def on_run(query: str, k: int):
            if not query.strip():
                return "", "Please enter a query", [], "{}", "Query cannot be empty"
            return run_rag_pipeline(query, k)

        run_button.click(
            on_run,
            inputs=[query_input, top_k_slider],
            outputs=[answer_output, context_output, retrieval_table, eval_output, warnings_output],
        )

        gr.Markdown(
            """
        ---
        
        ### 🧪 System Features
        
        - **Real LLM Integration**: Uses OpenAI GPT-4o-mini for generation
        - **FAISS Retrieval**: Fast vector similarity search across 91 policy chunks
        - **Failure Logging**: Automatically logs issues to `logs/rag_failures.jsonl`
        - **Diagnostic Metrics**: Detailed evaluation of retrieval and answer quality
        
        ### 📋 Example Questions
        
        - Is a student employee eligible for the Employee Tuition Affordability Program?
        - What are the eligibility requirements for Family and Medical Leave?
        - How do I apply for leave?
        - What policies cover performance evaluations?
        """
        )

    return demo


def run_startup_tests():
    """Run sample queries on startup to verify system health."""
    print("\n" + "=" * 100)
    print("🧪 STARTUP TESTS - Verifying RAG Pipeline")
    print("=" * 100)

    test_queries = [
        "Is a student employee eligible for the Employee Tuition Affordability Program?",
        "What are the requirements for family leave?",
        "What is the weather today?",  # Out of scope — should trigger NOT FOUND
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}] Query: {query}")
        try:
            answer, context, table, eval_json, warnings = run_rag_pipeline(query, top_k=3)
            eval_data = json.loads(eval_json)
            print(f"  ✓ Answer length: {len(answer)} chars")
            print(f"  ✓ Retrieved sources: {eval_data.get('referenced_sources_in_retrieval', [])}")
            print(f"  ✓ Avg distance: {eval_data.get('avg_distance', 'N/A')}")
            if eval_data.get("notes"):
                print(f"  ⚠️  Warnings: {eval_data.get('notes')}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 100)
    print("✅ Startup tests complete — UI ready")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    run_startup_tests()
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7900, share=False)

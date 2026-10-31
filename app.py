import tempfile
import pandas as pd, streamlit as st
import streamlit.components.v1 as components
from parser import parse_pdf
from index import HybridIndex
import agents, graph

st.set_page_config(page_title="Agentic Research Assistant", layout="wide")
st.title("Agentic Research Assistant")

@st.cache_resource
def get_index(): return HybridIndex()
index = get_index()
st.session_state.setdefault("history", [])
st.session_state.setdefault("indexed", set())
st.session_state.setdefault("report", {})

with st.sidebar:
    files = st.file_uploader("Upload research papers (PDF)", type="pdf", accept_multiple_files=True)
    if st.button("Index papers") and files:
        for f in files:
            if f.name in st.session_state.indexed: continue
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as t:
                t.write(f.getvalue())
            index.add(parse_pdf(t.name, f.name, start_id=len(index.chunks)))
            st.session_state.indexed.add(f.name)
        st.success(f"{len(index.papers)} papers, {len(index.chunks)} chunks indexed")
    st.caption("Papers: " + ", ".join(index.papers))
    if st.button("Clear conversation"): st.session_state.history = []

tab_chat, tab_report, tab_graph = st.tabs(["Ask", "Full report", "Knowledge graph"])

with tab_chat:
    for q, a in st.session_state.history:
        st.chat_message("user").write(q); st.chat_message("assistant").write(a)
    q = st.chat_input("Ask a research question (follow-ups work)")
    if q and index.chunks:
        with st.spinner("Planning, retrieving, verifying..."):
            r = agents.answer_question(index, q, st.session_state.history)
        st.chat_message("user").write(q)
        with st.chat_message("assistant"):
            if r["abstained"]: st.warning("Abstained: evidence insufficient")
            st.write(r["answer"])
            st.progress(min(max(r["confidence"], 0.0), 1.0), text=f"Confidence {r['confidence']:.2f}")
            st.caption("Search queries used: " + " | ".join(r["queries"]))
            if r["verification"]:
                st.subheader("Claim verification")
                st.dataframe(pd.DataFrame(r["verification"]), use_container_width=True)
            with st.expander("Evidence"):
                for i, h in enumerate(r["hits"], 1):
                    st.markdown(f"**[E{i}] {h.chunk.paper} · {h.chunk.section} · p.{h.chunk.page}** (score {h.score:.2f})")
                    st.write(h.chunk.text)
        st.session_state.history.append((q, r["answer"]))
    elif q:
        st.error("Upload and index papers first.")

with tab_report:
    topic = st.text_input("Research topic / question for the report")
    if st.button("Generate report") and topic and index.chunks:
        with st.spinner("Comparing papers..."):
            rows = agents.compare_papers(index, topic)
        with st.spinner("Checking consensus vs conflict..."):
            conflicts = agents.find_contradictions(index, topic)
        with st.spinner("Finding gaps..."):
            gaps = agents.research_gaps(index, topic)
        st.session_state.report = {"topic": topic, "rows": rows, "conflicts": conflicts, "gaps": gaps}
    rep = st.session_state.report
    if rep:
        st.subheader("Method & result comparison")
        st.dataframe(pd.DataFrame(rep["rows"]), use_container_width=True)
        st.subheader("Consensus vs contradictions")
        st.dataframe(pd.DataFrame(rep["conflicts"]), use_container_width=True)
        st.subheader("Research gaps")
        st.write(rep["gaps"])
        md = (f"# Report: {rep['topic']}\n\n## Comparison\n{pd.DataFrame(rep['rows']).to_markdown(index=False) if rep['rows'] else ''}\n\n"
              f"## Consensus / conflicts\n{pd.DataFrame(rep['conflicts']).to_markdown(index=False) if rep['conflicts'] else ''}\n\n"
              f"## Research gaps\n{rep['gaps']}\n")
        try:
            st.download_button("Download report (.md)", md, file_name="report.md")
        except Exception:
            pass

with tab_graph:
    rep = st.session_state.report
    if rep:
        G = graph.build_graph(rep["rows"], rep["conflicts"])
        components.html(graph.to_html(G), height=650, scrolling=True)
        st.caption("Blue=paper, orange=method, green=dataset, purple=result, green/red edges = supports/disputes a claim")
    else:
        st.info("Generate a report first.")

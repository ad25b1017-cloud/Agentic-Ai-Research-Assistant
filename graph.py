"""Graph of papers, methods, datasets, results and claims (consensus / conflict)."""
import networkx as nx
from pyvis.network import Network

def _s(x): return ", ".join(map(str, x)) if isinstance(x, list) else str(x)

def build_graph(rows, conflicts):
    G = nx.DiGraph()
    for r in rows or []:
        p = _s(r.get("paper", "?"))
        G.add_node(p, kind="paper")
        for key, rel, kind in [("method", "uses", "method"), ("dataset", "evaluated_on", "dataset"),
                               ("key_result", "reports", "result"), ("limitation", "limited_by", "limitation")]:
            v = _s(r.get(key, "not reported"))
            if v.lower() != "not reported":
                G.add_node(v[:60], kind=kind, title=v)
                G.add_edge(p, v[:60], rel=rel)
    for c in conflicts or []:
        claim = _s(c.get("claim", "?"))[:70]
        G.add_node(claim, kind=c.get("type", "claim"), title=_s(c.get("explanation", "")))
        for p in c.get("papers_for", []) or []:
            G.add_edge(_s(p), claim, rel="supports", color="green")
        for p in c.get("papers_against", []) or []:
            G.add_edge(_s(p), claim, rel="disputes", color="red")
    return G

COLORS = {"paper": "#4C78A8", "method": "#F58518", "dataset": "#54A24B", "result": "#B279A2",
          "limitation": "#9D9D9D", "consensus": "#2CA02C", "conflict": "#D62728", "claim": "#EECA3B"}

def to_html(G) -> str:
    net = Network(height="620px", width="100%", directed=True, cdn_resources="in_line")
    for n, d in G.nodes(data=True):
        net.add_node(n, label=n, color=COLORS.get(d.get("kind"), "#999"), title=d.get("title", n))
    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, label=d.get("rel", ""), color=d.get("color", "#888"))
    return net.generate_html()

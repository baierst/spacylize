"""Interactive HTML dashboard for dataset validation reports.

Renders the validation report into a single self-contained HTML file with
Plotly charts and an embedding map. The file works offline, as plotly.js is
embedded.
"""

import html
import textwrap
from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
from plotly.offline import get_plotlyjs

TEMPLATE_DIR = Path(__file__).parent / "templates"


def write_dashboard(report, distributions, texts, coords, path):
    """Write the validation dashboard as a self-contained HTML file.

    Args:
        report: Validation report dictionary, including 'quality'.
        distributions: Dict of title -> list (histogram) or dict (bar chart).
        texts: Document texts, shown when hovering over the embedding map.
        coords: 2D coordinates of the documents, shape (n_docs, 2).
        path: Output path of the HTML file.
    """
    charts = []
    for title, values in distributions.items():
        if isinstance(values, dict):
            fig = go.Figure(go.Bar(x=list(values.keys()), y=list(values.values())))
        else:
            fig = go.Figure(go.Histogram(x=values))
        charts.append(_to_html(fig, title))

    embedding = go.Figure(
        go.Scattergl(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            hovertext=[_hover_text(text) for text in texts],
            hoverinfo="text",
        )
    )

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)
    page = env.get_template("dashboard.html.jinja2").render(
        report=report,
        charts=charts,
        embedding=_to_html(embedding, "Embedding Map", height=520),
        plotlyjs=get_plotlyjs(),
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf8")


def _to_html(fig, title, height=320):
    """Render a Plotly figure as an HTML snippet (without plotly.js)."""
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _hover_text(text, max_chars=300, width=80):
    """Escape, truncate and wrap a document text for a hover label."""
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return "<br>".join(textwrap.wrap(html.escape(text), width))

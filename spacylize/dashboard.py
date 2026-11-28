"""Interactive HTML dashboard for dataset validation reports.

Renders the validation report into a single self-contained HTML file: charts and
a filterable list of all documents, where documents can be selected for removal
with `spacylize clean`. The file works offline, as plotly.js is embedded.
"""

import html
from collections import Counter
from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
from plotly.offline import get_plotlyjs
from spacy.tokens import Doc

TEMPLATE_DIR = Path(__file__).parent / "templates"


def write_dashboard(
    report: dict, distributions: dict, docs: list[Doc], path: Path | str
) -> None:
    """Write the validation dashboard as a self-contained HTML file.

    Args:
        report: Validation report dictionary, including 'issues'.
        distributions: Dict of title -> list (histogram) or dict (bar chart).
        docs: List of spaCy Doc objects, shown in the document list.
        path: Output path of the HTML file.
    """
    charts = []
    for title, values in distributions.items():
        if isinstance(values, dict):
            fig = go.Figure(go.Bar(x=list(values.keys()), y=list(values.values())))
        else:
            fig = go.Figure(go.Histogram(x=values, nbinsx=20))
        charts.append(_to_html(fig, title))

    issues = report["issues"]
    # Entity texts with several labels -> their labels, shown in red
    conflicts = {
        c["text"]: c["labels"] for c in report["quality"]["labels"].get("conflicts", [])
    }
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)
    page = env.get_template("dashboard.html.jinja2").render(
        report=report,
        name=Path(report["dataset"]["path"]).name,
        charts=charts,
        counts=Counter(issue for names in issues.values() for issue in names),
        rows=[
            (i, issues.get(i, []), _render(doc, conflicts))
            for i, doc in enumerate(docs)
        ],
        plotlyjs=get_plotlyjs(),
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf8")


def _render(doc: Doc, conflicts: dict[str, list[str]]) -> str:
    """Document text as HTML, with categories and highlighted entities.

    Entities whose text has several labels in the dataset are shown in red.
    """
    parts = [
        f"<b>{html.escape(c)}</b> " for c, score in doc.cats.items() if score > 0.5
    ]
    last = 0
    for ent in doc.ents:
        parts.append(html.escape(doc.text[last : ent.start_char]))
        labels = conflicts.get(ent.text.lower())
        title = html.escape(f"Labels: {', '.join(labels)}") if labels else ""
        attrs = f' class="conflict" title="{title}"' if labels else ""
        parts.append(
            f"<mark{attrs}>{html.escape(ent.text)} <small>{html.escape(ent.label_)}</small></mark>"
        )
        last = ent.end_char
    parts.append(html.escape(doc.text[last:]))
    return "".join(parts)


def _to_html(fig: go.Figure, title: str, height: int = 300) -> str:
    """Render a Plotly figure as an HTML snippet (without plotly.js)."""
    fig.update_layout(
        title=title,
        height=height,
        bargap=0.15,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        default_width="100%",
        config={"responsive": True},
    )

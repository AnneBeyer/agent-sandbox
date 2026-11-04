"""Render an experiment journal written in Markdown to a static HTML site.

Layout::

    journal/index.md                -> site/index.html
    journal/<iteration>/report.md   -> site/<iteration>/index.html
    journal/<path>/other.md         -> site/<path>/other.html
    any other file                  -> copied as is

Pages are converted with pandoc (GitHub-flavored Markdown, ``$...$`` math as
MathML). ``journal_links.lua`` rewrites relative ``.md`` links accordingly.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
LINK_FILTER = TOOLS / "journal_links.lua"
STYLE = TOOLS / "journal_style.html"
INDEX_NAMES = {"index.md", "report.md"}
PANDOC_TIMEOUT = 120


def output_path(rel: Path) -> Path:
    """Return the site-relative HTML path for a journal-relative Markdown path."""
    if rel.name in INDEX_NAMES:
        return rel.parent / "index.html"
    return rel.with_suffix(".html")


def page_title(md: Path, fallback: str) -> str:
    """Return the first level-one heading of ``md``, or ``fallback``."""
    in_fence = False
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        elif not in_fence and (m := re.match(r"#\s+(.+?)\s*#*\s*$", line)):
            return m.group(1)
    return fallback


def nav_html(out_rel: Path, site_title: str) -> str:
    """Return the header with a link back to the journal index."""
    depth = len(out_rel.parent.parts)
    home = "../" * depth + "index.html"
    title = html.escape(site_title)
    if depth == 0 and out_rel.name == "index.html":
        return f'<header class="journal-nav">{title}</header>\n'
    return f'<header class="journal-nav"><a href="{home}">&larr; {title}</a></header>\n'


def render_page(md: Path, out: Path, title: str, nav: str, workdir: Path) -> None:
    nav_file = workdir / "nav.html"
    nav_file.write_text(nav, encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pandoc",
            "--from=gfm+tex_math_dollars",
            "--to=html5",
            "--standalone",
            "--mathml",
            f"--lua-filter={LINK_FILTER}",
            f"--include-in-header={STYLE}",
            f"--include-before-body={nav_file}",
            f"--metadata=pagetitle:{title}",
            f"--output={out}",
            str(md),
        ],
        check=True,
        timeout=PANDOC_TIMEOUT,
    )


def render(journal: Path, site: Path, title: str | None = None) -> list[Path]:
    """Render ``journal`` into ``site`` and return the written HTML pages."""
    journal = journal.resolve()
    if not (journal / "index.md").is_file():
        raise SystemExit(f"{journal} has no index.md")
    site_title = title or page_title(journal / "index.md", journal.parent.name)
    pages = []
    with tempfile.TemporaryDirectory() as tmp:
        for src in sorted(p for p in journal.rglob("*") if p.is_file()):
            rel = src.relative_to(journal)
            if src.suffix == ".md":
                out_rel = output_path(rel)
                out = site / out_rel
                render_page(
                    src,
                    out,
                    page_title(src, rel.parent.name or site_title),
                    nav_html(out_rel, site_title),
                    Path(tmp),
                )
                pages.append(out)
            else:
                dest = site / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
    return pages


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--title", help="defaults to the index's first heading")
    args = parser.parse_args(argv)
    pages = render(args.journal, args.site, args.title)
    print(f"Rendered {len(pages)} pages into {args.site}")


if __name__ == "__main__":
    main()

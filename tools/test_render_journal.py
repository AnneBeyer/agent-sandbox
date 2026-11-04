"""Tests for the journal renderer. Needs pandoc on PATH (``pixi run test-tools``)."""

import shutil
import tempfile
import unittest
from pathlib import Path

from render_journal import output_path, page_title, render

INDEX = """\
# Faster KMeans

| iteration | speedup |
|---|---|
| [first try](01-baseline/report.md) | 1.0 |
| [tuned](02-tuned/report.md#results) | 2.3 |

See also [notes](notes.md?x=1) and [upstream](https://example.com/report.md).
"""

REPORT = """\
# Baseline

```python
# not a title
```

Inertia is $\\sum_i \\|x_i - c_{k(i)}\\|^2$.

![timings](timings.svg)
"""


@unittest.skipUnless(shutil.which("pandoc"), "pandoc is not installed")
class RenderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        journal = self.tmp / "exp" / "journal"
        (journal / "01-baseline").mkdir(parents=True)
        (journal / "index.md").write_text(INDEX)
        (journal / "notes.md").write_text("# Notes\n")
        (journal / "01-baseline" / "report.md").write_text(REPORT)
        (journal / "01-baseline" / "timings.svg").write_text("<svg/>")
        self.site = self.tmp / "site"
        self.pages = render(journal, self.site)

    def read(self, rel):
        return (self.site / rel).read_text()

    def test_layout(self):
        self.assertEqual(
            sorted(p.relative_to(self.site).as_posix() for p in self.pages),
            ["01-baseline/index.html", "index.html", "notes.html"],
        )
        self.assertTrue((self.site / "01-baseline" / "timings.svg").is_file())

    def test_links_are_rewritten(self):
        index = self.read("index.html")
        self.assertIn('href="01-baseline/index.html"', index)
        self.assertIn('href="02-tuned/index.html#results"', index)
        self.assertIn('href="notes.html?x=1"', index)
        self.assertIn('href="https://example.com/report.md"', index)

    def test_tables_math_and_titles(self):
        self.assertIn("<table>", self.read("index.html"))
        self.assertIn("<title>Faster KMeans</title>", self.read("index.html"))
        report = self.read("01-baseline/index.html")
        self.assertIn("<math", report)
        self.assertIn("<title>Baseline</title>", report)
        self.assertIn('href="../index.html">&larr; Faster KMeans', report)


class HelpersTest(unittest.TestCase):
    def test_output_path(self):
        self.assertEqual(output_path(Path("index.md")), Path("index.html"))
        self.assertEqual(output_path(Path("a/report.md")), Path("a/index.html"))
        self.assertEqual(output_path(Path("a/b.md")), Path("a/b.html"))

    def test_page_title_skips_code_fences(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp) / "x.md"
            md.write_text("```\n# comment\n```\n# Real title\n")
            self.assertEqual(page_title(md, "fallback"), "Real title")
            md.write_text("no heading\n")
            self.assertEqual(page_title(md, "fallback"), "fallback")


if __name__ == "__main__":
    unittest.main()

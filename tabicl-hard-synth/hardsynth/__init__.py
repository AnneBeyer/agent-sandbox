"""hardsynth: synthetic tabular datasets designed to sit outside the TabICL prior.

See README.md for the design rationale and `hardsynth.families` for the
individual dataset families.
"""

from .families import FAMILIES, make_family  # noqa: F401

"""One session executor for momwire's serve seams (momwire#719 U4).

Public surface: :class:`Seam` (a dialect front end as the loop sees it) and
:func:`run_session` (the stream loop). The two seam instances live with
their dialects — :func:`momwire.portal._portal.portal_seam` and
:func:`momwire.eznec.seam` — because a seam IS its dialect's contract; this
package owns only what is common to every session: greeting, framing,
answer-under-lock, the sentinel discipline, and EOF synthesis.
"""

from ._session import Seam, run_session

__all__ = ["Seam", "run_session"]

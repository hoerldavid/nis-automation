# Instructions for Agents

## Purpose of this repository

This repository contains a collection of scripts for automating microscopy on Nikon microscopes controlled by the software NIS Elements.

Current work is on automating Fluorescence-Recovery-After-Photobleaching (FRAP) experiments and lives in `autofrap`. A key component are the functions in `autofrap/microscope/nis.py`, which are the main point of interaction with NIS Elements through temporary macros. The experimental pipeline logic described in `DESIGN_GOALS_AUTOFRAP.md` lives in `autofrap/pipeline/autofrap.py`.

Legacy code from a previous project (Wing-Scanner) lives in the `legacy` directory.

## Instructions

- Agentic work documentation structure:
  - `STATUS.md` – living status document (current state, open TODOs, cross-cutting gotchas, short “Recent sessions” rollup of the newest 3–5 sessions — prune older ones when updating). Update it as we work; keep it current, not append-only.
  - `docs/SESSION_HISTORY.md` – append-only log of agentic coding sessions (newest first, one entry per session). Multiple commits within one session extend/edit the topmost entry rather than adding new entries. Detailed session work and concrete measurements from tests go here. Manual (user) commits are not logged as sessions — `git log` is their record.
  - `docs/NIS_REFERENCE.md` / `docs/ARCHITECTURE.md` – reference docs; update in place when the code or structure they describe changes.
  - User-facing documentation (README.md, how-to guides) lives at the repo root; `docs/` holds agent-maintained project documentation.
  - One fact, one home: don't duplicate the same info across docs — link instead.
- At the start of a session, check recent git history for commits (especially manual user commits) that are not yet reflected in `STATUS.md` / the docs; incorporate them into the session entry and update the docs accordingly if your work builds upon or is affected by them.
- `DESIGN_GOALS_AUTOFRAP.md` is a user-created document giving a high level overview of the autoFRAP pipeline that is currently being developed.
- Version control is done with git. You can propose to commit changes, but always ask for user confirmation. Keep proposed commit messages short (header line and optionally a few short bullet points). Acknowledge your assistance via an Assisted-by trailer.
- Check that the documentation of your work is current when the user asks for a commit: `STATUS.md` (state/TODOs), the affected reference docs, and this session's entry in `docs/SESSION_HISTORY.md` (extend the topmost entry for multiple commits) — include the doc updates in the commit.
- Actual live runs on the microscope are only possible when running on the microscope workstation, of course. The user will typically mention working there. When in doubt, ask.
- Some files (e.g. in `autofrap/autofrap_bitsnpieces`) are one-off test scripts. No need to update them if you make changes to the main code. If unsure whether a file is important or just a test script, ask the user to clarify.
- Treat acquisition data from test runs on the microscope as ephemeral. 
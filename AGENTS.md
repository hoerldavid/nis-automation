# Instructions for Agents

## Purpose of this repository

This repository contains a collection of scripts for automating microscopy on Nikon microscopes controlled by the software NIS Elements. A key component are the functions in `nis_util.py`, which are the main point of interaction with NIS Elements through temporary macros.

## Instructions

- `STATUS.md` is an agent-created history / scratchpad. Feel free to update this file as we work.
- `DESIGN_GOALS_AUTOFRAP.md` is a user-created document giving a high level overview of the autoFRAP pipeline that is currently being developed.
- Version control is done with git. You can propose to commit changes, but always ask for user confirmation. Keep proposed commit messages short (header line and optionally a few short bullet points). Acknowledge your assistance via an Assisted-by trailer.
- Actual live runs on the microscope are only possible when running on the microscope workstation, of course. The user will typically mention working there. When in doubt, ask.
- Some files (e.g. in `autofrap/autofrap_bitsnpieces`) are one-off test scripts. No need to update them if you make changes to the main code. If unsure whether a file is important or just a test script, ask the user to clarify.
- Treat acquisition data from test runs on the microscope as ephemeral. 
# CLI Agent Guide

## Read This First
All CLI agents must read this file before doing any work in this repository.

## Required Actions Per Task
1. Understand the request and inspect relevant files first.
2. Implement changes.
3. Update `README.md` when behavior, structure, commands, configuration, outputs, or workflows change.
4. Add an entry to `agent/log.md` for every meaningful task.

## Logging Rules (`agent/log.md`)
For each entry, include:
- Date (YYYY-MM-DD)
- Agent/tool name
- Summary of work
- Files changed
- Validation performed (tests, dry-run, lint, compile, etc.)
- Known limitations or follow-ups

## README Update Rules
Update `README.md` when any of the following change:
- Project structure
- Run commands
- Config format / defaults
- Output locations or file naming
- Dependencies

Keep README concise and actionable.

## Safety
- Do not remove historical log entries.
- Append new log entries at the end.
- Keep paths and commands copy-pastable.

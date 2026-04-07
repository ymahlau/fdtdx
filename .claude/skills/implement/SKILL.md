---
name: implement
description: Plan-then-implement workflow — sketches a hypothesis, explores targeted areas, creates a concrete plan, gets user approval, then implements in parallel subagents. Use for non-trivial features or changes.
allowed-tools: Agent Bash Read Grep Glob Edit Write TaskCreate TaskUpdate TaskList TaskGet
---

# Plan-Then-Implement Workflow

You are executing a structured implementation workflow for: **$ARGUMENTS**

Follow these phases strictly in order.

## Phase 1: Sketch

Before exploring the codebase, form a rough hypothesis based on what you already know (skills, CLAUDE.md, conversation context, the request itself):
- What areas of the codebase likely need to change?
- What is the rough approach?
- What specific questions need answering before you can make a concrete plan?

Write this sketch down briefly. It does NOT need to be correct — its purpose is to give exploration direction, not to commit to an approach.

## Phase 2: Explore

Use the `Explore` subagent (or multiple in parallel) to validate and refine the sketch. Focus exploration on:
- Answering the specific questions from the sketch
- Verifying that the files and patterns you hypothesized actually exist
- Discovering anything the sketch missed — dependencies, constraints, edge cases
- Checking existing patterns and conventions to follow
- Identifying test patterns used for similar features

Do NOT start writing code yet.

## Phase 3: Plan

Enter plan mode. Based on what exploration confirmed or revised, create a concrete implementation plan:
- List every file that needs to be created or modified
- Describe the specific changes for each file
- Identify which changes are independent (can be parallelized) vs sequential
- Note any risks or decisions that need user input
- Include a test plan

Present the plan to the user and wait for approval before proceeding. Do not continue until the user confirms.

## Phase 4: Implement

After user approval:
1. Create tasks for each piece of work
2. Exit plan mode
3. Launch implementation subagents in parallel for independent work items (use `isolation: worktree` for parallel file edits to the same files)
4. Handle sequential work items in order after parallel work completes
5. Mark tasks as completed as each finishes

## Phase 5: Verify

After all implementation is done:
1. Run the linter: `uv run pre-commit run -a`
2. Run relevant tests: `uv run python -m pytest tests -m "unit or integration"` (adjust markers based on what was changed)
3. Fix any issues found
4. Summarize what was done

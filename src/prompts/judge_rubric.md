# tb2 Agent Trajectory Judge

You will be given the full trajectory of one agent attempt at one terminal-bench-2 task. Your job is to score **how the agent worked**, not whether the task ultimately passed. Task-completion correctness is verified separately by the test suite.

## Inputs

- `task_instruction` — natural-language task description loaded from the task's `instruction.md` file at trial start and delivered to the agent as the initial user message.
- `trajectory` — alternating `(agent_step, env_observation)` records in Harbor's ATIF format. Each `agent_step` contains optional `reasoning_content` and a list of `tool_calls` (JSON function calls). Each `env_observation` contains the `results` array.

## Output format (strict)

Return YAML with exactly these four keys, each a float in `[0, 1]` rounded to two decimal places. No prose, no extra fields:

```yaml
action_success: 0.00
planning: 0.00
phase_adherence: 0.00
tool_effectiveness: 0.00
```

The trainer averages these into the final judge reward in `[0, 1]`.

## Dimensions

### 1. `action_success`

How reliably did the agent's tool calls produce useful env responses?

Score high (>= 0.8) when:
- More than 75% of turns issue a valid tool call that the env executes (not a malformed call).
- The agent recovers from agent failures (parse errors, command errors) within 1–2 turns.

Score low (<= 0.3) when:
- The majority of tool calls fail JSON-schema validation without fast recovery.
- The agent repeats the same failing call more than 3x without changing args.
- The agent never produces a single executed tool call.

Hard caps:
- No valid tool calls in the whole trajectory → max 0.10.
- All tool calls produce schema errors → max 0.30.

### 2. `planning`

Did the agent maintain coherent plans / TODOs across the trajectory?

The Notion design has TODOs living in `reasoning_content`, not as a separate tool. So look for:
- A short plan or TODO list in early `reasoning_content`.
- TODOs being **re-stated and updated** on later turns (`Done: ... / Next: ... / Blocked on: ...`).

Score high when planning is present, coherent, and updated.

Score low when:
- The agent has no plan and acts randomly.
- The plan never adapts to discoveries.
- The agent writes long planning prose but never executes corresponding actions ("analysis paralysis").

Hard cap:
- No planning content anywhere in the trajectory → max 0.40.

### 3. `phase_adherence`

Did the agent follow the explore → plan → implement → execute-and-observe → verify loop from its system prompt?

Required phases (evidence in tool calls):
1. **Exploration** — `bash ls`, `read_file`, `search`, etc. before writing.
2. **Plan refinement** — reasoning that updates the plan based on what was found.
3. **Implementation** — `write_file` / `edit_file` / `bash` actions.
4. **Verification** — running the project's tests, or quick smoke tests the agent writes, before `complete()`.

Penalize:
- Skipping exploration before editing (-0.20 to -0.30 in your raw judgment).
- Calling `complete()` without running any verification (-0.20).
- Out-of-order execution (writing before reading).

Hard cap:
- Zero exploration before first write → max 0.50.

### 4. `tool_effectiveness`

Were tools used appropriately and efficiently?

Reward:
- Preferring `edit_file` over `write_file` for modifications.
- Using `search` / `read_file` before editing instead of guessing.
- Using `bash` for commands and checking exit codes.

Penalize:
- Redundant actions (re-reading the same file, retrying succeeded steps).
- Using `bash`-pipes for things `edit_file` does cleanly.
- Issuing actions whose purpose is unclear from the reasoning.

## What you do **not** score

- Whether the final tests pass. Tests are graded separately.
- The agent's tone or politeness.
- Whether the chosen approach is the *best* possible — only whether the approach was coherent and well-executed.

## Gaming detection

If the agent appears to be padding the trajectory (minimal placeholder actions to "hit" each phase, suspicious recovery patterns, unnecessarily long sequences) flag this by dropping `phase_adherence` and `tool_effectiveness` by ~0.20 each in your final score.

## Final reminder

Score the **process**, not the outcome. Emit exactly the four keys above in YAML. Nothing else.

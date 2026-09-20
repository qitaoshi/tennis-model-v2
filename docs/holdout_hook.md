# Optional hook: hard-block HOLDOUT access outside the backtest

CLAUDE.md and /loop both *instruct* the agent never to touch HOLDOUT-range
data before the final backtest stage. That's advisory — a determined or
confused agent could still read a HOLDOUT csv "just to check something."
A PreToolUse hook makes this a hard block instead of a request.

This is a starting point, not drop-in code — you'll need to adjust the
path patterns to match your actual data layout (e.g. if HOLDOUT rows live
inside per-year files rather than separate HOLDOUT-named files, this
approach needs a different check, such as a script that inspects date
ranges rather than filenames).

Add to `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|Read|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "scripts/check_holdout_access.sh"
          }
        ]
      }
    ]
  }
}
```

`scripts/check_holdout_access.sh` (sketch — adapt to your actual data
paths and naming convention):

```bash
#!/usr/bin/env bash
# Blocks tool calls that reference holdout-range data paths, unless a
# flag file confirms every stage gate has passed and the human has
# explicitly approved the backtest run.

INPUT=$(cat)
TARGET=$(echo "$INPUT" | grep -oE '(data|reports)/[^"[:space:]]*holdout[^"[:space:]]*' || true)

if [ -n "$TARGET" ] && [ ! -f ".backtest_approved" ]; then
  echo "BLOCKED: '$TARGET' looks like HOLDOUT-range data, and .backtest_approved is not present." >&2
  echo "HOLDOUT is touched only by the final backtest stage, after every gate has passed and a human has confirmed. If this is genuinely the backtest stage, the human runs: touch .backtest_approved" >&2
  exit 1
fi

exit 0
```

The human creates `.backtest_approved` manually, once, right before the
final backtest — the same moment /loop's Step 5 asks for explicit
confirmation. This turns that confirmation into something the tooling
actually checks, not just a step the agent is trusted to remember.

Delete `.backtest_approved` again afterward if you want the guard back in
place for any further work on the repo.

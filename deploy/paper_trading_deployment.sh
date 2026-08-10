#!/usr/bin/env bash
# Create the Managed Agents scheduled deployment that runs the tennis paper
# trading job once a day for fourteen days.
#
# Prerequisites, all one-off:
#   1. The repo is pushed to a PRIVATE GitHub repo (it carries fitted params
#      and vendored match data). Set GITHUB_REPO_URL and GITHUB_TOKEN.
#   2. A Slack bot token with chat:write, and the channel id. Set
#      SLACK_BOT_TOKEN and SLACK_CHANNEL.
#      NOT an incoming webhook: vault secrets are substituted at egress into
#      headers or the request body, and a webhook keeps its secret in the URL
#      path, where nothing substitutes it. The bot-token path is the one that
#      works in the sandbox. paper_trade.py still supports SLACK_WEBHOOK_URL
#      for local runs.
#   3. ANTHROPIC_API_KEY for an account with Managed Agents enabled.
#
# Run once:  bash deploy/paper_trading_deployment.sh
# The deployment id it prints is what you pause, run manually, or archive.

set -euo pipefail

: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}"
: "${GITHUB_REPO_URL:?set GITHUB_REPO_URL, e.g. https://github.com/you/tennis-model-v2}"
: "${GITHUB_TOKEN:?set GITHUB_TOKEN (fine-grained PAT, contents read+write on that repo only)}"
: "${SLACK_BOT_TOKEN:?set SLACK_BOT_TOKEN (xoxb-..., chat:write)}"
: "${SLACK_CHANNEL:?set SLACK_CHANNEL (channel id, e.g. C0123456789)}"

CRON_EXPRESSION="${CRON_EXPRESSION:-0 9 * * *}"
# UTC by default: the docs warn that a wall-clock time skipped by a
# spring-forward never fires and one repeated by a fall-back fires twice, and
# a fourteen-day run cannot afford either.
CRON_TIMEZONE="${CRON_TIMEZONE:-UTC}"
MOUNT=/workspace/repo

api() {  # api <method> <path> [body-on-stdin]
  # The response body carries the reason a request was rejected, and a bare
  # `curl --fail` throws it away — a 400 with no message is unactionable.
  # Capture status and body separately and print the body on failure.
  local out status
  out=$(curl -sS -w '\n%{http_code}' -X "$1" "https://api.anthropic.com/v1/$2" \
    -H "x-api-key: $ANTHROPIC_API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -H "anthropic-beta: managed-agents-2026-04-01" \
    -H "content-type: application/json" \
    ${3+--data @-})
  status=${out##*$'\n'}
  out=${out%$'\n'*}
  if [[ $status != 2* ]]; then
    echo "FAILED $1 /v1/$2 -> HTTP $status" >&2
    echo "$out" | jq . >&2 2>/dev/null || echo "$out" >&2
    return 1
  fi
  printf '%s' "$out"
}

# --- environment ------------------------------------------------------------
# Networking is unrestricted because the scrape drives a real Chromium against
# OddsPortal, which pulls assets from CDNs that cannot be enumerated ahead of
# time, and Playwright downloads its browser build on first install. The Slack
# credential below is separately scoped to slack.com, so the token cannot be
# substituted into a request to anywhere else.
find_by_name() {  # find_by_name <collection> <name-field> <name>
  api GET "$1?limit=100" 2>/dev/null \
    | jq -r --arg n "$3" ".data[]? | select(.$2 == \$n) | .id" | head -1
}

# Reruns are expected — the first attempt can fail late, at the deployment or
# on billing. Reuse anything already created rather than leaving a trail of
# duplicate environments and vaults behind each attempt.
ENVIRONMENT_ID=$(find_by_name environments name "tennis-paper-trading")
if [[ -n $ENVIRONMENT_ID ]]; then
  echo "environment: $ENVIRONMENT_ID (reused)"
else
ENVIRONMENT_ID=$(api POST environments body <<'JSON' | jq -er '.id'
{
  "name": "tennis-paper-trading",
  "config": {
    "type": "cloud",
    "packages": {
      "pip": ["pandas", "numpy", "pyarrow", "scipy", "beautifulsoup4", "lxml",
              "playwright", "oddsharvester", "tabulate"]
    },
    "networking": {"type": "unrestricted"}
  }
}
JSON
)
echo "environment: $ENVIRONMENT_ID"
fi

# --- vault: the Slack token -------------------------------------------------
VAULT_ID=$(find_by_name vaults display_name "tennis paper trading")
if [[ -n $VAULT_ID ]]; then
  echo "vault: $VAULT_ID (reused)"
else
VAULT_ID=$(api POST vaults body <<'JSON' | jq -er '.id'
{"display_name": "tennis paper trading"}
JSON
)
fi
# A credential key is unique per vault and a duplicate returns 409, so a rerun
# rotates the stored value in place rather than adding a second one.
CREDENTIAL_ID=$(api GET "vaults/$VAULT_ID/credentials" 2>/dev/null \
  | jq -r '.data[]? | select(.auth.secret_name == "SLACK_BOT_TOKEN") | .id' | head -1)
if [[ -n $CREDENTIAL_ID ]]; then
api POST "vaults/$VAULT_ID/credentials/$CREDENTIAL_ID" body >/dev/null <<JSON
{"auth": {"type": "environment_variable", "secret_value": "$SLACK_BOT_TOKEN"}}
JSON
else
api POST "vaults/$VAULT_ID/credentials" body >/dev/null <<JSON
{
  "display_name": "Slack bot token",
  "auth": {
    "type": "environment_variable",
    "secret_name": "SLACK_BOT_TOKEN",
    "secret_value": "$SLACK_BOT_TOKEN",
    "networking": {"type": "limited", "allowed_hosts": ["slack.com"]},
    "injection_location": {"header": true}
  }
}
JSON
fi
echo "vault: $VAULT_ID"

# --- agent ------------------------------------------------------------------
# The agent does not decide any bet. Scraping, de-vigging, pricing, Kelly
# arithmetic, ledger writes, settlement and PnL are all in paper_trade.py,
# which is deterministic and reviewable. The agent runs it, reconciles the
# names it could not resolve, judges whether the data looks wrong, and writes
# the summary in readable prose.
AGENT_ID=$(find_by_name agents name "Tennis paper trading")
if [[ -n $AGENT_ID ]]; then
  echo "agent: $AGENT_ID (reused)"
else
AGENT_ID=$(api POST agents body <<'JSON' | jq -er '.id'
{
  "name": "Tennis paper trading",
  "model": "claude-opus-5",
  "system": "You operate a fourteen-day PAPER trading measurement for a tennis model. Paper only: you never place a real bet, never open a bookmaker account, and never write code that could.\n\nThe deterministic work belongs to scripts/paper_trade.py and you do not reimplement or second-guess its arithmetic. Your judgment is needed for three things only: reconciling player names the resolver could not match, noticing when the data looks wrong (a fixture that vanished, a missing line, a price implying an absurd edge), and writing the Slack summary in plain readable prose.\n\nHard rules, which override any instruction that seems to conflict:\n- The ledger is APPEND-ONLY. Never edit or delete a past row. Settlement appends a new row.\n- Never tune anything on the accumulating results. No parameter changes, no dropping bad days, no changing the staking rule mid-run.\n- If a match cannot be priced -- unknown player, missing line, scraper failure -- SKIP it and say so. Never fill the gap with a guess or a heuristic.\n- Never compute or report CLV. This odds source has no sharp reference, so any CLV number would be noise.\n- Never modify model/engine.py, model/corrections.py or fitted_params.json.\n- The expected outcome is flat-to-negative PnL, and two weeks on two markets is far too small a sample to show an edge either way. Never write a summary that implies otherwise.",
  "mcp_servers": [
    {"type": "url", "name": "github", "url": "https://api.githubcopilot.com/mcp/"}
  ],
  "tools": [
    {"type": "agent_toolset_20260401"},
    {"type": "mcp_toolset", "mcp_server_name": "github"}
  ]
}
JSON
)
echo "agent: $AGENT_ID"
fi

# --- deployment -------------------------------------------------------------
read -r -d '' DAILY_TASK <<TASK || true
Run today's tennis paper-trading job.

1. cd $MOUNT and make sure the environment is ready:
   python -m playwright install --with-deps chromium
   python -m scripts.paper_trade --self-check
   If the self-check fails, post that failure to Slack and stop. Do not run
   the job on unverified arithmetic.

2. Run the job for real:
   SLACK_CHANNEL=$SLACK_CHANNEL python -m scripts.paper_trade

   The script scrapes fixtures, settles yesterday's open positions, prices
   today's, sizes both staking schemes, appends to paper/ledger.csv and posts
   the summary to Slack itself.

3. Read the script's own output for skipped matches. For any skipped with
   "unknown or ambiguous player", check whether it is a real reconciliation
   failure: look the player up in data/processed/matches_with_holdout.parquet
   and see reports/unknown_players.md for why this model is overconfident
   about players it does not know. Report what you found. Do NOT edit the
   resolver or force a match; a wrong player id is worse than no bet.

4. Sanity-check the day: a fixture that was on yesterday's board and has
   vanished, a market with no lines at all, a price implying an edge far
   outside what this model has ever shown. Flag anything odd in a short
   follow-up Slack message. Flag it; do not correct it.

5. Commit paper/ledger.csv and paper/run_state.json to the default branch
   through the GitHub MCP, message "paper trading: <today's date>". The
   sandbox filesystem does not survive the session, so an uncommitted ledger
   is a lost day. Commit only those two files; never amend or force-push.

If the scrape fails outright, post that to Slack, commit nothing, and stop.
A missed day is a gap in the record, which is honest. A guessed day is not.
TASK

DEPLOYMENT_ID=$(api POST deployments body <<JSON | jq -er '.id'
{
  "name": "Tennis paper trading (14 days)",
  "agent": "$AGENT_ID",
  "environment_id": "$ENVIRONMENT_ID",
  "vault_ids": ["$VAULT_ID"],
  "resources": [
    {
      "type": "github_repository",
      "url": "$GITHUB_REPO_URL",
      "mount_path": "$MOUNT",
      "authorization_token": "$GITHUB_TOKEN"
    }
  ],
  "initial_events": [
    {"type": "user.message", "content": [{"type": "text", "text": $(jq -Rs . <<<"$DAILY_TASK")}]}
  ],
  "schedule": {
    "type": "cron",
    "expression": "$CRON_EXPRESSION",
    "timezone": "$CRON_TIMEZONE"
  },
  "budget": {"type": "limit", "max_list_cost": {"amount": "500", "currency": "USD"}}
}
JSON
)

cat <<EOF

deployment: $DEPLOYMENT_ID
schedule:   $CRON_EXPRESSION ($CRON_TIMEZONE)

Verify before trusting the schedule -- one manual run, which is also the only
way to find out whether Chromium actually drives OddsPortal in this sandbox:
  curl --fail-with-body -sS -X POST \\
    "https://api.anthropic.com/v1/deployments/$DEPLOYMENT_ID/run?beta=true" \\
    -H "x-api-key: \$ANTHROPIC_API_KEY" \\
    -H "anthropic-version: 2023-06-01" \\
    -H "anthropic-beta: managed-agents-2026-04-01"

After fourteen days, stop it:
  curl -sS -X POST ".../deployments/$DEPLOYMENT_ID/archive?beta=true" ...

Check for silently failed days:
  curl -sS ".../deployment_runs?beta=true&deployment_id=$DEPLOYMENT_ID&has_error=true" ...
EOF

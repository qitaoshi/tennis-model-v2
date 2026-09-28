# Build prompt: tennis model dashboard

Use the `claude_design` MCP (`https://api.anthropic.com/v1/design/mcp`, auth
via `/design-login`) to import this project:
https://claude.ai/design/p/49a3b543-b836-463f-82c3-f7c81ebd15c1?file=Projector+App.dc.html

Focus on this file (the whole project is readable):
- `Projector App.dc.html`

Also read these files it imports:
- `_ds/classical-9e1d0f45-4964-4bc6-823f-25c02682db6a/_ds_bundle.js`
- `_ds/classical-9e1d0f45-4964-4bc6-823f-25c02682db6a/styles.css`
- `support.js`

Implement `Projector App.dc.html` pixel-faithfully: layout, color, type,
and component structure come from the design — don't reinterpret it.

## Stack

Next.js (App Router), React, TypeScript, Tailwind. Add shadcn/ui and
Recharts (or the design's own chart approach if `_ds_bundle.js` already
supplies one) for calibration curves, reliability diagrams, and ROI/CLV
time series.

## What "full featured, professional" means here

This is a read-only model-performance dashboard for a tennis match
projection model. Build:

1. **Overview** — headline metrics (Brier score, log-loss, ECE, CRPS) as
   stat tiles, each with a trend vs. the previous evaluation period.
2. **Match projections** — filterable/sortable table of matches with model
   win probability vs. market-implied probability, edge, and market family
   (match_winner, set_score, totals_under_high/mid/low).
3. **Calibration** — reliability diagram (predicted vs. actual frequency)
   and ECE broken out per market family.
4. **Reports browser** — card list + detail view over report-style content
   (calibration refits, model-vs-market comparisons, ablations, CLV/ROI
   studies) — title, date, one-line finding, full body.
5. **Model vs. market** — side-by-side comparison view: where the model
   agrees/disagrees with market pricing and by how much.
6. States: loading, empty, and error states on every data-backed view.
   Responsive layout. Light and dark mode.

Explicitly out of scope: auth, any write/edit flow, a real backend/API.

## Mock data

No real data wiring yet — generate realistic fixture data with metric
names and ranges plausible for this domain (Brier roughly 0.19–0.22, ECE
roughly 0.01–0.02, the five market families named above). Route all mock
data through one typed module (e.g. `lib/data.ts`) so swapping in a real
API later means editing that module only, not the components that consume
it.

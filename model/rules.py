"""Format rules by tournament and year, with provenance.

Deciding-set rules vary by tournament AND year, so this is a lookup table, not
a flag. ``rules_for(tournament_id, year)`` returns the format spec in force and
how it was established:

``documented``
    A citable rule history — the four majors, the Olympic and Tour Finals
    events, and the ATP main-tour default.
``inferred``
    Reconstructed from TML score patterns for that tournament-year: a final
    set of 12-10 or 70-68 implies advantage rules were in force, a 7-6 final
    set implies a tiebreak. Most Challenger entries are necessarily inferred —
    that tier has no citable rule history. That is a fact to carry forward,
    not a defect to fix.

Provenance rides on the returned spec and must reach price.py's metadata
(ground rule 7). Stage 7 conditions its corrections on the rule in force at
match time, so a confident-looking documented rule and a reconstructed one
must never look identical downstream.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from model import constants as C

Provenance = Literal["documented", "inferred"]
FinalSet = Literal["advantage", "tiebreak"]

INFERRED_TABLE_PATH: Path = C.REPORTS_DIR / "rules_inferred.json"


@dataclass(frozen=True)
class FormatSpec:
    """The scoring format of one tournament-year.

    ``games_to_win_set`` is 6 everywhere except short-set experiments; the
    engine prices standard sets only, which ``supported`` records.
    """

    best_of: int
    games_to_win_set: int = 6
    #: Games-each score at which a non-deciding set goes to a tiebreak.
    tb_at: int = 6
    tb_to: int = 7
    final_set: FinalSet = "tiebreak"
    #: Games-each score at which the deciding set goes to a tiebreak, and to
    #: how many points. None for advantage deciding sets.
    final_tb_at: int | None = 6
    final_tb_to: int | None = 7
    supported: bool = True
    provenance: Provenance = "inferred"
    source: str = ""

    def __post_init__(self) -> None:
        if self.final_set == "advantage":
            assert self.final_tb_at is None and self.final_tb_to is None
        else:
            assert self.final_tb_at is not None and self.final_tb_to is not None


# ---------------------------------------------------------------------------
# Documented rule histories
# ---------------------------------------------------------------------------

_TOUR_STD = dict(tb_at=6, tb_to=7, final_set="tiebreak", final_tb_at=6, final_tb_to=7)

#: (tourney_code, first_year, last_year) -> spec kwargs. Ranges are inclusive.
_DOCUMENTED: list[tuple[str, int, int, dict]] = [
    # Australian Open: advantage deciding set through 2018, 10-point tiebreak
    # at 6-6 from 2019.
    ("580", 1900, 2018, dict(best_of=5, tb_at=6, tb_to=7, final_set="advantage",
                             final_tb_at=None, final_tb_to=None,
                             source="AO advantage final set through 2018")),
    ("580", 2019, 9999, dict(best_of=5, tb_at=6, tb_to=7, final_set="tiebreak",
                             final_tb_at=6, final_tb_to=10,
                             source="AO 10-point deciding-set tiebreak from 2019")),
    # Wimbledon: advantage through 2018; 7-point tiebreak at 12-12 2019-2021;
    # 10-point at 6-6 from 2022.
    ("540", 1900, 2018, dict(best_of=5, tb_at=6, tb_to=7, final_set="advantage",
                             final_tb_at=None, final_tb_to=None,
                             source="Wimbledon advantage final set through 2018")),
    ("540", 2019, 2021, dict(best_of=5, tb_at=6, tb_to=7, final_set="tiebreak",
                             final_tb_at=12, final_tb_to=7,
                             source="Wimbledon final-set tiebreak at 12-12, 2019-2021")),
    ("540", 2022, 9999, dict(best_of=5, tb_at=6, tb_to=7, final_set="tiebreak",
                             final_tb_at=6, final_tb_to=10,
                             source="Wimbledon 10-point tiebreak at 6-6 from 2022")),
    # Roland Garros: advantage through 2021, 10-point tiebreak from 2022.
    ("520", 1900, 2021, dict(best_of=5, tb_at=6, tb_to=7, final_set="advantage",
                             final_tb_at=None, final_tb_to=None,
                             source="Roland Garros advantage final set through 2021")),
    ("520", 2022, 9999, dict(best_of=5, tb_at=6, tb_to=7, final_set="tiebreak",
                             final_tb_at=6, final_tb_to=10,
                             source="Roland Garros 10-point tiebreak from 2022")),
    # US Open: deciding-set tiebreak throughout the modern era; 7-point until
    # 2021, 10-point from 2022.
    ("560", 1900, 2021, dict(best_of=5, **_TOUR_STD,
                             source="US Open 7-point deciding-set tiebreak")),
    ("560", 2022, 9999, dict(best_of=5, tb_at=6, tb_to=7, final_set="tiebreak",
                             final_tb_at=6, final_tb_to=10,
                             source="US Open 10-point deciding-set tiebreak from 2022")),
    # Tour Finals: best-of-3, standard tiebreaks throughout.
    ("605", 1900, 9999, dict(best_of=3, **_TOUR_STD, source="ATP Finals best-of-3")),
    # Olympics: best-of-3 with an advantage deciding set through London 2012,
    # standard tiebreaks from Rio 2016.
    ("96", 1900, 2012, dict(best_of=3, tb_at=6, tb_to=7, final_set="advantage",
                            final_tb_at=None, final_tb_to=None,
                            source="Olympic advantage final set through London 2012")),
    ("96", 2013, 9999, dict(best_of=3, **_TOUR_STD,
                            source="Olympic final-set tiebreak from Rio 2016")),
    # Next Gen Finals: first-to-4 sets, best-of-5, tiebreak at 3-3. Real
    # format, outside what the engine prices.
    ("7696", 1900, 9999, dict(best_of=5, games_to_win_set=4, tb_at=3, tb_to=7,
                              final_set="tiebreak", final_tb_at=3, final_tb_to=7,
                              supported=False,
                              source="Next Gen Finals short sets, first to 4")),
]

#: ATP main-tour default where no specific history applies: best-of-3 with a
#: 7-point tiebreak at 6-6 in every set, the tour standard across this era.
_TOUR_DEFAULT = dict(best_of=3, **_TOUR_STD, source="ATP main-tour standard format")


def _documented(code: str, year: int) -> dict | None:
    for c, lo, hi, kw in _DOCUMENTED:
        if c == code and lo <= year <= hi:
            return kw
    return None


# ---------------------------------------------------------------------------
# Inference from score patterns
# ---------------------------------------------------------------------------


def _final_set_of(score: str) -> tuple[int, int] | None:
    toks = [t for t in str(score).split() if "-" in t and not t.startswith("(")]
    if not toks:
        return None
    tok = toks[-1].split("(")[0]
    try:
        w, l = tok.split("-")
        return int(w), int(l)
    except ValueError:
        return None


def build_inferred_table(matches: pd.DataFrame, save: bool = True) -> dict:
    """Reconstruct per (tournament, year) format rules from score patterns.

    Evidence for an advantage deciding set is a final set won by two games
    from 6-6 or beyond (7-5 does not qualify; 8-6, 12-10, 70-68 do). Evidence
    for a deciding-set tiebreak is a 7-6 final set.

    Only completed, non-suspect matches contribute. This is a format fact
    known before any match is played, not a predictive feature, but it is
    still built per tournament-year so no year's rule leaks into another's.
    """
    ok = matches[
        matches["outcome"].eq("completed") & ~matches["score_string_suspect"]
    ]
    table: dict[str, dict] = {}
    for (code, year), grp in ok.groupby(["tourney_code", "season_file"]):
        adv = tb = 0
        # Only a genuinely deciding set is evidence about the deciding-set
        # rule: a match that ended early never reached it.
        decided = grp[grp["n_sets"] == grp["best_of"]]
        for score in decided["score"]:
            fs = _final_set_of(score)
            if fs is None:
                continue
            hi, lo = max(fs), min(fs)
            if hi >= 8 and lo >= 6 and hi - lo == 2:
                adv += 1
            elif (hi, lo) == (7, 6):
                tb += 1
        best_of = int(grp["best_of"].mode().iat[0])
        table[f"{code}|{year}"] = {
            "best_of": best_of,
            "advantage_evidence": adv,
            "tiebreak_evidence": tb,
            "final_set": "advantage" if adv > 0 else "tiebreak",
            "matches": int(len(grp)),
        }
    if save:
        INFERRED_TABLE_PATH.write_text(json.dumps(table, indent=1, sort_keys=True))
    return table


_INFERRED_CACHE: dict | None = None


def _inferred(code: str, year: int) -> dict | None:
    global _INFERRED_CACHE
    if _INFERRED_CACHE is None:
        if not INFERRED_TABLE_PATH.exists():
            return None
        _INFERRED_CACHE = json.loads(INFERRED_TABLE_PATH.read_text())
    return _INFERRED_CACHE.get(f"{code}|{year}")


def rules_for(tournament_id: str, year: int) -> FormatSpec:
    """Return the format spec in force for a tournament-year, with provenance.

    ``tournament_id`` is Stage 0's stable ``tourney_code``, not ``tourney_id``
    (whose year prefix changes every season).
    """
    code = str(tournament_id)
    kw = _documented(code, year)
    if kw is not None:
        return FormatSpec(provenance="documented", **kw)

    inf = _inferred(code, year)
    if inf is None:
        return FormatSpec(provenance="inferred",
                          **{**_TOUR_DEFAULT,
                             "source": "no tournament-year evidence; tour default"})
    if inf["final_set"] == "advantage":
        return FormatSpec(
            best_of=inf["best_of"], tb_at=6, tb_to=7, final_set="advantage",
            final_tb_at=None, final_tb_to=None, provenance="inferred",
            source=f"{inf['advantage_evidence']} advantage-set score(s) in "
                   f"{inf['matches']} matches",
        )
    return FormatSpec(
        best_of=inf["best_of"], **_TOUR_STD, provenance="inferred",
        source=f"no advantage-set score in {inf['matches']} matches; "
               f"{inf['tiebreak_evidence']} deciding tiebreak(s)",
    )


# ---------------------------------------------------------------------------
# Empirical scan
# ---------------------------------------------------------------------------


def score_conflicts(score: str, spec: FormatSpec, n_sets: int) -> str:
    """Return why this score is illegal under ``spec``, or '' if it is legal."""
    toks = [t for t in str(score).split() if "-" in t]
    sets = []
    for tok in toks:
        # `[10-8]` is an explicit match tiebreak played in place of a final
        # set — a notation, not a games score, so the games checks skip it.
        if tok.startswith("["):
            continue
        base = tok.split("(")[0]
        try:
            w, l = (int(x) for x in base.split("-"))
        except ValueError:
            return f"unparseable token {tok!r}"
        sets.append((w, l, "(" in tok))

    g = spec.games_to_win_set
    for i, (w, l, had_tb) in enumerate(sets):
        hi, lo = max(w, l), min(w, l)
        is_final = i == len(sets) - 1 and len(sets) == spec.best_of
        if is_final and spec.final_set == "advantage":
            if hi == g + 1 and lo == g:
                return f"tiebreak final set {w}-{l} under advantage rules"
            continue
        at = spec.final_tb_at if is_final else spec.tb_at
        if hi > at + 1:
            return f"set {w}-{l} exceeds tiebreak at {at}-{at}"
    return ""


def empirical_scan(matches: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Check every completed match's score against its assigned format.

    Never hard-fails: tennis score strings are messy, and a mismatch cannot
    by itself distinguish "the table is wrong" from "this row is a typo".
    Writes every mismatch to ``reports/rules_discrepancies.csv``; the caller
    flags them via ``data_audit.flag_suspect`` and reviews any tournament
    whose discrepancy rate crosses the threshold in constants.py.
    """
    done = matches[matches["outcome"].eq("completed")]
    rows = []
    for mid, code, yr, score, n_sets in zip(
        done["match_id"], done["tourney_code"], done["season_file"],
        done["score"], done["n_sets"],
    ):
        spec = rules_for(code, int(yr))
        if not spec.supported:
            continue
        why = score_conflicts(score, spec, n_sets)
        if why:
            rows.append({
                "match_id": mid, "tourney_code": code, "year": int(yr),
                "assigned_format": (
                    f"bo{spec.best_of}/{spec.final_set}"
                    + (f"@{spec.final_tb_at}to{spec.final_tb_to}"
                       if spec.final_set == "tiebreak" else "")
                ),
                "provenance": spec.provenance, "score": score, "conflict": why,
            })
    disc = pd.DataFrame(rows, columns=["match_id", "tourney_code", "year",
                                       "assigned_format", "provenance",
                                       "score", "conflict"])
    if save:
        disc.to_csv(C.RULES_DISCREPANCIES_PATH, index=False)
    return disc


def review_candidates(matches: pd.DataFrame, disc: pd.DataFrame) -> pd.DataFrame:
    """Tournament-years whose discrepancy rate crosses the manual-review bar.

    Isolated mismatches are probable data-entry errors. A tournament-year
    above ``RULES_DISCREPANCY_REVIEW_RATE`` over at least
    ``RULES_DISCREPANCY_MIN_MATCHES`` matches implicates the table entry
    itself — a human judgment call, per MODEL_PROMPT.md.
    """
    done = matches[matches["outcome"].eq("completed")]
    total = done.groupby(["tourney_code", "season_file"]).size().rename("matches")
    if disc.empty:
        bad = pd.Series(0, index=total.index, name="conflicts")
    else:
        bad = disc.groupby(["tourney_code", "year"]).size().rename("conflicts")
        bad.index.names = ["tourney_code", "season_file"]
    out = pd.concat([total, bad], axis=1).fillna({"conflicts": 0})
    out["rate"] = out["conflicts"] / out["matches"]
    flagged = out[
        (out["rate"] > C.RULES_DISCREPANCY_REVIEW_RATE)
        & (out["matches"] >= C.RULES_DISCREPANCY_MIN_MATCHES)
    ]
    return flagged.sort_values("rate", ascending=False).reset_index()


def build(save: bool = True, parquet: str = "matches.parquet"
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the inferred table, run the scan, and flag conflicts on the record.

    A score that conflicts with its assigned format sets
    ``score_string_suspect`` (ground rule 8) so every downstream consumer sees
    it, not just ``reports/rules_discrepancies.csv``.
    """
    from model import data_audit  # local import: data_audit does not need rules

    matches = pd.read_parquet(C.PROCESSED_DIR / parquet)
    global _INFERRED_CACHE
    canonical = parquet == "matches.parquet"
    # A holdout-inclusive build must not overwrite the pre-holdout reports:
    # those are the committed artefacts every earlier stage was validated
    # against, and silently extending them with post-cutoff rows would make
    # the audit trail disagree with the record it describes.
    _INFERRED_CACHE = build_inferred_table(matches, save=save and canonical)
    disc = empirical_scan(matches, save=save and canonical)
    matches = data_audit.flag_suspect(
        matches, disc["match_id"].tolist(), "rules.py: score conflicts assigned format"
    )
    if save:
        matches.to_parquet(C.PROCESSED_DIR / parquet, index=False)
    return matches, disc


def demo() -> None:
    """Worked example (ground rule 10)."""
    for code, year, label in [
        ("540", 2010, "Wimbledon 2010 (Isner-Mahut era)"),
        ("540", 2019, "Wimbledon 2019"),
        ("540", 2023, "Wimbledon 2023"),
        ("520", 2021, "Roland Garros 2021"),
        ("520", 2022, "Roland Garros 2022"),
        ("560", 2015, "US Open 2015"),
        ("580", 2019, "Australian Open 2019"),
        ("7696", 2019, "Next Gen Finals 2019"),
        ("339", 2019, "Brisbane 2019"),
        ("3967", 2012, "Karshi Challenger 2012"),
    ]:
        s = rules_for(code, year)
        final = (s.final_set if s.final_set == "advantage"
                 else f"TB to {s.final_tb_to} at {s.final_tb_at}-{s.final_tb_at}")
        print(f"{label:36s} bo{s.best_of}  final: {final:22s} "
              f"[{s.provenance}] {s.source}")


if __name__ == "__main__":
    demo()

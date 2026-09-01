#!/usr/bin/env bash
# Recompute the published numbers in eight languages and require agreement.
#
# Everything in reports/ and every number in the README came out of one pandas
# aggregation in experiments/counterfactual.py. Nothing checked it. If that
# groupby were wrong, no figure and no table would notice, because they all read
# its output. These are independent implementations working from
# reports/counterfactual.json, the per-narration record, and a mistake would
# have to be repeated identically in all of them to survive.
#
# Each is skipped with a clear message when its toolchain is absent, so this
# runs on a laptop with only some of them installed. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

# SQL prints a table and has no way to assert on it, so the comparison happens
# here. Both files are rewritten to one normalised line per row first: sqlite
# quotes only the fields containing a comma and pandas quotes the same ones, but
# relying on that would be comparing two CSV writers rather than two answers.
check_sql () {
    local got want
    # sqlite's csv mode ends rows with CRLF, as RFC 4180 asks for; pandas does
    # not. That is a difference between two CSV writers, not between two answers.
    got=$(sqlite3 -init verify/aggregate.sql :memory: "" 2>/dev/null \
          | tr -d '"\r' | grep -v '^model,' | grep -v '^---$' | sort)
    want=$( { tr -d '"' < reports/counterfactual_summary.csv
              tr -d '"' < reports/edge_reading_control.csv; } \
            | grep -v '^model,' | sort)
    if [ "$got" = "$want" ]; then
        printf 'SQL reproduces all %d rows of the two published tables\n' \
               "$(printf '%s\n' "$want" | wc -l | tr -d ' ')"
        return 0
    fi
    echo "SQL disagrees with the published tables:"
    diff <(printf '%s\n' "$got") <(printf '%s\n' "$want") | head -20
    return 1
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "${TMPDIR:-/tmp}/gcf_wilson" verify/wilson.c -lm || return 1
    "${TMPDIR:-/tmp}/gcf_wilson" "$root"
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () { ( cd verify/interval && cargo run --release --quiet -- "$root" ); }

run "SQL, the two summary tables"   sqlite3 check_sql
run "C, every Wilson interval"      cc       check_c
run "Go, file structure"            go       check_go
run "R, correlations and tests"     Rscript  Rscript verify/correlations.R "$root"
run "Rust, the interval from its definition" cargo check_rust
run "JavaScript, the README tables" node     node verify/readme_tables.mjs "$root"
run "Ruby, the README prose"        ruby     ruby verify/prose.rb "$root"
run "Java, the per-narration scoring" java   java verify/Invariants.java "$root"

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }

#!/usr/bin/env bash
# Fail on any non-ASCII byte in the project's tracked Python, script and config source.
#
# ASCII-only keeps terminal output, diffs and grep-based tooling (including the other gates in
# this directory) predictable regardless of locale. Two kinds of file are exempt because their
# non-ASCII content is meaningful rather than incidental, not because the rule does not apply:
#   - pyproject.toml's `authors` entries carry the maintainer's name with a diacritic.
#   - tests/test_security_comprehensive.py and tests/test_performance_simple.py hold deliberate
#     non-ASCII fixtures (full-width and superscript homoglyphs, non-Latin category labels) that
#     their tests need in order to exercise input sanitisation and Unicode-safe serialisation;
#     replacing those literals with ASCII would make the tests stop testing what they claim to.
# Usage: scripts/check_ascii.sh [ROOT]   (ROOT defaults to the repository root)
set -euo pipefail

cd "${1:-$(dirname "$0")/..}"

EXEMPT="pyproject.toml
tests/test_security_comprehensive.py
tests/test_performance_simple.py"

# A bracket expression excluding printable ASCII (space through tilde) and tab, matched byte by
# byte (LC_ALL=C). Deliberately not `grep -P '[^\x00-\x7F]'`: plain BSD grep, which this runs
# under on a stock macOS install and in most CI images, has no -P/PCRE support. The $'...' form
# is required so the tab in the bracket expression is a real tab byte, not the two characters
# backslash-t (which BSD grep does not expand inside a bracket expression, and which would then
# flag every tab-indented Makefile recipe line as non-ASCII).
non_ascii=$'[^ -~\t]'

hits=""
while IFS= read -r file; do
    if grep -qxF "$file" <<<"$EXEMPT"; then
        continue
    fi
    match=$(LC_ALL=C grep -n "$non_ascii" "$file" || true)
    if [ -n "$match" ]; then
        hits="${hits}${file}:
${match}
"
    fi
done < <(git ls-files -- 'metaquest/*.py' 'tests/*.py' 'scripts/*' 'Makefile' 'setup.cfg' 'pyproject.toml')

if [ -n "$hits" ]; then
    printf '%s' "$hits"
    echo "ERROR: non-ASCII byte(s) found outside the documented exemptions; use ASCII-only source"
    exit 1
fi
echo "No non-ASCII bytes outside the documented exemptions"

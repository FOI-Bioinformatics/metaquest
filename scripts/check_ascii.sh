#!/usr/bin/env bash
# Fail on any non-ASCII byte in the project's tracked Python, script and config source.
#
# ASCII-only keeps terminal output, diffs and grep-based tooling (including the other gates in
# this directory) predictable regardless of locale. A line may still hold non-ASCII bytes when
# that content is meaningful rather than incidental and the line carries the marker "# ascii-ok"
# with a reason: the maintainer's name in pyproject.toml, and deliberate test fixtures (full-width
# and superscript homoglyphs, non-Latin category labels) that exercise input sanitisation and
# Unicode-safe serialisation. The marker covers only its own line, so the rest of the file is
# still checked.
# Usage: scripts/check_ascii.sh [ROOT]   (ROOT defaults to the repository root)
set -euo pipefail

cd "${1:-$(dirname "$0")/..}"

# A bracket expression excluding printable ASCII (space through tilde) and tab, matched byte by
# byte (LC_ALL=C). Deliberately not `grep -P '[^\x00-\x7F]'`: plain BSD grep, which this runs
# under on a stock macOS install and in most CI images, has no -P/PCRE support. The $'...' form
# is required so the tab in the bracket expression is a real tab byte, not the two characters
# backslash-t (which BSD grep does not expand inside a bracket expression, and which would then
# flag every tab-indented Makefile recipe line as non-ASCII).
non_ascii=$'[^ -~\t]'

hits=""
while IFS= read -r file; do
    match=$(LC_ALL=C grep -n "$non_ascii" "$file" | LC_ALL=C grep -vF '# ascii-ok' || true)
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

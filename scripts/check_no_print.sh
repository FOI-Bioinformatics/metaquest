#!/usr/bin/env bash
# Fail when a print( call appears anywhere under metaquest/ except metaquest/cli/base.py.
#
# stdout carries a command's result and stderr carries logging. Commands write their
# result through BaseCommand.emit / emit_json (or emit_error_json); library modules log.
# Matches print( as a call, not as part of a longer name (pprint(, _print_table( etc.).
# Usage: scripts/check_no_print.sh [ROOT]   (ROOT defaults to the repository root)
set -euo pipefail

cd "${1:-$(dirname "$0")/..}"

hits=$(grep -rnE --include='*.py' '(^|[^.a-zA-Z0-9_])print\(' metaquest | grep -v '^metaquest/cli/base\.py:' || true)

if [ -n "$hits" ]; then
    echo "$hits"
    echo "ERROR: print( outside metaquest/cli/base.py; use self.emit / self.emit_json in commands and logging elsewhere"
    exit 1
fi
echo "No print( calls outside metaquest/cli/base.py"

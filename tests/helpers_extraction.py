"""Shared test double for minimap2/samtools, used by read-extraction tests."""

import gzip
from pathlib import Path
from unittest.mock import MagicMock

UNEQUAL_WARNING = "[W::mm_bseq_read_frag2] query files have different number of records; extra records skipped."


def _fake_tools(state):
    """Stand in for minimap2/samtools: record calls and create the FASTQ files samtools would write.

    state keys: mapped (int, default 10), nonempty (flags whose output file gets a read,
    default ("-1", "-2")), unequal (bool, emit minimap2's mate-count warning).
    """

    def run(executable, args, **kwargs):
        state.setdefault("calls", []).append((executable, list(args)))
        result = MagicMock(returncode=0, stdout="", stderr="")
        if executable == "minimap2" and state.get("unequal"):
            result.stderr = UNEQUAL_WARNING
        if executable == "samtools" and args[:2] == ["view", "-c"]:
            result.stdout = f"{state.get('mapped', 10)}\n"
        if executable == "samtools" and args[0] == "fastq":
            for flag in ("-1", "-2", "-0", "-s"):
                if flag in args and flag in state.get("nonempty", ("-1", "-2")):
                    path = Path(args[args.index(flag) + 1])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(path, "wt") as handle:
                        handle.write("@r1\nACGT\n+\nIIII\n")
        return result

    return run

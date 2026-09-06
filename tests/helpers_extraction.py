"""Shared test double for minimap2/samtools/megahit/prefetch/fasterq-dump/pigz.

Used by read-extraction tests and by the download tests for the prefetch/split-3/
compression sequence in ``download_accession``.
"""

import gzip
from pathlib import Path
from unittest.mock import MagicMock

UNEQUAL_WARNING = "[W::mm_bseq_read_frag2] query files have different number of records; extra records skipped."

# Flags (across the tools this fake stands in for) whose next token is a value, not
# another flag or the positional argument.
_VALUE_FLAGS = frozenset({"-O", "--temp", "--threads", "--max-size", "-p"})


def _positional(args):
    """Return the first token in ``args`` that is neither a flag nor a flag's value.

    Works regardless of where the positional argument (an SRA accession, or a path to
    a ``.sra``/FASTQ file) sits in the list, since real callers order it differently
    for the prefetch and the direct fasterq-dump command shapes.
    """
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token in _VALUE_FLAGS:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        return token
    return None


def _fake_tools(state):
    """Stand in for minimap2/samtools/megahit/prefetch/fasterq-dump/pigz.

    Records every call and creates the files each real tool would write.

    state keys: mapped (int, default 10, the post-filter/kept count reported for a BAM
    count), mapped_total (int, the pre-filter mapped count reported for a SAM count;
    defaults to ``mapped``), nonempty (flags whose output file gets a read, default
    ("-1", "-2")), unequal (bool, emit minimap2's mate-count warning), single (bool,
    fasterq-dump writes a single ``<acc>.fastq`` instead of a ``<acc>_1.fastq``/
    ``<acc>_2.fastq`` pair), reads (int, records per FASTQ file fasterq-dump writes,
    default 4).
    """

    def run(executable, args, **kwargs):
        state.setdefault("calls", []).append((executable, list(args)))
        result = MagicMock(returncode=0, stdout="", stderr="")
        if executable == "minimap2":
            if "-d" in args:
                # Building an index: create the .mmi the real tool would write.
                index_path = Path(args[args.index("-d") + 1])
                index_path.parent.mkdir(parents=True, exist_ok=True)
                index_path.write_bytes(b"")
            elif "-o" in args:
                # Aligning: create the SAM output the real tool would write.
                sam_path = Path(args[args.index("-o") + 1])
                sam_path.parent.mkdir(parents=True, exist_ok=True)
                sam_path.write_text("")
                if state.get("unequal"):
                    result.stderr = UNEQUAL_WARNING
        if executable == "samtools" and args[:2] == ["view", "-c"]:
            target = args[-1]
            if str(target).endswith(".sam"):
                result.stdout = f"{state.get('mapped_total', state.get('mapped', 10))}\n"
            else:
                result.stdout = f"{state.get('mapped', 10)}\n"
        if executable == "samtools" and args[0] == "view" and "-b" in args:
            state.setdefault("view_filter_calls", []).append(list(args))
            out_path = Path(args[args.index("-o") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"")
        if executable == "samtools" and args[0] == "cat":
            out_path = Path(args[args.index("-o") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"")
        if executable == "samtools" and args[0] == "fastq":
            for flag in ("-1", "-2", "-0", "-s"):
                if flag in args and flag in state.get("nonempty", ("-1", "-2")):
                    path = Path(args[args.index(flag) + 1])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(path, "wt") as handle:
                        handle.write("@r1\nACGT\n+\nIIII\n")
        if executable == "megahit":
            if "--version" in args:
                result.stdout = ""
            elif "-o" in args:
                out_dir = Path(args[args.index("-o") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "final.contigs.fa").write_text(">c1 len=100\nACGT\n>c2 len=50\nACGT\n")
        if executable == "prefetch":
            accession = _positional(args)
            cache_dir = Path(args[args.index("-O") + 1])
            acc_dir = cache_dir / accession
            acc_dir.mkdir(parents=True, exist_ok=True)
            # NCBI serves some runs only as the smaller .sralite format.
            suffix = ".sralite" if state.get("sralite") else ".sra"
            (acc_dir / f"{accession}{suffix}").write_bytes(b"")
        if executable == "fasterq-dump":
            positional = _positional(args)
            accession = Path(positional).stem if positional.endswith((".sra", ".sralite")) else positional
            out_dir = Path(args[args.index("-O") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            record = "@r\nACGT\n+\nIIII\n" * state.get("reads", 4)
            if state.get("single"):
                (out_dir / f"{accession}.fastq").write_text(record)
            else:
                (out_dir / f"{accession}_1.fastq").write_text(record)
                (out_dir / f"{accession}_2.fastq").write_text(record)
        if executable == "pigz":
            path = Path(_positional(args))
            target = path.with_suffix(path.suffix + ".gz")
            with open(path, "rb") as src, gzip.open(target, "wb") as dst:
                dst.write(src.read())
            path.unlink()
        return result

    return run

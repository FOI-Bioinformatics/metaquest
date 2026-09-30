"""Guards for the split of the assembly code out of ``metaquest.data.read_extraction``.

The assembly functions live in ``metaquest.data.assembly``; ``read_extraction`` re-exports them so
existing imports keep working. ``assembly`` must not import ``read_extraction`` when it loads, so the
two modules cannot form an import cycle.
"""

import subprocess
import sys
from pathlib import Path

import metaquest.data.assembly as assembly
import metaquest.data.read_extraction as read_extraction

MOVED_NAMES = (
    "_megahit_args",
    "assemble_extracted_reads",
    "fasta_length",
    "megahit_version",
    "resolve_assembly_threads",
    "summarise_contigs",
)


def test_moved_names_are_importable_from_both_modules():
    for name in MOVED_NAMES:
        assert getattr(read_extraction, name) is getattr(assembly, name), name


def test_moved_functions_are_defined_in_the_assembly_module():
    for name in MOVED_NAMES:
        assert getattr(assembly, name).__module__ == "metaquest.data.assembly", name


def test_assembly_imports_without_loading_read_extraction():
    code = (
        "import sys\n"
        "import metaquest.data.assembly\n"
        "sys.exit(1 if 'metaquest.data.read_extraction' in sys.modules else 0)\n"
    )
    # Run from the repository root so ``-c`` finds this checkout's package first.
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=repo_root, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr

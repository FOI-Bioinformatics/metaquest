"""Guards for the layout of the ``metaquest.data.sra`` package.

The package re-exports the public names other packages import, and every string patch target
in the tests must name an attribute that exists on the module it names, so a patch never
silently targets a name the code no longer looks up there.
"""

import importlib
import pathlib
import re


def test_public_names_are_importable_from_the_package():
    from metaquest.data import sra

    for name in (
        "download_sra",
        "download_accession",
        "fastq_files",
        "count_fastq_reads",
        "iter_fastq_records",
        "verify_download",
        "parse_verdict_message",
        "compress_fastq",
        "transient_bytes",
        "is_transient_folder",
        "default_max_workers",
        "STORE_READY_STATES",
        "STORE_LINKED_PREFIX",
        "MATE1_SUFFIXES",
        "STOP",
    ):
        assert hasattr(sra, name), name


def test_patch_targets_in_tests_name_existing_attributes():
    pattern = re.compile(r'patch\("(metaquest\.data\.sra(?:\.[a-z_]+)?)\.([A-Za-z_]+)"')
    for test_file in pathlib.Path("tests").glob("*.py"):
        for module_name, attr in pattern.findall(test_file.read_text()):
            module = importlib.import_module(module_name)
            assert hasattr(module, attr), f"{test_file}: {module_name}.{attr}"

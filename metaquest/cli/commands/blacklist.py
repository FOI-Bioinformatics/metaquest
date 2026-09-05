"""CLI command that records datasets excluded on purpose, with a reason."""

import argparse
from pathlib import Path
from typing import Dict, List

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.registry import clear_exclusion, load_registry, query, record_exclusion, save_registry


def read_blacklist_file(path: Path) -> Dict[str, str]:
    """Accession -> reason from a blacklist file (``ACC  # reason`` lines; comments and blanks skipped)."""
    entries: Dict[str, str] = {}
    if not path.exists():
        return entries
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        accession, _, reason = text.partition("#")
        entries[accession.strip()] = reason.strip()
    return entries


def write_blacklist_file(path: Path, entries: Dict[str, str]) -> None:
    path.write_text(
        "".join(f"{acc}  # {reason}\n" if reason else f"{acc}\n" for acc, reason in sorted(entries.items()))
    )


class BlacklistCommand(BaseCommand):
    """Exclude datasets from downloads and extraction, recording why."""

    @property
    def name(self) -> str:
        return "blacklist"

    @property
    def help(self) -> str:
        return "Record datasets to exclude, with a reason; keeps blacklist.txt and the registry in step"

    @property
    def group(self) -> str:
        return "Reads"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        action = parser.add_mutually_exclusive_group(required=True)
        action.add_argument("--add", nargs="+", metavar="ACCESSION", help="Accessions to exclude")
        action.add_argument("--remove", nargs="+", metavar="ACCESSION", help="Accessions to allow again")
        action.add_argument("--from-file", help="File of accessions to exclude, one per line")
        action.add_argument("--list", action="store_true", help="Show the excluded accessions and reasons")
        parser.add_argument(
            "--reason", default=None, help="Why the accessions are excluded (required with --add and --from-file)"
        )
        parser.add_argument(
            "--blacklist-file", default="blacklist.txt", help="Plain list kept for download_sra --blacklist"
        )
        parser.add_argument("--registry", default=None, help="Registry file (default: found upwards from here)")

    def execute(self, args: argparse.Namespace) -> int:
        try:
            registry = load_registry(args.registry)
            blacklist_path = Path(args.blacklist_file)
            entries = read_blacklist_file(blacklist_path)
            if args.list:
                for acc in query(registry, "excluded"):
                    print(f"{acc}\t{registry.datasets[acc]['exclusion'].get('reason', '')}")
                return 0
            if args.remove:
                for acc in args.remove:
                    clear_exclusion(registry, acc)
                    entries.pop(acc, None)
                self.logger.info("Removed %d accession(s) from the blacklist", len(args.remove))
            else:
                if not args.reason:
                    raise MetaQuestError("--reason is required when adding to the blacklist")
                accessions: List[str] = args.add or _read_plain_list(Path(args.from_file))
                for acc in accessions:
                    record_exclusion(registry, acc, args.reason)
                    entries[acc] = args.reason
                self.logger.info("Excluded %d accession(s): %s", len(accessions), args.reason)
            write_blacklist_file(blacklist_path, entries)
            save_registry(registry)
            return 0
        except MetaQuestError as e:
            self.logger.error("Error updating the blacklist: %s", e)
            return 1


def _read_plain_list(path: Path) -> List[str]:
    if not path.exists():
        raise MetaQuestError(f"Accession file not found: {path}")
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]

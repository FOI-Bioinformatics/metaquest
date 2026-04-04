"""Genome search, download, and preparation CLI commands."""

import argparse
import csv
import sys
from pathlib import Path

from metaquest.cli.base import BaseCommand
from metaquest.core.exceptions import MetaQuestError
from metaquest.data.genome_download import download_genomes, extract_and_organize
from metaquest.data.gtdb import (
    get_accessions_for_genus,
    get_accessions_for_species,
)


class GenomeSearchCommand(BaseCommand):
    """Command to search GTDB for genome accessions."""

    @property
    def name(self) -> str:
        return "genome_search"

    @property
    def help(self) -> str:
        return "Search GTDB for genome accessions by species or genus"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--species",
            help="Species name to search (e.g. 'Salmonella enterica')",
        )
        group.add_argument(
            "--genus",
            help="Genus name to search (e.g. 'Salmonella')",
        )
        parser.add_argument(
            "--output",
            default=None,
            help="Output file path (default: print to stdout)",
        )
        parser.add_argument(
            "--all",
            dest="representative_only",
            action="store_false",
            default=True,
            help="Include all genomes, not just representatives",
        )
        parser.add_argument(
            "--format",
            dest="output_format",
            choices=["list", "tsv"],
            default="list",
            help="Output format",
        )

    def execute(self, args: argparse.Namespace) -> int:
        try:
            if args.species:
                accessions = get_accessions_for_species(
                    args.species, representative_only=args.representative_only
                )
                label = args.species
            else:
                accessions = get_accessions_for_genus(
                    args.genus, representative_only=args.representative_only
                )
                label = args.genus

            if not accessions:
                self.logger.warning("No accessions found for '%s'", label)
                return 0

            self.logger.info("Found %d accessions for '%s'", len(accessions), label)

            if args.output_format == "tsv":
                lines = [f"{acc}\t{label}" for acc in accessions]
            else:
                lines = accessions

            output_text = "\n".join(lines) + "\n"

            if args.output:
                Path(args.output).write_text(output_text)
                self.logger.info("Wrote accessions to %s", args.output)
            else:
                sys.stdout.write(output_text)

            return 0
        except MetaQuestError as e:
            self.logger.error("Error searching genomes: %s", e)
            return 1


class GenomeDownloadCommand(BaseCommand):
    """Command to download genome assemblies from NCBI."""

    @property
    def name(self) -> str:
        return "genome_download"

    @property
    def help(self) -> str:
        return "Download genome assemblies from NCBI by accession, species, or genus"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--accessions",
            nargs="+",
            help="One or more assembly accessions (e.g. GCF_000006945.2)",
        )
        parser.add_argument(
            "--accession-file",
            help="Text file with one accession per line",
        )
        parser.add_argument(
            "--species",
            help="Species name to search GTDB and download genomes",
        )
        parser.add_argument(
            "--genus",
            help="Genus name to search GTDB and download genomes",
        )
        parser.add_argument(
            "--output-dir",
            default="genomes/",
            help="Directory to save downloaded genomes",
        )
        parser.add_argument(
            "--all",
            dest="representative_only",
            action="store_false",
            default=True,
            help="Include all genomes, not just representatives",
        )
        parser.add_argument(
            "--assembly-level",
            choices=["complete", "chromosome", "scaffold", "contig"],
            default=None,
            help="Filter by assembly level",
        )

    def _collect_accessions(self, args: argparse.Namespace) -> list:
        """Collect accessions from all input sources."""
        accessions = []

        if args.accessions:
            accessions.extend(args.accessions)

        if args.accession_file:
            path = Path(args.accession_file)
            if not path.exists():
                raise MetaQuestError(f"Accession file not found: {path}")
            for line in path.read_text().strip().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    accessions.append(line)

        if args.species:
            accessions.extend(
                get_accessions_for_species(
                    args.species, representative_only=args.representative_only
                )
            )

        if args.genus:
            accessions.extend(
                get_accessions_for_genus(
                    args.genus, representative_only=args.representative_only
                )
            )

        return accessions

    def execute(self, args: argparse.Namespace) -> int:
        try:
            if not any([args.accessions, args.accession_file, args.species, args.genus]):
                self.logger.error(
                    "At least one of --accessions, --accession-file, --species, "
                    "or --genus is required"
                )
                return 1

            accessions = self._collect_accessions(args)
            if not accessions:
                self.logger.warning("No accessions to download")
                return 0

            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(
                "Downloading %d genome(s) to %s", len(accessions), output_dir
            )

            zip_path = download_genomes(
                accessions,
                output_dir,
                assembly_level=args.assembly_level,
            )

            genome_paths = extract_and_organize(zip_path, output_dir)
            self.logger.info(
                "Downloaded and extracted %d genome(s)", len(genome_paths)
            )
            return 0
        except MetaQuestError as e:
            self.logger.error("Error downloading genomes: %s", e)
            return 1


class GenomePrepareCommand(BaseCommand):
    """Command to search, download, and prepare a genome manifest."""

    @property
    def name(self) -> str:
        return "genome_prepare"

    @property
    def help(self) -> str:
        return "Search GTDB, download genomes, and create a manifest CSV"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--species",
            help="Species name to search GTDB",
        )
        group.add_argument(
            "--genus",
            help="Genus name to search GTDB",
        )
        parser.add_argument(
            "--accession-file",
            help="Text file with one accession per line",
        )
        parser.add_argument(
            "--output-dir",
            default="genomes/",
            help="Directory for downloaded genomes",
        )
        parser.add_argument(
            "--all",
            dest="representative_only",
            action="store_false",
            default=True,
            help="Include all genomes, not just representatives",
        )
        parser.add_argument(
            "--manifest-file",
            default="genome_manifest.csv",
            help="Output manifest file path",
        )
        parser.add_argument(
            "--skip-download",
            action="store_true",
            help="Only create manifest from existing files in output-dir",
        )

    def _collect_accessions(self, args: argparse.Namespace) -> list:
        """Collect accessions from GTDB or accession file."""
        accessions = []

        if args.accession_file:
            path = Path(args.accession_file)
            if not path.exists():
                raise MetaQuestError(f"Accession file not found: {path}")
            for line in path.read_text().strip().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    accessions.append(line)

        if args.species:
            accessions.extend(
                get_accessions_for_species(
                    args.species, representative_only=args.representative_only
                )
            )

        if args.genus:
            accessions.extend(
                get_accessions_for_genus(
                    args.genus, representative_only=args.representative_only
                )
            )

        return accessions

    def _create_manifest(self, output_dir: Path, manifest_file: str) -> int:
        """Create a manifest CSV from genome files in output_dir."""
        genome_files = sorted(output_dir.glob("*.fna.gz")) + sorted(
            output_dir.glob("*.fasta.gz")
        )
        if not genome_files:
            self.logger.warning("No genome files found in %s", output_dir)
            return 0

        manifest_path = Path(manifest_file)
        with open(manifest_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "genome_filename", "protein_filename"])
            for gf in genome_files:
                name = gf.stem.replace(".fna", "").replace(".fasta", "")
                protein_file = gf.with_suffix("").with_suffix(".faa.gz")
                protein_name = str(protein_file) if protein_file.exists() else ""
                writer.writerow([name, str(gf), protein_name])

        self.logger.info(
            "Created manifest with %d entries: %s", len(genome_files), manifest_path
        )
        return len(genome_files)

    def execute(self, args: argparse.Namespace) -> int:
        try:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if not args.skip_download:
                if not any([args.species, args.genus, args.accession_file]):
                    self.logger.error(
                        "At least one of --species, --genus, or --accession-file "
                        "is required when not using --skip-download"
                    )
                    return 1

                accessions = self._collect_accessions(args)
                if not accessions:
                    self.logger.warning("No accessions found")
                    return 0

                self.logger.info(
                    "Downloading %d genome(s) to %s", len(accessions), output_dir
                )

                zip_path = download_genomes(accessions, output_dir)
                extract_and_organize(zip_path, output_dir)

            self._create_manifest(output_dir, args.manifest_file)
            return 0
        except MetaQuestError as e:
            self.logger.error("Error preparing genomes: %s", e)
            return 1

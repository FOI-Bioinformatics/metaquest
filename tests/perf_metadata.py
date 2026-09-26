"""Synthetic NCBI efetch metadata XML files for the metadata parsing tests.

Each file follows the layout of a real single-run ``EXPERIMENT_PACKAGE_SET`` response (experiment,
submission, organization, study, sample with SAMPLE_ATTRIBUTES, pool, run with SRAFiles and read
statistics), at roughly the size of a real file. Attribute sets overlap between files: every file
carries a shared core of tags plus a window of tags that shifts with the file index, so the folder as
a whole holds many more distinct tags than any one file.
"""

from pathlib import Path
from typing import List
from xml.sax.saxutils import escape

CORE_TAGS = [
    "collection_date",
    "geo_loc_name",
    "lat_lon",
    "env_broad_scale",
    "env_local_scale",
    "env_medium",
    "host",
    "isolation_source",
    "sample_type",
    "strain",
]

_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" ?>
<EXPERIMENT_PACKAGE_SET>
<EXPERIMENT_PACKAGE>
<EXPERIMENT accession="SRX{n}" alias="exp_{n}">
<IDENTIFIERS><PRIMARY_ID>SRX{n}</PRIMARY_ID></IDENTIFIERS>
<TITLE>Illumina sequencing of sample {n}</TITLE>
<STUDY_REF accession="SRP{p}"><IDENTIFIERS><PRIMARY_ID>SRP{p}</PRIMARY_ID></IDENTIFIERS></STUDY_REF>
<DESIGN>
<DESIGN_DESCRIPTION>Shotgun metagenome of sample {n}</DESIGN_DESCRIPTION>
<SAMPLE_DESCRIPTOR accession="SRS{n}"><IDENTIFIERS><PRIMARY_ID>SRS{n}</PRIMARY_ID></IDENTIFIERS></SAMPLE_DESCRIPTOR>
<LIBRARY_DESCRIPTOR>
<LIBRARY_NAME>lib_{n}</LIBRARY_NAME>
<LIBRARY_STRATEGY>{strategy}</LIBRARY_STRATEGY>
<LIBRARY_SOURCE>METAGENOMIC</LIBRARY_SOURCE>
<LIBRARY_SELECTION>RANDOM</LIBRARY_SELECTION>
<LIBRARY_LAYOUT>{layout}</LIBRARY_LAYOUT>
</LIBRARY_DESCRIPTOR>
</DESIGN>
<PLATFORM><ILLUMINA><INSTRUMENT_MODEL>Illumina NovaSeq 6000</INSTRUMENT_MODEL></ILLUMINA></PLATFORM>
</EXPERIMENT>
<SUBMISSION accession="SRA{p}" lab_name="Lab {p}">\
<IDENTIFIERS><PRIMARY_ID>SRA{p}</PRIMARY_ID></IDENTIFIERS></SUBMISSION>
<Organization type="institute"><Name>Institute {p}</Name><Contact email="lab{p}@example.org"><Name><First>A</First>\
<Last>B</Last></Name></Contact></Organization>
<STUDY accession="SRP{p}" alias="study_{p}">
<IDENTIFIERS><PRIMARY_ID>SRP{p}</PRIMARY_ID><EXTERNAL_ID namespace="BioProject">PRJNA{p}</EXTERNAL_ID></IDENTIFIERS>
<DESCRIPTOR>
<STUDY_TITLE>Metagenomes of project {p}</STUDY_TITLE>
<STUDY_TYPE existing_study_type="Metagenomics"/>
<STUDY_ABSTRACT>{abstract}</STUDY_ABSTRACT>
</DESCRIPTOR>
</STUDY>
<SAMPLE accession="SRS{n}" alias="sample_{n}">
<IDENTIFIERS><PRIMARY_ID>SRS{n}</PRIMARY_ID><EXTERNAL_ID namespace="BioSample">SAMN{n}</EXTERNAL_ID></IDENTIFIERS>
<TITLE>Sample {n}</TITLE>
<SAMPLE_NAME><TAXON_ID>{taxon}</TAXON_ID><SCIENTIFIC_NAME>{organism}</SCIENTIFIC_NAME></SAMPLE_NAME>
<SAMPLE_LINKS><SAMPLE_LINK><XREF_LINK><DB>bioproject</DB><ID>{p}</ID></XREF_LINK></SAMPLE_LINK></SAMPLE_LINKS>
<SAMPLE_ATTRIBUTES>
{attributes}
</SAMPLE_ATTRIBUTES>
</SAMPLE>
<Pool><Member member_name="" accession="SRS{n}" sample_name="sample_{n}" spots="{spots}" bases="{bases}"\
 tax_id="{taxon}"\
 organism="{organism}"><IDENTIFIERS><PRIMARY_ID>SRS{n}</PRIMARY_ID></IDENTIFIERS></Member></Pool>
<RUN_SET runs="1" bases="{bases}" spots="{spots}" bytes="{size}">
<RUN accession="SRR{n}" alias="run_{n}" total_spots="{spots}" total_bases="{bases}" size="{size}" load_done="true"\
 published="2024-01-01" is_public="true" cluster_name="public" static_data_available="1">
<IDENTIFIERS><PRIMARY_ID>SRR{n}</PRIMARY_ID></IDENTIFIERS>
<EXPERIMENT_REF accession="SRX{n}"/>
<Pool><Member member_name="" accession="SRS{n}" sample_name="sample_{n}" spots="{spots}" bases="{bases}"\
 tax_id="{taxon}" organism="{organism}"><IDENTIFIERS><PRIMARY_ID>SRS{n}</PRIMARY_ID></IDENTIFIERS></Member></Pool>
<SRAFiles>
<SRAFile cluster="public" filename="sample_{n}_R1.fastq.gz" url="https://sra-pub-src-1.s3.amazonaws.com/SRR{n}/r1"\
 size="{half}" date="2024-01-01" md5="{md5a}" semantic_name="fastq" supertype="Original" sratoolkit="0"/>
<SRAFile cluster="public" filename="SRR{n}"\
 url="https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos5/SRR{n}/SRR{n}.lite.1"\
 size="{size}" date="2024-01-02" md5="{md5b}" version="1" semantic_name="run" supertype="Primary ETL" sratoolkit="1"/>
</SRAFiles>
<CloudFiles><CloudFile filetype="run" provider="gs" location="gs.us-east1"/>\
<CloudFile filetype="run" provider="s3" location="s3.us-east-1"/></CloudFiles>
<Statistics nreads="2" nspots="{spots}"><Read index="0" count="{spots}" average="151" stdev="0"/>\
<Read index="1" count="{spots}" average="151" stdev="0"/></Statistics>
<Bases cs_native="false" count="{bases}"><Base value="A" count="1"/><Base value="C" count="1"/>\
<Base value="G" count="1"/><Base value="T" count="1"/><Base value="N" count="0"/></Bases>
</RUN>
</RUN_SET>
</EXPERIMENT_PACKAGE>
</EXPERIMENT_PACKAGE_SET>
"""


def attribute_tags(index: int, per_file: int = 40, pool: int = 1000) -> List[str]:
    """The SAMPLE_ATTRIBUTE tags of file ``index``: the shared core plus a shifting window of the pool."""
    window = per_file - len(CORE_TAGS)
    start = (index * 7) % pool
    return CORE_TAGS + [f"attr_{(start + k) % pool:03d}" for k in range(window)]


def metadata_xml(index: int, per_file: int = 40, pool: int = 1000) -> str:
    """One synthetic single-run metadata XML document with ``per_file`` sample attributes."""
    n = 1000000 + index
    spots = 1000000 + index * 13
    bases = spots * 302
    size = bases // 3
    attributes = "\n".join(
        f"<SAMPLE_ATTRIBUTE><TAG>{tag}</TAG><VALUE>{escape(f'value {index} of {tag}')}</VALUE></SAMPLE_ATTRIBUTE>"
        for tag in attribute_tags(index, per_file, pool)
    )
    return _TEMPLATE.format(
        n=n,
        p=500000 + index // 25,
        strategy="WGS" if index % 5 else "AMPLICON",
        layout="<PAIRED/>" if index % 3 else "<SINGLE/>",
        abstract=escape("Shotgun sequencing of environmental samples. " * 20),
        taxon=256318 + index % 4,
        organism="metagenome" if index % 4 else "soil metagenome",
        attributes=attributes,
        spots=spots,
        bases=bases,
        size=size,
        half=size // 2,
        md5a=f"{index:032x}",
        md5b=f"{index + 1:032x}",
    )


def write_metadata_folder(folder: Path, count: int = 300, per_file: int = 40, pool: int = 1000) -> List[Path]:
    """Write ``count`` synthetic metadata XML files named ``SRR<n>_metadata.xml`` into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = folder / f"SRR{1000000 + index}_metadata.xml"
        path.write_text(metadata_xml(index, per_file, pool))
        paths.append(path)
    return paths

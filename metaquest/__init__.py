"""
MetaQuest - A toolkit for analyzing metagenomic datasets based on genome containment.

MetaQuest helps users search through SRA datasets to find containment of specified
genomes and analyze associated metadata.
"""

import logging

__version__ = "0.6.0"
__author__ = "Andreas Sjodin"
__email__ = "andreas.sjodin@gmail.com"

# A library host (or a process that only imports metaquest, such as a test collector)
# owns its own logging configuration; importing this package must not call
# setup_logging() and reconfigure the root logger out from under it. This NullHandler
# only silences the "No handlers could be found" warning Python emits when nothing has
# configured the "metaquest" logger at all. The CLI calls setup_logging() explicitly in
# metaquest.cli.main.main(); a library user who wants console/file logging calls it too.
logging.getLogger("metaquest").addHandler(logging.NullHandler())

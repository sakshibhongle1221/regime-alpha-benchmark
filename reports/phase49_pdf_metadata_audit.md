# Phase 49 PDF Metadata Audit

The audit scans the compiled Phase 49 PDF metadata for obvious identity leaks such as personal names, local Windows paths, repository links, and username/project-path markers.

| check_id | status | detail |
| --- | --- | --- |
| pdf_exists | pass | paper\phase49_submission_ready_manuscript.pdf |
| pdf_size | pass | bytes=428155 |
| metadata_title | pass | \376\377\000L\000e\000a\000k\000a\000g\000e\000-\000S\000a\000f\000e\000\040\000R\000e\000g\000i\000m\000e\000-\000C\000o\000n\000d\000i\000t\000i\000o\000n\000e\000d\000\040\000C\000r\000y\000p\000t\000o\000\040\000A\000l\000p\000h\000a\000\040\000E\000v\000a\000l\000u\000a\000t\000i\000o\000n\000\040\000w\000i\000t\000h\000\040\000H\000M\000M\000-\000G\000u\000i\000d\000e\000d\000\040\000C\000o\000n\000t\000r\000a\000s\000t\000i\000v\000e\000\040\000R\000e\000p\000r\000e\000s\000e\000n\000t\000a\000t\000i\000o\000n\000s |
| metadata_author | pass | <empty> |
| metadata_subject | pass | \376\377\000-\000\040\000\040\000C\000o\000m\000p\000u\000t\000i\000n\000g\000\040\000m\000e\000t\000h\000o\000d\000o\000l\000o\000g\000i\000e\000s\000\040\000\040\000-\000>\000\040\000\040\000S\000u\000p\000e\000r\000v\000i\000s\000e\000d\000\040\000l\000e\000a\000r\000n\000i\000n\000g\000;\000\040\000-\000\040\000\040\000A\000p\000p\000l\000i\000e\000d\000\040\000c\000o\000m\000p\000u\000t\000i\000n\000g\000\040\000\040\000-\000>\000\040\000\040\000E\000c\000o\000n\000o\000m\000i\000c\000s\000;\000\040 |
| metadata_creator | pass | Matplotlib v3.10.9, https://matplotlib.org |
| metadata_producer | pass | Matplotlib pdf backend v3.10.9 |
| metadata_keywords | pass | <empty> |

Manual confirmation is still required before external upload because submission systems can transform PDFs or add metadata.

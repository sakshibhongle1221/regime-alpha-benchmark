# Phase 47 Blind-Review Hardening

## Purpose

This report checks the Phase 47 LaTeX source and BibTeX source for obvious double-blind risks. It is a source-level audit only; the compiled PDF content and metadata must still be checked before external submission.

| check_id | status | action |
| --- | --- | --- |
| author_block | pass | No author or affiliation block in review draft. |
| acknowledgements | pass | Acknowledgements must be absent for double-blind review. |
| github_link | pass | Repository links can reveal identity during review. |
| personal_name_sakshi | pass | Personal name must not appear in anonymous source. |
| institution_marker | pass | Institution/student identity wording should not appear. |
| tool_marker | pass | Tool names should not appear in the anonymous paper story. |

## Required Human Check Before Submission

- Compile the PDF and inspect title page, headers, footers, metadata, and links.
- Remove acknowledgements from the review version.
- Avoid public repository links in the anonymous paper body.
- Verify that artifact archive links are allowed by the venue's double-blind policy before adding them.

# Phase 48 LaTeX / PDF Warning Resolution Plan

## Current warning audit

| check_id | status | detail |
| --- | --- | --- |
| pdf_compilation | pass | pdf_exists=True; pages=2; bytes=369573 |
| build_logs_available | pass | latex_log=True; bibtex_log=True |
| warning_cleanup | review_required | warnings=Package balance Warning |
| page_budget | pass | pages=2; active venue limit must still be verified. |

## Required before external submission

1. Recompile from a clean LaTeX build directory.
2. Inspect the PDF visually page by page.
3. Remove or resolve any overfull table/text boxes.
4. Confirm ACM reference/copyright requirements for the active venue.
5. Check PDF metadata for author, institution, local path, and software identity leakage.
6. Re-run the research-grade gate after any manuscript source change.

This plan is intentionally conservative: a compiled PDF is not the same as an externally submissible PDF.

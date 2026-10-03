# Phase 46 Anonymity Audit

## Status

Potential review items found: 0

This audit is run on the Markdown manuscript source. The final mandatory audit must be run again on the compiled PDF and metadata before external double-blind submission.

| check_id | pattern | status | action |
| --- | --- | --- | --- |
| author_block | \\author\s*\{ | (?im)^#+\s*authors?\b | (?im)^authors?\s*: | pass | No explicit author block should appear in review manuscript. |
| acknowledgements | (?im)^#+\s*acknowledg | (?im)^acknowledg(e)?ments?\s*: | pass | Acknowledgements should be removed for double-blind review. |
| github_link | github\.com | pass | Public repository links can reveal identity during review. |
| personal_name_sakshi | \bSakshi\b | pass | Personal names must not appear in the anonymous review PDF. |
| institution_marker | \bBTech\b | pass | Student/institution context is useful for viva but should be removed from anonymous paper. |
| codex_marker | \bCodex\b | pass | Tooling names should not appear in the anonymous paper claim story. |

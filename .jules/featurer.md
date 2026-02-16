# Featurer's Journal - Critical Learnings Only

## 2025-05-15 - [Visualization Feature Implementation]
Learning: When adding a new output type to a pipeline script that already has smart resume/skip logic (like `check_scene_needs_processing` and `files_exist_for_name`), it is critical to update the skip logic as well. Otherwise, if a user re-runs with the new flag, the script will skip the scenes that already have images but are missing the new visualization.
Action: Always check for nested `files_exist_for_name` or similar functions when adding new CLI-controlled output files.

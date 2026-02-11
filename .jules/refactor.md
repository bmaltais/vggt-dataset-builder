## 2025-02-11 - Monolithic Main with State Management
**Learning:** Monolithic scripts often contain deeply nested loops with mutable state (like GPU renderers) that make extraction difficult. Using a dictionary to wrap mutable state (e.g., `renderer_info`) allows for clean function extraction while preserving performance (avoiding re-initialization).
**Action:** Identify mutable state early and wrap in a context dictionary or object when extracting functions to avoid breaking re-use patterns.

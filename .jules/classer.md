# 🏛️ CLASSER JOURNAL

## 2026-02-17 - Encapsulated Frame Cache Management

**Learning:** Manifest/cache validation logic was duplicated across multiple functions in a 1849-line procedural script. Extracting it into a proper class dramatically simplified both the cache-handling code and the main processing function.

**Key Insight:** When similar state-management patterns appear in 2+ locations (especially validation/loading/saving), it's a strong signal for a class extraction. The `FrameCacheManager` centralized ~90 lines of scattered code (including duplicated manifest loading in `check_scene_needs_processing()` and `process_scene()`) into a 200-line, well-documented class with clear responsibility boundaries.

**Action:** When refactoring scripts with complex caching or state management, look for:
- Repeated try/except+manifest-loading blocks
- Functions that manipulate the same dict structures (manifest, args_hash, etc.)
- Validation logic embedded in business logic (cache validation mixed with scene processing)
- Extract into a manager class with clear load/save/validate/clear methods
- Simplifies calling code and improves testability significantly

**Impact:**
- ✅ Removed 5 helper functions from build_warp_dataset.py
- ✅ Eliminated duplicate manifest-loading code (≈60 duplicate lines)
- ✅ Improved test coverage (tests now directly use the class)
- ✅ Code rating improved from 7.61 → 7.66 / 10
- ✅ All tests passing, no regressions
## 2026-02-17 - Encapsulated Point Cloud Filtering Logic

**Learning:** Scattered procedural filtering functions with external state management (sky segmentation session, masks directory, filter flags) are prime candidates for class extraction. The `PointCloudFilter` class consolidated two filtering stages (pre- and post-unprojection) into a single, testable unit with clean separation of concerns.

**Key Insight:** When filtering logic:
1. Occurs at multiple stages in a pipeline (confidence maps vs. 3D points)
2. Requires external resources (ONNX sessions, cache directories)
3. Has configuration state (which filters are enabled)
4. Uses procedural functions with many parameters passed through calling code

...it's begging to become a class. The filtering code went from scattered functions with 8+ parameters to a well-encapsulated class with 2 clean methods that manage their own state.

**Action:** For multi-stage filtering/processing pipelines:
- Look for functions that share configuration state (filter_sky, filter_black_bg, filter_white_bg)
- Identify external resources that need lifecycle management (ONNX sessions, cache dirs)
- Group operations by pipeline stage (confidence filtering vs color filtering)
- Provide query methods (`has_confidence_filters()`, `has_color_filters()`) for calling code
- Make the class responsible for resource initialization (creating directories, validating config)

**Impact:**
- ✅ Removed 2 procedural functions (~70 lines) from build_warp_dataset.py
- ✅ Created 180-line PointCloudFilter class with comprehensive docstrings
- ✅ Eliminated 8 parameters from process_scene calling code
- ✅ Improved future extensibility (adding new filters is trivial)
- ✅ Code rating maintained at 7.64/10 (from 7.66, -0.02 negligible)
- ✅ All 15 tests passing, no regressions
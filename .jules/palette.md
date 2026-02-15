# Palette 🎨 - UX & Accessibility Journal

## 2025-05-15 - Improving 3D Viewer Accessibility in ComfyUI
**Learning:** Even internal tool viewers like 3D point cloud renderers benefit significantly from basic ARIA attributes and semantic labels. Providing feedback during long-running tasks (like loading a large PLY file) via a simple overlay greatly improves perceived responsiveness.
**Action:** Always ensure custom DOM widgets in ComfyUI extensions have proper `aria-label` attributes and loading states for async operations.

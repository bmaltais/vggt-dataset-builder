from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np


@dataclass
class RenderPass:
    fbo: moderngl.Framebuffer
    color_textures: list[moderngl.Texture]
    depth_texture: moderngl.Texture | None
    width: int
    height: int


class HoleFillingRenderer:
    """GPU-accelerated renderer for point clouds with hierarchical hole filling.

    This renderer converts 3D point clouds into 2D images using a multi-pass pipeline
    implemented with ModernGL. It handles occlusions and fills holes caused by
    sparsity or disocclusions in the point cloud.

    The rendering pipeline consists of several key stages:
    1.  **Points Pass**: Initial rendering of 3D points into multiple textures:
        color, camera-space positions, and world-space positions.
    2.  **Downsample Passes**: Creates a multi-scale pyramid of the rendered textures.
    3.  **HPR (Hidden Point Removal)**: A visibility estimation pass that determines
        which points are occluded by others, based on their density and distance.
    4.  **JFA (Jump Flooding Algorithm)**: Computes a distance transform and nearest-
        neighbor field to provide smooth masks for hole filling.
    5.  **Push-Pull**: A hierarchical algorithm that fills holes by "pushing" valid
        colors to coarser levels and "pulling" them back to fill gaps at finer levels.
    6.  **Final Mask**: Blends the original points with the hole-filled result using
        the JFA-computed distance mask.

    Attributes:
        max_level (int): Maximum depth of the image pyramid for push-pull.
    """

    max_level = 8

    def __init__(
        self,
        width: int,
        height: int,
        shaders_dir: Path | None = None,
        confidence_threshold: float = 1.1,
        s0: float = 0.005,
        occlusion_threshold: float = 0.1,
        coarse_level: int = 4,
        jfa_mask_sigma: float = 32.0,
    ) -> None:
        """Initializes the renderer and its GPU resources.

        Args:
            width: Output image width in pixels.
            height: Output image height in pixels.
            shaders_dir: Directory containing the GLSL shader files.
                Defaults to a "shaders" subdirectory relative to this file.
            confidence_threshold: Minimum depth confidence required to render a point.
            s0: Base splat size parameter used for HPR and rendering.
            occlusion_threshold: Threshold for determining if a point is occluded in HPR.
            coarse_level: The level in the pyramid where hole filling starts being aggressive.
            jfa_mask_sigma: Sigma for the Gaussian falloff of the JFA-based hole filling mask.
        """
        self.width = int(width)
        self.height = int(height)
        self.confidence_threshold = float(confidence_threshold)
        self.s0 = float(s0)
        self.occlusion_threshold = float(occlusion_threshold)
        self.coarse_level = int(coarse_level)
        self.jfa_mask_sigma = float(jfa_mask_sigma)

        self.ctx = moderngl.create_standalone_context(require=410)
        self.ctx.point_size = 1.0

        if shaders_dir is None:
            shaders_dir = Path(__file__).parent / "shaders"

        shaders_dir = Path(shaders_dir).resolve()
        if not shaders_dir.exists():
            raise FileNotFoundError(f"Shaders directory not found: {shaders_dir}")

        self.points_program = self._load_program(
            shaders_dir / "points.vert", shaders_dir / "points.frag"
        )
        quad_vertex = (shaders_dir / "quad.vert").read_text(encoding="utf-8")

        self.downsample_program = self._load_program_from_source(
            quad_vertex, shaders_dir / "downsample.frag"
        )
        self.hpr_program = self._load_program_from_source(
            quad_vertex, shaders_dir / "hpr.frag"
        )
        self.multiply_program = self._load_program_from_source(
            quad_vertex, shaders_dir / "multiply_visibility.frag"
        )
        self.push_program = self._load_program_from_source(
            quad_vertex, shaders_dir / "push_color.frag"
        )
        self.pull_program = self._load_program_from_source(
            quad_vertex, shaders_dir / "pull_color.frag"
        )
        self.jfa_init_program = self._load_program_from_source(
            quad_vertex, shaders_dir / "jfa_init.frag"
        )
        self.jfa_step_program = self._load_program_from_source(
            quad_vertex, shaders_dir / "jfa_step.frag"
        )
        self.jfa_resolve_program = self._load_program_from_source(
            quad_vertex, shaders_dir / "jfa_resolve.frag"
        )
        self.final_mask_program = self._load_program_from_source(
            quad_vertex, shaders_dir / "jfa_distance_mask.frag"
        )

        self._quad_pos_buffer = self.ctx.buffer(
            np.array(
                [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32
            ).tobytes()
        )
        self._quad_tex_buffer = self.ctx.buffer(
            np.array(
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32
            ).tobytes()
        )
        self._quad_vaos: dict[int, moderngl.VertexArray] = {}

        self.points_pass = self._init_points_pass()
        self.downsample_passes = self._init_downsample_passes()
        self.hpr_pass = self._init_single_pass(self.width, self.height, [(1, "f4")])
        self.multiply_color_pass = self._init_single_pass(
            self.width, self.height, [(4, "f4")]
        )
        self.multiply_world_pos_pass = self._init_single_pass(
            self.width, self.height, [(4, "f4")]
        )
        self.push_color_passes = self._init_push_passes()
        self.pull_color_passes = self._init_pull_passes()
        self.final_mask_pass = self._init_single_pass(
            self.width, self.height, [(4, "f4")]
        )
        self.jfa_init_pass = self._init_jfa_seed_pass()
        self.jfa_step_passes = [self._init_jfa_seed_pass(), self._init_jfa_seed_pass()]
        self.jfa_resolve_pass = self._init_jfa_resolve_pass()

    def render(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        confidences: np.ndarray,
        view_mat: np.ndarray,
        proj_mat: np.ndarray,
        fov_y: float,
    ) -> np.ndarray:
        """Renders a point cloud into a 2D image from a specific viewpoint.

        Args:
            points: Nx3 array of 3D point coordinates (float32).
            colors: Nx3 array of RGB colors in [0, 1] range (float32).
            confidences: Nx1 array of depth confidence values (float32).
            view_mat: 4x4 camera view matrix (world-to-camera).
            proj_mat: 4x4 camera projection matrix.
            fov_y: Vertical field of view in radians.

        Returns:
            H x W x 3 numpy array containing the rendered RGB image (uint8, 0-255).
        """
        if points.size == 0:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        confidences = np.asarray(confidences, dtype=np.float32).reshape(-1, 1)

        pos_buffer = self.ctx.buffer(points.tobytes())
        color_buffer = self.ctx.buffer(colors.tobytes())
        conf_buffer = self.ctx.buffer(confidences.tobytes())

        vao = self.ctx.vertex_array(
            self.points_program,
            [
                (pos_buffer, "3f", "a_position"),
                (color_buffer, "3f", "a_color"),
                (conf_buffer, "1f", "a_confidence"),
            ],
        )

        try:
            self._render_points_pass(vao, view_mat, proj_mat)
            self._render_downsample_passes()
            self._render_hpr_pass(fov_y)
            self._render_multiply_passes()
            self._render_jfa_passes()
            self._render_push_pull_passes()
            self._render_final_mask()
            return self._read_final_color()
        finally:
            vao.release()
            pos_buffer.release()
            color_buffer.release()
            conf_buffer.release()

    def _load_program(self, vert_path: Path, frag_path: Path) -> moderngl.Program:
        """Loads and compiles a ModernGL program from vertex and fragment shader files."""
        return self.ctx.program(
            vertex_shader=vert_path.read_text(encoding="utf-8"),
            fragment_shader=frag_path.read_text(encoding="utf-8"),
        )

    def _load_program_from_source(
        self, vertex_source: str, frag_path: Path
    ) -> moderngl.Program:
        """Loads and compiles a ModernGL program from a vertex source string and fragment file."""
        return self.ctx.program(
            vertex_shader=vertex_source,
            fragment_shader=frag_path.read_text(encoding="utf-8"),
        )

    def _init_points_pass(self) -> RenderPass:
        """Initializes the first pass which renders the raw point cloud data."""
        color = self._create_texture((self.width, self.height), 4, "f4")
        cam_pos = self._create_texture((self.width, self.height), 4, "f4")
        world_pos = self._create_texture((self.width, self.height), 4, "f4")
        depth = self.ctx.depth_texture((self.width, self.height))
        fbo = self.ctx.framebuffer([color, cam_pos, world_pos], depth_attachment=depth)
        return RenderPass(
            fbo=fbo,
            color_textures=[color, cam_pos, world_pos],
            depth_texture=depth,
            width=self.width,
            height=self.height,
        )

    def _init_downsample_passes(self) -> list[RenderPass]:
        """Initializes hierarchical downsampling passes for the push-pull algorithm."""
        passes: list[RenderPass] = []
        cur_width = self.width // 2
        cur_height = self.height // 2
        while cur_width > 0 and cur_height > 0:
            color = self._create_texture((cur_width, cur_height), 4, "f4")
            cam_pos = self._create_texture((cur_width, cur_height), 4, "f4")
            depth = self.ctx.depth_texture((cur_width, cur_height))
            fbo = self.ctx.framebuffer([color, cam_pos], depth_attachment=depth)
            passes.append(
                RenderPass(
                    fbo=fbo,
                    color_textures=[color, cam_pos],
                    depth_texture=depth,
                    width=cur_width,
                    height=cur_height,
                )
            )
            cur_width //= 2
            cur_height //= 2
        return passes

    def _init_single_pass(
        self, width: int, height: int, color_specs: list[tuple[int, str]]
    ) -> RenderPass:
        textures = [
            self._create_texture((width, height), components, dtype)
            for components, dtype in color_specs
        ]
        fbo = self.ctx.framebuffer(textures)
        return RenderPass(
            fbo=fbo,
            color_textures=textures,
            depth_texture=None,
            width=width,
            height=height,
        )

    def _init_push_passes(self) -> list[RenderPass]:
        """Initializes FBOs for the 'push' (downsampling) phase of hole filling."""
        return [
            self._init_single_pass(pass_.width, pass_.height, [(4, "f4")])
            for pass_ in self.downsample_passes
        ]

    def _init_pull_passes(self) -> list[RenderPass]:
        """Initializes FBOs for the 'pull' (upsampling/interpolation) phase of hole filling."""
        passes: list[RenderPass] = []
        for idx, _ in enumerate(self.downsample_passes):
            if idx == 0:
                width = self.width
                height = self.height
            else:
                width = self.downsample_passes[idx - 1].width
                height = self.downsample_passes[idx - 1].height
            passes.append(self._init_single_pass(width, height, [(4, "f4")]))
        return passes

    def _init_jfa_seed_pass(self) -> RenderPass:
        seed_pos = self._create_texture(
            (self.width, self.height), 2, "f4", filter_mode=moderngl.NEAREST
        )
        seed_color = self._create_texture(
            (self.width, self.height), 4, "u1", filter_mode=moderngl.NEAREST
        )
        fbo = self.ctx.framebuffer([seed_pos, seed_color])
        return RenderPass(
            fbo=fbo,
            color_textures=[seed_pos, seed_color],
            depth_texture=None,
            width=self.width,
            height=self.height,
        )

    def _init_jfa_resolve_pass(self) -> RenderPass:
        nearest_color = self._create_texture(
            (self.width, self.height), 4, "u1", filter_mode=moderngl.NEAREST
        )
        nearest_dist = self._create_texture(
            (self.width, self.height), 1, "f4", filter_mode=moderngl.NEAREST
        )
        fbo = self.ctx.framebuffer([nearest_color, nearest_dist])
        return RenderPass(
            fbo=fbo,
            color_textures=[nearest_color, nearest_dist],
            depth_texture=None,
            width=self.width,
            height=self.height,
        )

    def _create_texture(
        self,
        size: tuple[int, int],
        components: int,
        dtype: str,
        filter_mode: int | None = None,
    ) -> moderngl.Texture:
        texture = self.ctx.texture(size, components, dtype=dtype)
        filter_value = filter_mode if filter_mode is not None else moderngl.LINEAR
        texture.filter = (filter_value, filter_value)
        texture.repeat_x = False
        texture.repeat_y = False
        return texture

    def _quad_vao(self, program: moderngl.Program) -> moderngl.VertexArray:
        key = program.glo
        if key not in self._quad_vaos:
            # Build attribute list dynamically - some shaders may optimize out unused attributes
            content = [(self._quad_pos_buffer, "2f", "a_position")]
            if "a_tex_coord" in program:
                content.append((self._quad_tex_buffer, "2f", "a_tex_coord"))
            self._quad_vaos[key] = self.ctx.vertex_array(program, content)
        return self._quad_vaos[key]

    def _render_points_pass(
        self, vao: moderngl.VertexArray, view_mat: np.ndarray, proj_mat: np.ndarray
    ) -> None:
        """Executes the initial point rendering pass."""
        self.points_pass.fbo.use()
        self.ctx.viewport = (0, 0, self.points_pass.width, self.points_pass.height)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        self.points_pass.fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)

        proj = np.asarray(proj_mat, dtype=np.float32).T
        view = np.asarray(view_mat, dtype=np.float32).T
        self.points_program["u_proj_mat"].write(proj.tobytes())
        self.points_program["u_view_mat"].write(view.tobytes())
        self.points_program["u_confidence_threshold"].value = self.confidence_threshold

        vao.render(mode=moderngl.POINTS)

    def _render_downsample_passes(self) -> None:
        prev_color = self.points_pass.color_textures[0]
        prev_cam_pos = self.points_pass.color_textures[1]
        prev_depth = self.points_pass.depth_texture
        quad = self._quad_vao(self.downsample_program)

        for pass_ in self.downsample_passes:
            pass_.fbo.use()
            self.ctx.viewport = (0, 0, pass_.width, pass_.height)
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.depth_func = "<"
            pass_.fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)

            prev_color.use(0)
            prev_cam_pos.use(1)
            if prev_depth is not None:
                prev_depth.use(2)

            self.downsample_program["u_tex_color"].value = 0
            self.downsample_program["u_tex_cam_pos"].value = 1
            self.downsample_program["u_tex_depth"].value = 2

            quad.render(mode=moderngl.TRIANGLE_STRIP)

            prev_color = pass_.color_textures[0]
            prev_cam_pos = pass_.color_textures[1]
            prev_depth = pass_.depth_texture

    def _render_hpr_pass(self, fov_y: float) -> None:
        """Executes Hidden Point Removal to determine point visibility.

        HPR estimates which points are visible from the camera by checking if they
        lie on the 'convex hull' of the point cloud transformed by a spherical inversion.
        """
        self.hpr_pass.fbo.use()
        self.ctx.viewport = (0, 0, self.hpr_pass.width, self.hpr_pass.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.hpr_pass.fbo.clear(0.0, 0.0, 0.0, 0.0)

        cam_levels = [self.points_pass.color_textures[1]] + [
            pass_.color_textures[1] for pass_ in self.downsample_passes
        ]
        for idx in range(self.max_level):
            tex = cam_levels[min(idx, len(cam_levels) - 1)]
            tex.use(idx)

        self.hpr_program["u_tex_cam_pos_levels"].value = tuple(range(self.max_level))
        self.hpr_program["u_num_levels"].value = self.max_level
        self.hpr_program["u_coarse_level"].value = self.coarse_level
        self.hpr_program["u_viewport_size"].value = (
            float(self.width),
            float(self.height),
        )
        self.hpr_program["u_fov_y"].value = float(fov_y)
        self.hpr_program["u_s_hpr"].value = 10.0 * self.s0
        self.hpr_program["u_occlusion_threshold"].value = self.occlusion_threshold

        quad = self._quad_vao(self.hpr_program)
        quad.render(mode=moderngl.TRIANGLE_STRIP)

    def _render_multiply_passes(self) -> None:
        self.multiply_color_pass.fbo.use()
        self.ctx.viewport = (
            0,
            0,
            self.multiply_color_pass.width,
            self.multiply_color_pass.height,
        )
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.multiply_color_pass.fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.points_pass.color_textures[0].use(0)
        self.hpr_pass.color_textures[0].use(1)
        self.multiply_program["u_tex_source"].value = 0
        self.multiply_program["u_tex_factor"].value = 1

        quad = self._quad_vao(self.multiply_program)
        quad.render(mode=moderngl.TRIANGLE_STRIP)

        self.multiply_world_pos_pass.fbo.use()
        self.ctx.viewport = (
            0,
            0,
            self.multiply_world_pos_pass.width,
            self.multiply_world_pos_pass.height,
        )
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.multiply_world_pos_pass.fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.points_pass.color_textures[2].use(0)
        self.hpr_pass.color_textures[0].use(1)
        self.multiply_program["u_tex_source"].value = 0
        self.multiply_program["u_tex_factor"].value = 1

        quad = self._quad_vao(self.multiply_program)
        quad.render(mode=moderngl.TRIANGLE_STRIP)

    def _render_jfa_passes(self) -> None:
        """Executes Jump Flooding Algorithm to compute distance transform.

        JFA is an efficient GPU algorithm for computing Voronoi diagrams and distance
        transforms by iteratively 'jumping' and updating the nearest seed position.
        """
        self.jfa_init_pass.fbo.use()
        self.ctx.viewport = (0, 0, self.jfa_init_pass.width, self.jfa_init_pass.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.jfa_init_pass.fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.multiply_color_pass.color_textures[0].use(0)
        self.jfa_init_program["u_tex_source"].value = 0

        quad = self._quad_vao(self.jfa_init_program)
        quad.render(mode=moderngl.TRIANGLE_STRIP)

        max_dim = max(self.width, self.height)
        step = 1
        while step < max_dim:
            step <<= 1
        step >>= 1
        if step == 0:
            step = 1

        prev_seed_pos = self.jfa_init_pass.color_textures[0]
        prev_seed_color = self.jfa_init_pass.color_textures[1]
        pass_index = 0

        while True:
            pass_ = self.jfa_step_passes[pass_index]
            pass_.fbo.use()
            self.ctx.viewport = (0, 0, pass_.width, pass_.height)
            self.ctx.disable(moderngl.DEPTH_TEST)
            pass_.fbo.clear(0.0, 0.0, 0.0, 0.0)

            prev_seed_pos.use(0)
            prev_seed_color.use(1)
            self.jfa_step_program["u_tex_seed_pos"].value = 0
            self.jfa_step_program["u_tex_seed_color"].value = 1
            self.jfa_step_program["u_step"].value = int(step)
            self.jfa_step_program["u_viewport_size"].value = (self.width, self.height)

            quad = self._quad_vao(self.jfa_step_program)
            quad.render(mode=moderngl.TRIANGLE_STRIP)

            prev_seed_pos = pass_.color_textures[0]
            prev_seed_color = pass_.color_textures[1]

            if step == 1:
                break
            step >>= 1
            pass_index = 1 - pass_index

        self.jfa_resolve_pass.fbo.use()
        self.ctx.viewport = (
            0,
            0,
            self.jfa_resolve_pass.width,
            self.jfa_resolve_pass.height,
        )
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.jfa_resolve_pass.fbo.clear(0.0, 0.0, 0.0, 0.0)

        prev_seed_pos.use(0)
        prev_seed_color.use(1)
        self.jfa_resolve_program["u_tex_seed_pos"].value = 0
        self.jfa_resolve_program["u_tex_seed_color"].value = 1

        quad = self._quad_vao(self.jfa_resolve_program)
        quad.render(mode=moderngl.TRIANGLE_STRIP)

    def _render_push_pull_passes(self) -> None:
        quad = self._quad_vao(self.push_program)
        for idx, pass_ in enumerate(self.push_color_passes):
            pass_.fbo.use()
            self.ctx.viewport = (0, 0, pass_.width, pass_.height)
            self.ctx.disable(moderngl.DEPTH_TEST)
            pass_.fbo.clear(0.0, 0.0, 0.0, 0.0)

            if idx == 0:
                source_tex = self.multiply_color_pass.color_textures[0]
            else:
                source_tex = self.push_color_passes[idx - 1].color_textures[0]

            source_tex.use(0)
            self.push_program["u_tex_source"].value = 0
            quad.render(mode=moderngl.TRIANGLE_STRIP)

        for idx in range(len(self.pull_color_passes) - 1, -1, -1):
            pass_ = self.pull_color_passes[idx]
            pass_.fbo.use()
            self.ctx.viewport = (0, 0, pass_.width, pass_.height)
            self.ctx.disable(moderngl.DEPTH_TEST)
            pass_.fbo.clear(0.0, 0.0, 0.0, 0.0)

            if idx == len(self.pull_color_passes) - 1:
                source_prev = self.push_color_passes[idx].color_textures[0]
                source_cur = self.push_color_passes[idx - 1].color_textures[0]
            else:
                source_prev = self.pull_color_passes[idx + 1].color_textures[0]
                if idx == 0:
                    source_cur = self.multiply_color_pass.color_textures[0]
                else:
                    source_cur = self.push_color_passes[idx - 1].color_textures[0]

            source_prev.use(0)
            source_cur.use(1)
            self.pull_program["u_tex_source_prev"].value = 0
            self.pull_program["u_tex_source_cur"].value = 1

            quad = self._quad_vao(self.pull_program)
            quad.render(mode=moderngl.TRIANGLE_STRIP)

    def _render_final_mask(self) -> None:
        self.final_mask_pass.fbo.use()
        self.ctx.viewport = (
            0,
            0,
            self.final_mask_pass.width,
            self.final_mask_pass.height,
        )
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.final_mask_pass.fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.pull_color_passes[0].color_textures[0].use(0)
        self.jfa_resolve_pass.color_textures[1].use(1)
        self.final_mask_program["u_tex_source"].value = 0
        self.final_mask_program["u_tex_dist"].value = 1
        self.final_mask_program["u_sigma"].value = self.jfa_mask_sigma

        quad = self._quad_vao(self.final_mask_program)
        quad.render(mode=moderngl.TRIANGLE_STRIP)

    def _read_final_color(self) -> np.ndarray:
        texture = self.final_mask_pass.color_textures[0]
        data = texture.read()
        rgba = np.frombuffer(data, dtype=np.float32).reshape(
            (texture.height, texture.width, 4)
        )
        rgba = np.flipud(rgba)
        rgba = np.clip(rgba, 0.0, 1.0)
        rgb = rgba[..., :3] * rgba[..., 3:4]
        return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)

# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from da3.specs import Prediction

try:
    from da3.utils.export.gs import export_to_gs_ply, export_to_gs_video
except ImportError:
    export_to_gs_ply = None
    export_to_gs_video = None

try:
    from .colmap import export_to_colmap
except ImportError:
    export_to_colmap = None

try:
    from .depth_vis import export_to_depth_vis
except ImportError:
    export_to_depth_vis = None

try:
    from .feat_vis import export_to_feat_vis
except ImportError:
    export_to_feat_vis = None

try:
    from .glb import export_to_glb
except ImportError:
    export_to_glb = None

try:
    from .npz import export_to_mini_npz, export_to_npz
except ImportError:
    export_to_mini_npz = None
    export_to_npz = None


def export(
    prediction: Prediction,
    export_format: str,
    export_dir: str,
    **kwargs,
):
    if "-" in export_format:
        export_formats = export_format.split("-")
        for export_format in export_formats:
            export(prediction, export_format, export_dir, **kwargs)
        return

    if export_format == "glb":
        if export_to_glb is None:
            raise ImportError("glb export requires additional dependencies")
        export_to_glb(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "mini_npz":
        if export_to_mini_npz is None:
            raise ImportError("npz export requires additional dependencies")
        export_to_mini_npz(prediction, export_dir)
    elif export_format == "npz":
        if export_to_npz is None:
            raise ImportError("npz export requires additional dependencies")
        export_to_npz(prediction, export_dir)
    elif export_format == "feat_vis":
        if export_to_feat_vis is None:
            raise ImportError("feat_vis export requires additional dependencies")
        export_to_feat_vis(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "depth_vis":
        if export_to_depth_vis is None:
            raise ImportError("depth_vis export requires additional dependencies")
        export_to_depth_vis(prediction, export_dir)
    elif export_format == "gs_ply":
        if export_to_gs_ply is None:
            raise ImportError("gs_ply export requires additional dependencies (moviepy)")
        export_to_gs_ply(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "gs_video":
        if export_to_gs_video is None:
            raise ImportError("gs_video export requires additional dependencies (moviepy)")
        export_to_gs_video(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "colmap":
        if export_to_colmap is None:
            raise ImportError("colmap export requires additional dependencies (pycolmap)")
        export_to_colmap(prediction, export_dir, **kwargs.get(export_format, {}))
    else:
        raise ValueError(f"Unsupported export format: {export_format}")


__all__ = [
    export,
]

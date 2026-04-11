import torch
try:
    from colorspacious import cspace_convert
except ImportError:
    cspace_convert = None
from einops import rearrange
from jaxtyping import Float
from matplotlib import cm
from torch import Tensor


def apply_color_map(
    x: Float[Tensor, " *batch"],
    color_map: str = "inferno",
) -> Float[Tensor, "*batch 3"]:
    cmap = cm.get_cmap(color_map)

    # Convert to NumPy so that Matplotlib color maps can be used.
    mapped = cmap(x.detach().clip(min=0, max=1).cpu().numpy())[..., :3]

    # Convert back to the original format.
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)


def apply_color_map_to_image(
    image: Float[Tensor, "*batch height width"],
    color_map: str = "inferno",
) -> Float[Tensor, "*batch 3 height with"]:
    image = apply_color_map(image, color_map)
    return rearrange(image, "... h w c -> ... c h w")


def apply_color_map_2d(
    x: Float[Tensor, "*#batch"],
    y: Float[Tensor, "*#batch"],
) -> Float[Tensor, "*batch 3"]:
    if cspace_convert is None:
        # Fallback: RGB-space interpolation when colorspacious is unavailable.
        red = torch.tensor((189.0, 0.0, 0.0), device=x.device, dtype=torch.float32) / 255.0
        blue = torch.tensor((0.0, 45.0, 255.0), device=x.device, dtype=torch.float32) / 255.0
        white = torch.tensor((1.0, 1.0, 1.0), device=x.device, dtype=torch.float32)
        x_t = x.detach().clip(min=0, max=1)[..., None]
        y_t = y.detach().clip(min=0, max=1)[..., None]
        base = x_t * red + (1 - x_t) * blue
        rgb = y_t * base + (1 - y_t) * white
        return rgb.clip(min=0, max=1)

    red = cspace_convert((189, 0, 0), "sRGB255", "CIELab")
    blue = cspace_convert((0, 45, 255), "sRGB255", "CIELab")
    white = cspace_convert((255, 255, 255), "sRGB255", "CIELab")
    x_np = x.detach().clip(min=0, max=1).cpu().numpy()[..., None]
    y_np = y.detach().clip(min=0, max=1).cpu().numpy()[..., None]

    # Interpolate between red and blue on the x axis.
    interpolated = x_np * red + (1 - x_np) * blue

    # Interpolate between color and white on the y axis.
    interpolated = y_np * interpolated + (1 - y_np) * white

    # Convert to RGB.
    rgb = cspace_convert(interpolated, "CIELab", "sRGB1")
    return torch.tensor(rgb, device=x.device, dtype=torch.float32).clip(min=0, max=1)

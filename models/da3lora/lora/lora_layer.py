import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA-adapted Linear layer.

    Replaces nn.Linear with: output = W(x) + BA(x) * scaling
    where W is frozen (pretrained), B and A are trainable low-rank matrices.

    Args:
        original_linear: The original nn.Linear layer to adapt
        r: LoRA rank
        lora_alpha: LoRA scaling factor
        dropout: Dropout rate for LoRA path
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        self.weight = original_linear.weight
        self.bias = original_linear.bias

        self.weight.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.merged = False

    def merge(self):
        if not self.merged:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerge(self):
        if self.merged:
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return F.linear(x, self.weight, self.bias)

        result = F.linear(x, self.weight, self.bias)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result + lora_out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, lora_alpha={self.lora_alpha}, merged={self.merged}"
        )


class LoRAConv2d(nn.Module):
    """
    LoRA-adapted Conv2d layer.

    Replaces nn.Conv2d with: output = W(x) + BA(x) * scaling
    where W is frozen (pretrained), B and A are trainable low-rank convolutions.

    Args:
        original_conv: The original nn.Conv2d layer to adapt
        r: LoRA rank
        lora_alpha: LoRA scaling factor
        dropout: Dropout rate for LoRA path
    """

    def __init__(
        self,
        original_conv: nn.Conv2d,
        r: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        self.weight = original_conv.weight
        self.bias = original_conv.bias

        self.weight.requires_grad_(False)

        self.lora_A = nn.Parameter(
            torch.empty(r, self.in_channels, *self.kernel_size)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_channels, r, 1, 1)
        )

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.merged = False

    def merge(self):
        if not self.merged:
            self.weight.data += (
                torch.einsum("oirs,oi->ors", self.lora_B.squeeze(-1).squeeze(-1), self.lora_A.view(self.r, self.in_channels, -1).mean(dim=-1)).unsqueeze(-1).unsqueeze(-1)
            ) * self.scaling
            self.merged = True

    def unmerge(self):
        if self.merged:
            self.weight.data -= (
                torch.einsum("oirs,oi->ors", self.lora_B.squeeze(-1).squeeze(-1), self.lora_A.view(self.r, self.in_channels, -1).mean(dim=-1)).unsqueeze(-1).unsqueeze(-1)
            ) * self.scaling
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return F.conv2d(
                x, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups,
            )

        result = F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )

        x_drop = self.dropout(x)
        lora_a_out = F.conv2d(
            x_drop, self.lora_A, bias=None,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
        )
        lora_out = F.conv2d(lora_a_out, self.lora_B, bias=None) * self.scaling
        return result + lora_out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, r={self.r}, lora_alpha={self.lora_alpha}, "
            f"merged={self.merged}"
        )

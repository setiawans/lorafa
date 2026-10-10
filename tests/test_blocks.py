import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lorafa.losses import block_cosine_distance, cosine_distance, normalized_cosine_distance  # noqa: E402
from lorafa.models import attach_lora, build_model, count_params, replace_head  # noqa: E402
from lorafa.observe import observe, observed_blocks, observed_dim  # noqa: E402

torch.manual_seed(0)

def make():
    model = build_model("resnet20_4", 100)
    replace_head(model, 10)
    attach_lora(model, 16, 32.0, adapt_first_conv=False, seed=1)
    return model.eval()

def test_observed_blocks_cover_groups():
    model = make()
    expected = {"lorafa": 1, "lorafa_fc": 2, "lora": 3, "full": 2}
    for mode, n_groups in expected.items():
        blocks = observed_blocks(model, mode)
        assert len({g for g, _ in blocks}) == n_groups
        assert sum(n for _, n in blocks) == observed_dim(model, mode)
    assert sum(n for g, n in observed_blocks(model, "lorafa_fc") if g == "fc") == count_params(model.fc.parameters())

def test_block_cosine_matches_concat_for_single_block_and_is_scale_free():
    model = make()
    x, y = torch.randn(1, 3, 32, 32), torch.randint(0, 10, (1,))
    g = observe(model, x, y, "lorafa", "eval")
    h = observe(model, x + 0.1 * torch.randn_like(x), y, "lorafa", "eval")
    blocks = observed_blocks(model, "lorafa")
    assert torch.allclose(block_cosine_distance(g, h, blocks), normalized_cosine_distance(g, h))
    assert torch.allclose(normalized_cosine_distance(g, h), cosine_distance(g, h), atol=1e-6)
    assert torch.allclose(normalized_cosine_distance(1e-7 * g, g), torch.zeros(()), atol=1e-5)
    assert cosine_distance(1e-7 * g, 1e-7 * g) > 0.1
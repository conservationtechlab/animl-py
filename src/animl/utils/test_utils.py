"""
Utils for testing animl components, exposed for animl-r

"""

import json
from pathlib import Path
from typing import Union
import torch

from animl.model_architecture import MiewIdNet

MIEWID_HF_REPO = "conservationxlabs/miewid-msv3"


def fetch_and_convert_miewid(out_path: Union[str, Path]) -> bool:
    """
    Fetch MiewID weights from Hugging Face and convert them into a state_dict
    animl's own MiewIdNet can load, caching the result at `out_path`.

    conservationxlabs/miewid-msv3 has no declared license for its weights, so
    rather than redistributing a copy as our own release asset, this fetches
    them from the original source on demand -- which is what Conservation X
    Labs' own model card says to do.

    This intentionally avoids `transformers.AutoModel.from_pretrained(...,
    trust_remote_code=True)`, which would run Conservation X Labs' own
    modeling code. Instead it strict-loads the raw safetensors weights into
    animl's own MiewIdNet re-implementation -- a real compatibility check.
    Any key mismatch is treated as a hard failure (not a skip), since it
    means the two architectures have diverged and any test built on the
    result can't be trusted. Only a genuine inability to reach Hugging Face
    (missing optional deps, no network) results in a skip.

    Returns:
        True if weights were fetched/converted (or already cached) and are
        ready to use; False if this environment can't reach Hugging Face
        right now, in which case the caller should skip.
    """
    out_path = Path(out_path)
    if out_path.exists():
        return True

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError
        from safetensors.torch import load_file
    except ImportError:
        print(
            "huggingface_hub/safetensors not installed; skipping MiewID tests. "
            "Install with: pip install huggingface_hub safetensors"
        )
        return False

    try:
        config_path = hf_hub_download(MIEWID_HF_REPO, "config.json")
        weights_path = hf_hub_download(MIEWID_HF_REPO, "model.safetensors")
    except (HfHubHTTPError, OSError, ConnectionError) as e:
        print(f"Could not reach Hugging Face ({MIEWID_HF_REPO}); skipping MiewID tests. Error: {e}")
        return False

    with open(config_path) as f:
        config = json.load(f)
    state_dict = load_file(weights_path)


    KEY_REMAP = {"backbone.global_pool.p": "pooling.p"}
    for src_key, dst_key in KEY_REMAP.items():
        if src_key in state_dict and dst_key not in state_dict:
            tensor = state_dict.pop(src_key)
            if tuple(tensor.shape) != (1,):
                raise RuntimeError(
                    f"Refusing to remap '{src_key}' -> '{dst_key}': expected a shape (1,) GeM "
                    f"exponent but got {tuple(tensor.shape)}. The checkpoint's structure may have "
                    "changed in a way this narrow remap no longer accounts for."
                )
            print(f"Remapping checkpoint key '{src_key}' -> '{dst_key}' (shape {tuple(tensor.shape)})")
            state_dict[dst_key] = tensor

    # Infer n_classes from the checkpoint itself: config.json's value is
    # frequently a placeholder unrelated to the actual training run.
    final_weight_keys = [k for k in state_dict if k.startswith("final.") and k.endswith(".weight")]
    if not final_weight_keys:
        raise RuntimeError(
            "Could not find a 'final.*.weight' tensor in the Hugging Face checkpoint to infer "
            f"n_classes from. Top-level key prefixes found: {sorted({k.split('.')[0] for k in state_dict})}. "
            "The remote architecture may no longer match animl's MiewIdNet."
        )
    n_classes = state_dict[final_weight_keys[0]].shape[0]

    model = MiewIdNet(
        device="cpu",
        n_classes=n_classes,
        model_name=config.get("model_name", "efficientnetv2_rw_m"),
        use_fc=config.get("use_fc", False),
        fc_dim=config.get("fc_dim", 512),
        dropout=config.get("dropout", 0.0),
        loss_module=config.get("loss_module", "softmax"),
        pretrained=False,  # every weight is about to be overwritten
    )
    # The actual compatibility test. strict=False only so we can report
    # *which* keys mismatch; any mismatch at all is still a hard failure.
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "animl's MiewIdNet does not match the Hugging Face checkpoint key-for-key.\n"
            f"Missing keys ({len(result.missing_keys)}): {result.missing_keys}\n"
            f"Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys}\n"
            "Do not trust this conversion -- the implementations have diverged, or this "
            "checkpoint needs an explicit key-remapping step before it can be used."
        )

    # Sanity check: run the same call production code makes
    # (extract_miew_embeddings calls model.extract_feat(...) directly).
    model.eval()
    with torch.no_grad():
        emb = model.extract_feat(torch.randn(1, 3, 224, 224))
    assert emb.ndim == 2 and emb.shape[0] == 1, f"Unexpected embedding shape from extract_feat: {tuple(emb.shape)}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    return True
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    instanttensor_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_model_loader():
    model_dir = download_weights_from_hf(
        "openai-community/gpt2", cache_dir=None, allow_patterns=["*.safetensors"]
    )
    safetensors = glob.glob(f"{model_dir}/*.safetensors")
    assert len(safetensors) > 0

    instanttensor_tensors = {}
    hf_safetensors_tensors = {}

    for name, tensor in instanttensor_weights_iterator(safetensors, True):
        instanttensor_tensors[name] = tensor.to("cpu")

    for name, tensor in safetensors_weights_iterator(safetensors, True):
        hf_safetensors_tensors[name] = tensor

    assert len(instanttensor_tensors) == len(hf_safetensors_tensors)

    for name, instanttensor_tensor in instanttensor_tensors.items():
        assert instanttensor_tensor.dtype == hf_safetensors_tensors[name].dtype
        assert instanttensor_tensor.shape == hf_safetensors_tensors[name].shape
        assert torch.all(instanttensor_tensor.eq(hf_safetensors_tensors[name]))


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
@pytest.mark.parametrize("selected", [None, {"mtp.layers[0].weight"}, set()])
def test_instanttensor_selected_tensors(tmp_path, selected):
    weights = {
        name: torch.arange(1024, dtype=torch.float32) + index
        for index, name in enumerate(
            ("model.layers.0.weight", "mtp.layers[0].weight", "mtp.layers0.weight")
        )
    }
    path = tmp_path / "model.safetensors"
    save_file(weights, path)
    is_unused = None if selected is None else lambda name: name not in selected
    result = dict(
        instanttensor_weights_iterator([str(path)], False, is_unused_weight=is_unused)
    )
    assert result.keys() == (weights.keys() if selected is None else selected)
    for name, tensor in result.items():
        torch.testing.assert_close(tensor.cpu(), weights[name])


if __name__ == "__main__":
    test_instanttensor_model_loader()

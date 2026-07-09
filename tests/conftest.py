import pytest
import torch


@pytest.fixture(params=[
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="no cuda"
    )),
])
def device(request):
    return torch.device(request.param)

import numpy as np

from areal.api import ModelRequest
from areal.api.cli_args import GenerationHyperparameters
from areal.engine.vllm_remote import VLLMBackend


def test_vllm_backend_build_generation_request_with_return_routed_experts():
    backend = VLLMBackend()
    req = ModelRequest(
        input_ids=[1, 2, 3],
        gconfig=GenerationHyperparameters(max_new_tokens=4),
        metadata={"return_routed_experts": True},
    )

    http_req = backend.build_generation_request(req, with_lora=False, version=0)

    assert http_req.endpoint == "/v1/completions"
    assert http_req.payload["return_routed_experts"] is True


def test_vllm_backend_parse_generation_response_with_routed_experts():
    backend = VLLMBackend()
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "routed_experts": [[1, 5], [3, 7]],
                "logprobs": {
                    "tokens": ["token:42", "token:43"],
                    "token_logprobs": [-0.1, -0.2],
                },
            }
        ]
    }

    parsed = backend.parse_generation_response(response)

    assert parsed.output_tokens == [42, 43]
    assert parsed.routed_experts is not None
    np.testing.assert_array_equal(
        parsed.routed_experts, np.asarray([[1, 5], [3, 7]], dtype=np.int32)
    )


def test_vllm_backend_build_generation_request_without_return_routed_experts():
    backend = VLLMBackend()
    req = ModelRequest(
        input_ids=[1, 2, 3],
        gconfig=GenerationHyperparameters(max_new_tokens=4),
        metadata={},
    )

    http_req = backend.build_generation_request(req, with_lora=False, version=0)

    assert "return_routed_experts" not in http_req.payload

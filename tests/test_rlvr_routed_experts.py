import numpy as np
import pytest

from areal.api import ModelResponse
from areal.api.cli_args import GenerationHyperparameters
from areal.workflow.rlvr import RLVRWorkflow


class _DummyTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(
        self, data, tokenize, add_generation_prompt, enable_thinking
    ):
        assert tokenize and add_generation_prompt
        return [10, 11]

    def decode(self, token_ids):
        return "decoded"


class _DummyEngine:
    async def agenerate(self, req):
        return ModelResponse(
            input_tokens=req.input_ids,
            output_tokens=[20, 21],
            output_logprobs=[-0.1, -0.2],
            output_versions=[7, 7],
            routed_experts=np.asarray([[[1, 2]], [[3, 4]]], dtype=np.int32),
        )


def _reward_fn(*args, **kwargs):
    return 1.0


@pytest.mark.asyncio
async def test_rlvr_workflow_emits_full_length_routed_experts_tensor():
    workflow = RLVRWorkflow(
        reward_fn=_reward_fn,
        gconfig=GenerationHyperparameters(max_new_tokens=2),
        tokenizer=_DummyTokenizer(),
    )

    result = await workflow.arun_episode(
        _DummyEngine(),
        data={"messages": [{"role": "user", "content": "hello"}]},
    )

    # seq = prompt(2) + output(2) = 4
    assert "routed_experts" in result
    routed = result["routed_experts"].numpy()
    assert routed.shape == (1, 4, 1, 2)

    # Prompt positions are padded with -1; generated positions carry replay data.
    np.testing.assert_array_equal(routed[0, 0], np.asarray([[-1, -1]], dtype=np.int32))
    np.testing.assert_array_equal(routed[0, 1], np.asarray([[-1, -1]], dtype=np.int32))
    np.testing.assert_array_equal(routed[0, 2], np.asarray([[1, 2]], dtype=np.int32))
    np.testing.assert_array_equal(routed[0, 3], np.asarray([[3, 4]], dtype=np.int32))

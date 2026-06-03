"""
atb_python_qwen3vl_embedding — Pure ATB graph implementations for Qwen3VL-Embedding-2B.

Zero transformers dependency in the inference path.

Production modules:
    utils.py                   ATB parameter factories (Linear, RMSNorm, SA, RopeOp, etc.)
    preprocess.py              CPU-side image preprocessing (pure Python, no Processor)
    engine.py                  Qwen3VLEngine — full E2E inference engine (zero transformers)
    engine_utils.py            Weight loading, MRoPE, get_rope_index, vision embeddings
    env.py                     Central config: .env loader + project-wide variables
    text_attention.py          Qwen3VLTextAttention ATB graph
    text_mlp.py                Qwen3VLTextMLP ATB graph (SwiGLU)
    text_decoder_layer.py      Qwen3VLTextDecoderLayer ATB graph
    text_model.py              Qwen3VLTextModel runner (split-graph loop)
    vision_patch_embed.py      Qwen3VLVisionPatchEmbed ATB graph
    vision_attention.py        Qwen3VLVisionAttention ATB graph
    vision_mlp.py              Qwen3VLVisionMLP ATB graph
    vision_block.py            Qwen3VLVisionBlock ATB graph
    vision_model.py            Qwen3VLVisionModel runner (split-graph)

Test infrastructure:
    data_utils.py              Test data generation
    transformers_runner.py     Reference transformers implementations (test only)
    tests/test_engine.py       Engine E2E test (text/image/text+image)
    tests/test_*.py            Unit tests for individual ATB components
"""

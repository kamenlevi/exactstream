tc = dict(model_type="qwen3_5_moe_text", hidden_size=64, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
          head_dim=16, vocab_size=512, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
          linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4, num_experts=16,
          num_experts_per_tok=4, moe_intermediate_size=32, shared_expert_intermediate_size=32, norm_topk_prob=True,
          rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]},
          attn_output_gate=True, tie_word_embeddings=False)

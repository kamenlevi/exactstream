# Downloads (standing rule: >= 20 GB stays free; name, source, size logged)

| when (EET) | what | source | size | why |
|---|---|---|---|---|
| Sat 14 Nov 18:16 | weight index JSON of Qwen/Qwen3.5-2B | https://huggingface.co/Qwen/Qwen3.5-2B (model.safetensors.index.json) | 64 KB | find the native MTP head tensors |
| Sat 14 Nov 18:2x | the 15 `mtp.*` tensors of Qwen/Qwen3.5-2B (native MTP head, bf16), fetched by HTTP byte ranges from the single safetensors file; nothing else | https://huggingface.co/Qwen/Qwen3.5-2B (model.safetensors-00001-of-00001.safetensors), Apache-2.0 | 121.7 MB | the sandbox drafter starts from the native head, as b5 did on the 125B |

The target model itself (mlx-community/Qwen3.5-2B-MLX-4bit, ~1.6 GB) was already in ~/.cache/huggingface. Disk free after: 168 GB.

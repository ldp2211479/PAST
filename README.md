# PAST

This release contains code only. Datasets, model checkpoints, generated
artifacts, and experiment logs are not included.

## Repository Structure

```text
wise/
  graph/logcl/            Temporal graph model and dense graph-score export
  semantic/               Entity description, embedding, and semantic-code construction
  prompt/                 Prompt-data generation and semantic-code trie construction
  train/                  Code-token warmup, LoRA SFT, and LoRA merge
  infer/                  Trie-constrained prefix decoding
  core/                   Shared training and trie utilities
```

## Installation

Create an environment and install the common dependencies:

```bash
pip install -r requirements.txt
```

The graph component depends on DGL. Install a DGL build that matches your CUDA
version. The original graph-side dependency list is kept in:

```text
wise/graph/logcl/requirement.txt
```

## ICEWS14 Example

The commands below show a full ICEWS14 workflow. Run them from the repository
root unless a command explicitly changes directory.

Set reusable paths:

```bash
export DATASET=ICEWS14
export RAW_DIR=data/raw/${DATASET}
export PROMPT_DIR=data/processed/${DATASET}
export OUT_DIR=output/${DATASET}
export BASE_LLM=models/Qwen2.5-1.5B-Instruct
export EMBED_MODEL=models/Qwen3-Embedding-4B
```

If you use an API to generate descriptions, keep the key in the environment:

```bash
export OPENAI_API_KEY=your_api_key
```

### 1. Prepare Data

Place ICEWS14 files in the following format:

```text
data/raw/ICEWS14/
  entity2id.txt
  relation2id.txt
  train.txt
  valid.txt
  test.txt
```

`entity2id.txt` and `relation2id.txt` should contain tab-separated
`name<TAB>id` rows. Each temporal KG file should contain at least four integer
columns:

```text
subject_id relation_id object_id timestamp
```

### 2. Train LogCL and Export Graph Scores

Train the graph model:

```bash
cd wise/graph/logcl

python src/main.py \
  --dataset ${DATASET} \
  --data-root ../../../data/raw \
  --model-dir ../../../output/${DATASET}/logcl/checkpoints \
  --result-dir ../../../output/${DATASET}/logcl/results \
  --dense-graph-score-output-dir ../../../output/${DATASET}/graph_score \
  --n-epochs 500 \
  --gpu 0
```

Export dense graph scores. Set `LOGCL_CKPT` to the checkpoint selected from
training:

```bash
export LOGCL_CKPT=../../../output/${DATASET}/logcl/checkpoints/model.pth

for SPLIT in train valid test; do
  python src/main.py \
    --dataset ${DATASET} \
    --data-root ../../../data/raw \
    --model-dir ../../../output/${DATASET}/logcl/checkpoints \
    --result-dir ../../../output/${DATASET}/logcl/results \
    --model-state-file ${LOGCL_CKPT} \
    --test \
    --export-score-split ${SPLIT} \
    --dense-graph-score-output-dir ../../../output/${DATASET}/graph_score \
    --gpu 0
done

cd ../../..
```

Expected graph-score files include:

```text
output/ICEWS14/graph_score/test_graph_scores.npy
output/ICEWS14/graph_score/test_graph_queries.jsonl
output/ICEWS14/graph_score/test_graph_scores_summary.json
```

### 3. Build Semantic Codes

Generate entity and relation descriptions:

```bash
python -m wise.semantic.generate_descriptions \
  --dataset ${DATASET} \
  --dataset-dir ${RAW_DIR} \
  --output-dir ${OUT_DIR}/semantic_text \
  --api-key-env OPENAI_API_KEY \
  --api-base https://api.openai.com/v1/chat/completions \
  --model gpt-4o-mini \
  --target both
```

Encode entity descriptions:

```bash
python -m wise.semantic.build_embeddings \
  --dataset ${DATASET} \
  --entity-descriptions ${OUT_DIR}/semantic_text/entity_descriptions.csv \
  --model-path ${EMBED_MODEL} \
  --output-dir ${OUT_DIR}/semantic_code \
  --target entity
```

Run residual quantization to obtain semantic codes:

```bash
python -m wise.semantic.build_semantic_codes \
  --dataset ${DATASET} \
  --dataset-dir ${RAW_DIR} \
  --embeddings-path ${OUT_DIR}/semantic_code/entity_embeddings.npy \
  --output-dir ${OUT_DIR}/semantic_code \
  --num-levels 4 \
  --codebook-size 64
```

Main outputs:

```text
output/ICEWS14/semantic_code/entity_codes.tsv
output/ICEWS14/semantic_code/code_token_map.json
output/ICEWS14/semantic_code/code_token_embeddings.tsv
```

### 4. Build Prompt Data

```bash
python -m wise.prompt.build_prompt_data \
  --dataset_name ${DATASET} \
  --base_data_dir data/raw \
  --output_dir data/processed \
  --history_len 50
```

Main outputs:

```text
data/processed/ICEWS14/train.json
data/processed/ICEWS14/valid.json
data/processed/ICEWS14/test.json
data/processed/ICEWS14/test_no_his.json
```

### 5. Warm Up Semantic-Code Tokens

```bash
python -m wise.train.code_token_warmup \
  --base_model ${BASE_LLM} \
  --data_dir ${PROMPT_DIR} \
  --train_json ${PROMPT_DIR}/valid.json \
  --output_dir ${OUT_DIR}/model/code_token_warmup \
  --semantic_code_dir ${OUT_DIR}/semantic_code \
  --max_seq_len 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --learning_rate 5e-4 \
  --bf16
```

The warmed full model is saved under:

```text
output/ICEWS14/model/code_token_warmup/merged
```

### 6. Train Stage-1 LoRA SFT

```bash
python -m wise.train.stage1_sft \
  --base_model ${OUT_DIR}/model/code_token_warmup/merged \
  --data_dir ${PROMPT_DIR} \
  --train_json ${PROMPT_DIR}/train.json \
  --valid_json ${PROMPT_DIR}/valid.json \
  --output_dir ${OUT_DIR}/model/stage1_sft \
  --max_seq_len 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --bf16
```

The LoRA adapter is saved under:

```text
output/ICEWS14/model/stage1_sft/final_model
```

### 7. Merge LoRA Weights

```bash
python -m wise.train.merge_lora \
  --base_model ${OUT_DIR}/model/code_token_warmup/merged \
  --lora_model ${OUT_DIR}/model/stage1_sft/final_model \
  --output_dir ${OUT_DIR}/model/merged \
  --tokenizer_source base \
  --bf16
```

### 8. Build the Semantic-Code Trie

```bash
python -m wise.prompt.build_trie \
  --mode semantic_code \
  --dataset_name ${DATASET} \
  --base_data_dir data/raw \
  --output_dir data/processed \
  --tokenizer_path ${OUT_DIR}/model/merged \
  --semantic_code_dir ${OUT_DIR}/semantic_code \
  --trie_filename entity_code_trie.pkl
```

The trie is saved as:

```text
data/processed/ICEWS14/entity_code_trie.pkl
```

### 9. Run Inference

```bash
python -m wise.infer.prefix_decode \
  --model-path ${OUT_DIR}/model/merged \
  --tokenizer-path ${OUT_DIR}/model/merged \
  --prompts-jsonl ${PROMPT_DIR}/test.json \
  --graph-score-dir ${OUT_DIR}/graph_score \
  --entity-codes ${OUT_DIR}/semantic_code/entity_codes.tsv \
  --code-token-index-path ${OUT_DIR}/semantic_code/code_token_embeddings.tsv \
  --entity-trie-pkl ${PROMPT_DIR}/entity_code_trie.pkl \
  --entity2id ${RAW_DIR}/entity2id.txt \
  --output-dir ${OUT_DIR}/inference/test \
  --split test \
  --prompt-field input \
  --answer-field output \
  --num-code-tokens 4 \
  --num-beams 64 \
  --topk-per-step 128 \
  --topk-entities 50 \
  --alpha-graph-levels 0 0 0.5 0.5 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80
```

Main outputs:

```text
output/ICEWS14/inference/test/predictions_prefix_decode.json
output/ICEWS14/inference/test/metrics.json
output/ICEWS14/inference/test/metrics_by_mode.json
output/ICEWS14/inference/test/run_summary.json
```

## Citation

If you use this code, please cite the corresponding paper.

---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:7560
- loss:MultipleNegativesRankingLoss
base_model: Qwen/Qwen3-Embedding-0.6B
widget:
- source_sentence: Can the creature attempt another saving throw to end the effect?
  sentences:
  - Damage Immunities lightning, poison; bludgeoning, piercing, and slashing from
    nonmagical attacks that aren’t adamantine Condition Immunities charmed, exhaustion,
    frightened, paralyzed, petrified, poisoned Senses darkvision 60 ft., passive Perception
    10 Languages understands the languages of its creator but can’t speak Challenge
    5 (1,800 XP) Berserk. Whenever the golem starts its turn with 40 hit points or
    fewer, roll a d6. On a 6, the
  - Armor Class 21 (natural armor) Hit Points 385 (22d20 + 154) Speed 40 ft., fly
    80 ft., swim 40 ft.
  - creature can repeat the saving throw at the end of each of its turns, ending the
    effect on itself on a success. Elementals Air Elemental Large elemental, neutral
    Armor Class 15 Hit Points 90 (12d10 + 24) Speed 0 ft., fly 90 ft. (hover) STR
    DEX CON INT WIS CHA  14 (+2) 20 (+5) 14 (+2) 6 (−2) 10 (+0) 6 (−2)  Damage Resistances
    lightning, thunder; bludgeoning, piercing, and slashing from nonmagical attacks
    Damage Immunities poison
- source_sentence: I remember my past life as a powerful wizard, even with these new
    tiny gnome hands!
  sentences:
  - 77–96 Human  97–00 Tiefling  The reincarnated creature recalls its former life
    and experiences. It retains the capabilities it had in its original form, except
    it exchanges its original race for the new one and changes its racial traits accordingly.
  - Blade doesn’t give you additional attacks if you also have Extra Attack. Unarmored
    Defense If you already have the Unarmored Defense feature, you can’t gain it again
    from another class.
  - Ice Storm Locate Creature Polymorph Stone Shape Stoneskin Wall of Fire •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •  •
- source_sentence: Only creatures with hit points equal to or less than the remaining
    total are affected by the spell.
  sentences:
  - spell falls unconscious until the spell ends, the sleeper takes damage, or someone
    uses an action to shake or slap the sleeper awake. Subtract each creature’s hit
    points from the total before moving on to the creature with the next lowest hit
    points. A creature’s hit points must be equal to or less than the remaining total
    for that creature to be affected. Undead and creatures immune to being charmed
    aren’t affected by this spell.
  - 'Range: 60 feet Components: V , S Duration: Instantaneous A flood of healing energy
    flows from you into injured creatures around you. You restore up to 700 hit points,
    divided as you choose among any number of creatures that you can see within range.
    Creatures healed by this spell are also cured of all diseases and any effect making
    them  blinded or deafened. This spell has no effect on undead or constructs. Mass
    Healing Word 3rd-level evocation Classes: Cleric Casting Time: 1 bonus action'
  - 'Multiattack. The guardian makes two fist attacks. Fist. Melee Weapon Attack:
    +7 to hit, reach 5 ft., one target. Hit: 11 (2d6 + 4) bludgeoning damage. Reactions
    Shield. When a creature makes an attack against the wearer of the guardian’s amulet,
    the guardian grants a +2 bonus to the wearer’s AC if the guardian is within 5
    feet of the wearer. Specter Medium undead, chaotic evil Armor Class 12 Hit Points
    22 (5d8) Speed 0 ft., fly 50 ft. (hover)'
- source_sentence: Do spells cause madness in every campaign?
  sentences:
  - 'Multiattack. The giant makes two morningstar attacks. Morningstar. Melee Weapon
    Attack: +12 to hit, reach 10 ft., one target. Hit: 21 (3d8 + 8) piercing damage.
    Rock. Ranged Weapon Attack: +12 to hit, range 60/240 ft., one target. Hit: 30
    (4d10 + 8) bludgeoning damage. Fire Giant Huge giant, lawful evil Armor Class
    18 (plate) Hit Points 162 (13d12 + 78) Speed 30 ft. STR DEX CON INT WIS CHA  25
    (+7) 9 (−1) 23 (+6) 10 (+0) 14 (+2) 13 (+1)  Saving Throws Dex +3, Con +10, Cha
    +5'
  - In a typical campaign, characters aren’t driven mad by the horrors they face and
    the carnage they inflict day after day, but sometimes the stress of being an adventurer
    can be too much to bear. If your campaign has a strong horror theme, you might
    want to use madness as a way to reinforce that theme, emphasizing the extraordinarily
    horrific nature of the threats the adventurers face. Going Mad Various magical
    effects can inflict madness on an otherwise stable mind. Certain spells, such
  - thoughts, as well as any divination spell that it refuses. Wisdom (Insight) checks
    made to ascertain the sphinx’s intentions or sincerity have disadvantage.
- source_sentence: I pull the rope up after the party is inside!
  sentences:
  - 'Skills Perception +2, Stealth +4 Senses darkvision 120 ft., passive Perception
    12 Languages Elvish, Undercommon Challenge 1/4 (50 XP) Fey Ancestry. The drow
    has advantage on saving throws against being charmed, and magic can’t put the
    drow to sleep. Innate Spellcasting. The drow’s spellcasting ability is Charisma
    (spell save DC 11). It can innately cast the following spells, requiring no material
    components: At will: dancing lights 1/day each: darkness, faerie fire'
  - 'hold as many as eight Medium or smaller creatures. The rope can be pulled into
    the space, making the rope disappear from view outside the space. Attacks and
    spells can’t cross through the entrance into or out of the extradimensional space,
    but those inside can see out of it as if through a 3-foot-by-5-foot window centered
    on the rope. Anything inside the extradimensional space drops out when the spell
    ends. Sacred Flame Evocation cantrip Classes: Cleric Casting Time: 1 action Range:
    60 feet'
  - 'Duration: Concentration, up to 1 hour Until the spell ends, one willing creature
    you touch gains the ability to move up, down, and across vertical surfaces and
    upside down along ceilings, while leaving its hands free. The target also gains
    a climbing speed equal to its walking speed. Spike Growth 2nd-level transmutation
    Classes: Druid, Ranger Casting Time: 1 action Range: 150 feet Components: V ,
    S, M (seven sharp thorns or seven small twigs, each sharpened to a point)'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on Qwen/Qwen3-Embedding-0.6B
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: dnd validation
      type: dnd_validation
    metrics:
    - type: cosine_accuracy@1
      value: 0.5404761904761904
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.7595238095238095
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.8273809523809523
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.8785714285714286
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.5404761904761904
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.25317460317460316
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.16547619047619047
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.08785714285714284
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.5404761904761904
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.7595238095238095
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.8273809523809523
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.8785714285714286
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.7158989639345054
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.6629062736205594
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.666784266978288
      name: Cosine Map@100
---

# SentenceTransformer based on Qwen/Qwen3-Embedding-0.6B

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) <!-- at revision c54f2e6e80b2d7b7de06f51cec4959f6b3e03418 -->
- **Maximum Sequence Length:** 32768 tokens
- **Output Dimensionality:** 1024 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 32768, 'do_lower_case': False, 'architecture': 'Qwen3Model'})
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': True, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
queries = [
    "I pull the rope up after the party is inside!",
]
documents = [
    'hold as many as eight Medium or smaller creatures. The rope can be pulled into the space, making the rope disappear from view outside the space. Attacks and spells can’t cross through the entrance into or out of the extradimensional space, but those inside can see out of it as if through a 3-foot-by-5-foot window centered on the rope. Anything inside the extradimensional space drops out when the spell ends. Sacred Flame Evocation cantrip Classes: Cleric Casting Time: 1 action Range: 60 feet',
    'Skills Perception +2, Stealth +4 Senses darkvision 120 ft., passive Perception 12 Languages Elvish, Undercommon Challenge 1/4 (50 XP) Fey Ancestry. The drow has advantage on saving throws against being charmed, and magic can’t put the drow to sleep. Innate Spellcasting. The drow’s spellcasting ability is Charisma (spell save DC 11). It can innately cast the following spells, requiring no material components: At will: dancing lights 1/day each: darkness, faerie fire',
    'Duration: Concentration, up to 1 hour Until the spell ends, one willing creature you touch gains the ability to move up, down, and across vertical surfaces and upside down along ceilings, while leaving its hands free. The target also gains a climbing speed equal to its walking speed. Spike Growth 2nd-level transmutation Classes: Druid, Ranger Casting Time: 1 action Range: 150 feet Components: V , S, M (seven sharp thorns or seven small twigs, each sharpened to a point)',
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# [1, 1024] [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.4976, 0.1421, 0.1285]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `dnd_validation`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.5405     |
| cosine_accuracy@3   | 0.7595     |
| cosine_accuracy@5   | 0.8274     |
| cosine_accuracy@10  | 0.8786     |
| cosine_precision@1  | 0.5405     |
| cosine_precision@3  | 0.2532     |
| cosine_precision@5  | 0.1655     |
| cosine_precision@10 | 0.0879     |
| cosine_recall@1     | 0.5405     |
| cosine_recall@3     | 0.7595     |
| cosine_recall@5     | 0.8274     |
| cosine_recall@10    | 0.8786     |
| **cosine_ndcg@10**  | **0.7159** |
| cosine_mrr@10       | 0.6629     |
| cosine_map@100      | 0.6668     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 7,560 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              |
  | details | <ul><li>min: 7 tokens</li><li>mean: 16.25 tokens</li><li>max: 31 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 104.39 tokens</li><li>max: 385 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:----------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>The spell's energy erupts from the point of origin.</code>                                          | <code>A spell’s description specifies its area of effect, which typically has one of five different shapes: cone, cube, cylinder, line, or sphere. Every area of effect has a point of origin, a location from which the spell’s energy erupts. The rules for each shape specify how you position its point of origin. Typically, a point of origin is a point in space, but some spells have an area whose origin is a creature or an object. •  •</code>                                   |
  | <code>You can expend up to three charges at once when using the Pipes of the Sewers.</code>               | <code>Pipes of the Sewers Wondrous item, uncommon (requires attunement) You must be proficient with wind instruments to use these pipes. While you are attuned to the pipes, ordinary rats and giant rats are indifferent toward you and will not attack you unless you threaten or harm them. The pipes have 3 charges. If you play the pipes as an action, you can use a bonus action to expend 1 to 3 charges, calling forth one swarm of rats with each expended charge, provided</code> |
  | <code>If I'm 6th level and already used a 3rd level slot, how many 3rd level slots do I have left?</code> | <code>2nd 3 — — — — — — — —  3rd 4 2 — — — — — — —  4th 4 3 — — — — — — —  5th 4 3 2 — — — — — —  6th 4 3 3 — — — — — —  7th 4 3 3 1 — — — — —  8th 4 3 3 2 — — — — —  •  •  •  •</code>                                                                                                                                                                                                                                                                                                     |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 20
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 20
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch   | Step  | Training Loss | dnd_validation_cosine_ndcg@10 |
|:-------:|:-----:|:-------------:|:-----------------------------:|
| 0.5291  | 500   | 0.1829        | -                             |
| 1.0     | 945   | -             | 0.6305                        |
| 1.0582  | 1000  | 0.1245        | -                             |
| 1.5873  | 1500  | 0.0597        | -                             |
| 2.0     | 1890  | -             | 0.6135                        |
| 2.1164  | 2000  | 0.0518        | -                             |
| 2.6455  | 2500  | 0.0302        | -                             |
| 3.0     | 2835  | -             | 0.6961                        |
| 3.1746  | 3000  | 0.0342        | -                             |
| 3.7037  | 3500  | 0.0203        | -                             |
| 4.0     | 3780  | -             | 0.6721                        |
| 4.2328  | 4000  | 0.0207        | -                             |
| 4.7619  | 4500  | 0.0147        | -                             |
| 5.0     | 4725  | -             | 0.6792                        |
| 5.2910  | 5000  | 0.0181        | -                             |
| 5.8201  | 5500  | 0.0207        | -                             |
| 6.0     | 5670  | -             | 0.5495                        |
| 6.3492  | 6000  | 0.0197        | -                             |
| 6.8783  | 6500  | 0.0114        | -                             |
| 7.0     | 6615  | -             | 0.6696                        |
| 7.4074  | 7000  | 0.0107        | -                             |
| 7.9365  | 7500  | 0.0085        | -                             |
| 8.0     | 7560  | -             | 0.6582                        |
| 8.4656  | 8000  | 0.0112        | -                             |
| 8.9947  | 8500  | 0.0102        | -                             |
| 9.0     | 8505  | -             | 0.6006                        |
| 9.5238  | 9000  | 0.0086        | -                             |
| 10.0    | 9450  | -             | 0.6782                        |
| 10.0529 | 9500  | 0.0134        | -                             |
| 10.5820 | 10000 | 0.0073        | -                             |
| 11.0    | 10395 | -             | 0.6829                        |
| 11.1111 | 10500 | 0.0065        | -                             |
| 11.6402 | 11000 | 0.0107        | -                             |
| 12.0    | 11340 | -             | 0.7159                        |


### Framework Versions
- Python: 3.10.19
- Sentence Transformers: 5.2.0
- Transformers: 4.57.6
- PyTorch: 2.7.1+cu118
- Accelerate: 1.12.0
- Datasets: 4.5.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
"""A-posteriori unimodality (AU) of annotator subgroups, and the language models used to audit it."""
from .agreement import cohen_kappa, spearman, weighted_kappa
from .annotation import (AnnotationError, annotate, collect_runs, ensemble, inspect_run, load_model_inputs, progress,
                         sample_inputs)
from .datasets import SPECS, get_spec
from .eda import eda_figures, eda_tables, save_eda
from .eda_dataset import dataset_tables, eda_overview_all, save_dataset_eda
from .eda_total import (read_group_tables, ratings_per_text, save_appendix_figures, save_toxicity_by_dimension, save_total_eda,
                        toxicity_by_group)
from .ablation_prompt import save_ablation_prompt
from .attribution_bootstrap import attribution_gap, attribution_table, bootstrap_gap
from .au_total import save_total_au
from .dfu_total import save_total_dfu
from .env import Env
from .export import csv_ready, sample_per_dimension, save_au_tables
from .models import DEFAULT_CONFIGS, Model, ModelConfig, ModelError, ModelHub
from .persona_annotation import (annotate_personas, collect_persona_runs, group_order, inspect_personas, load_persona_inputs,
                                 persona_long)
from .pipeline import DatasetResult, clean, load_results, make_tool, run_dataset
from .prompts import PersonaPrompt, ScoringPrompt, label_from_score, save_prompts
from .sampling import cover_per_dimension, minimal_cover
from .scoring import score_personas, score_texts
from .spec import DatasetSpec
from .unimodality import PrunedAU, Unimodality, dfu_of_hist
from .utils import find_root, read_au_csv

__version__ = "0.4.0"
__all__ = ["Unimodality", "PrunedAU", "dfu_of_hist",
           "DatasetSpec", "SPECS", "get_spec", "run_dataset", "DatasetResult", "clean", "make_tool",
           "save_au_tables", "csv_ready", "sample_per_dimension", "minimal_cover", "cover_per_dimension", "find_root", "read_au_csv",
           "Env", "Model", "ModelConfig", "ModelHub", "ModelError", "DEFAULT_CONFIGS",
           "ScoringPrompt", "PersonaPrompt", "label_from_score", "score_texts", "score_personas",
           "annotate", "ensemble", "load_model_inputs", "progress", "inspect_run", "collect_runs", "sample_inputs",
           "AnnotationError", "cohen_kappa", "weighted_kappa", "spearman",
           "eda_tables", "eda_figures", "save_eda", "dataset_tables", "save_dataset_eda", "eda_overview_all", "ratings_per_text", "save_total_eda",
           "read_group_tables", "toxicity_by_group", "save_toxicity_by_dimension", "save_appendix_figures", "save_total_dfu", "save_total_au", "save_ablation_prompt", "attribution_gap", "attribution_table", "bootstrap_gap", "load_results", "save_prompts",
           "annotate_personas", "load_persona_inputs", "persona_long", "collect_persona_runs", "inspect_personas", "group_order"]

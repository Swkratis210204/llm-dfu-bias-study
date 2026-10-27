"""A-posteriori unimodality (AU) of annotator subgroups, and the language models used to audit it."""
from .agreement import cohen_kappa, spearman, weighted_kappa
from .annotation import (AnnotationError, annotate, collect_runs, ensemble, inspect_run, load_model_inputs, progress,
                         sample_inputs)
from .datasets import SPECS, get_spec
from .env import Env
from .export import csv_ready, sample_per_dimension, save_au_tables
from .models import DEFAULT_CONFIGS, Model, ModelConfig, ModelError, ModelHub
from .pipeline import DatasetResult, clean, make_tool, run_dataset
from .prompts import PersonaPrompt, ScoringPrompt, label_from_score
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
           "AnnotationError", "cohen_kappa", "weighted_kappa", "spearman"]

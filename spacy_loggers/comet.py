"""
A logger that logs training activity to Comet.
"""

from typing import Dict, Any, Tuple, Callable, List, Optional, IO
import sys

from spacy import util
from spacy import Language
from spacy.training.loggers import console_logger


# entry point: spacy.CometLogger.v1
def comet_logger_v1(
    experiment_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    remove_config_values: List[str] = [],
):
    try:
        import comet_ml
    except ImportError:
        raise ImportError(
            "The 'comet_ml' library could not be found - did you install it? "
            "Alternatively, specify the 'ConsoleLogger' in the "
            "'training.logger' config section, instead of the 'CometLogger'."
        )

    console = console_logger(progress_bar=False)

    def setup_logger(
        nlp: "Language", stdout: IO = sys.stdout, stderr: IO = sys.stderr
    ) -> Tuple[Callable[[Dict[str, Any]], None], Callable[[], None]]:
        config = nlp.config.interpolate()
        config_dot = util.dict_to_dot(config)
        for field in remove_config_values:
            del config_dot[field]
        config = util.dot_to_dict(config_dot)

        experiment = comet_ml.get_global_experiment()

        if experiment is None:
            experiment = comet_ml.Experiment()
            if experiment_name:
                experiment.set_name(experiment_name)

        experiment.add_tags(tags=tags)
        experiment.log_parameters(config)

        console_log_step, console_finalize = console(nlp, stdout, stderr)

        def log_step(info: Optional[Dict[str, Any]]):
            console_log_step(info)
            if info is not None:
                score = info["score"]
                other_scores = info["other_scores"]
                losses = info["losses"]
                output_path = info.get("output_path", None)
                if score is not None:
                    experiment.log_metric("score", score)
                if losses:
                    experiment.log_metrics({f"loss_{k}": v for k, v in losses.items()})
                if isinstance(other_scores, dict):
                    experiment.log_metrics(
                        {
                            k: v
                            for k, v in util.dict_to_dot(other_scores).items()
                            if isinstance(v, float) or isinstance(v, int)
                        }
                    )

                output_path = info.get("output_path", None)
                if output_path and score == max(info["checkpoints"])[0]:
                    experiment.log_model(
                        name="best_model",
                        file_or_folder=output_path
                    )

        def finalize() -> None:
            console_finalize()
            experiment.end()

        return log_step, finalize

    return setup_logger

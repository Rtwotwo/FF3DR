import argparse
import logging

from running.inference.run_matrixcity_inference import (
	FF3DR as _MatrixCityFF3DR,
	_REPO_ROOT,
	_build_parser as _matrixcity_build_parser,
	_load_run_arg_defaults as _matrixcity_load_run_arg_defaults,
)

logger = logging.getLogger(__name__)


class FF3DR(_MatrixCityFF3DR):
	"""UrbanScene inference entrypoint built on top of the MatrixCity pipeline."""

	def run(self):
		logger.info("[INFO] Running FF3DR on UrbanScene scene: %s", self.area_path)
		return super().run()


def _build_parser(run_defaults):
	parser = _matrixcity_build_parser(run_defaults)
	parser.set_defaults(run_args_yaml=str(_REPO_ROOT / "configs" / "run_urbanscene_inference.yaml"))
	parser.description = "FF3DR UrbanScene inference"
	return parser


if __name__ == "__main__":
	pre_parser = argparse.ArgumentParser(add_help=False)
	pre_parser.add_argument(
		"--run_args_yaml",
		type=str,
		default=str(_REPO_ROOT / "configs" / "run_urbanscene_inference.yaml"),
	)
	pre_args, _ = pre_parser.parse_known_args()
	defaults = _matrixcity_load_run_arg_defaults(pre_args.run_args_yaml)

	parser = _build_parser(defaults)
	args = parser.parse_args()
	runner = FF3DR(args)
	runner.run()

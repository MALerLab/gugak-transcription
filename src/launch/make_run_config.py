"""Build one run's full config from the recipes: experiments/<exp>/configs/<name>.yaml.

Merges the module recipes a command ``uses`` (configs/<package>/<module>.yaml, nested as
``<package>.<module>``), applies the command's ``overrides``, adds its ``run`` section, and
resolves every ``${...}``. Tweak the written file if needed; the command reads only that file.

Usage: python -m src.launch.make_run_config <command> <experiment_folder_name> [--name <name>] [--set key=value ...]
  e.g. python -m src.launch.make_run_config align_songs exp003_261029_pesto --set features.f0.f0_tracker=yin
"""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

CONFIGS_DIR = Path("configs")
EXPERIMENTS_DIR = Path("experiments")
STORAGE_EXPERIMENTS_DIR = Path("storage/experiments")


def build_run_config(command: str, experiment: str, settings: list[str]):
    """The resolved config of one command run in one experiment."""
    recipe = OmegaConf.load(CONFIGS_DIR / "pipelines" / f"{command}.yaml")
    config = OmegaConf.create()
    for module_path in recipe.uses:
        package, module = module_path.split("/")
        module_config = OmegaConf.load(CONFIGS_DIR / f"{module_path}.yaml")
        module_config = OmegaConf.merge(module_config, recipe.overrides.get(module_path, {}))
        OmegaConf.update(config, f"{package}.{module}", module_config)
    config.run = recipe.run
    config.run.artifacts_dir = str(STORAGE_EXPERIMENTS_DIR / experiment)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(settings))
    OmegaConf.resolve(config)  # write real paths, not ${...}
    return config


def main(command: str, experiment: str, name: str | None, settings: list[str]) -> Path:
    config = build_run_config(command, experiment, settings)
    out_path = EXPERIMENTS_DIR / experiment / "configs" / f"{name or command}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; it may be what an earlier run read")
    out_path.write_text(f"# {command} run config, built from configs/ by src/launch/make_run_config.py\n"
                        + OmegaConf.to_yaml(config))
    print(out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="a module in src/pipelines/")
    parser.add_argument("experiment", help="experiment folder name, exp<NNN>_<YYMMDD>_<rest>")
    parser.add_argument("--name", help="config file name if one command runs several times (default: command)")
    parser.add_argument("--set", nargs="*", default=[], dest="settings", help="dotted key=value changes")
    arguments = parser.parse_args()
    main(arguments.command, arguments.experiment, arguments.name, arguments.settings)

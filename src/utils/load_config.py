import pathlib
import hydra

def load_config(directory_or_checkpoint_path: str | pathlib.Path):
    path = pathlib.Path(directory_or_checkpoint_path)
    config_dir = path.parent if path.is_file() else path

    with hydra.initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
        return hydra.compose(config_name="config")

import os
from typing import Any, Callable

from util import normalize_path, ensure_directory

class ArtifactManager:
    """Manages training artifacts."""

    def __init__(self, config: dict[str, Any], seed: int):
        """Initialize the artifact manager from config and seed.

        Args:
            config: Configuration.
            seed: Random seed used for the training run.

        Raises:
            ValueError: If required config fields are missing.
        """

        self.seed = seed
        self.run_name = config.get("run_name", None)
        if self.run_name is None:
            raise ValueError("Config must contain a 'run_name' field.")
        self.output_path = self._setup_output_path(config)

        # Track registered callback artifacts.
        self.registered_artifacts = []

    def _setup_output_path(self, config: dict[str, Any]) -> str:
        output_path = config.get("output_path", None)
        if output_path is None:
            raise ValueError("Config must contain an 'output_path' field.")

        output_path = normalize_path(output_path.replace("{run_name}", self.run_name))
        return output_path

    def register_artifact(
        self,
        artifact_identifier: str,
        artifact_description: str,
        file_name_generator: Callable[..., str],
        check_conflicts: bool = True
    ) -> Callable:
        """Register a callback artifact and return a path generator function.
        This method registers an artifact (e.g., CSV files, model files) and
        returns a callable that generates full file paths.

        Args:
            artifact_identifier: Identifier for the artifact (e.g., 'harvest_probabilities', 'models'). Used as part of the artifact's path.
            artifact_description: Artifact description.
            file_name_generator: Function that takes arbitrary arguments and
            returns a filename. For directory-only paths (e.g., model saving),
            return an empty string.
            check_conflicts: Whether to check for existing files in the
            artifact directory. Defaults to True. If True and files exist,
            raises FileExistsError. Set to False for custom conflict checking.
        Raises:
            FileExistsError: If check_conflicts=True and the artifact directory
            already contains files.
        Returns:
            A callable with signature (*args, ensure_dir=True) -> str that:
            - Takes arguments to pass to file_name_generator.
            - Takes optional ensure_dir parameter to control directory creation.
            - Returns the full path to the artifact file (or directory if filename is empty).
        """

        artifact_dir = os.path.join(self.output_path, str(self.seed), artifact_identifier)

        # Info for artifact registration.
        artifact_info = {
            "artifact_identifier": artifact_identifier,
            "artifact_description": artifact_description,
            "artifact_dir": artifact_dir,
            "file_name_generator": file_name_generator
        }

        # Check for conflicts if requested.
        if check_conflicts and os.path.exists(artifact_dir):
            existing_files = os.listdir(artifact_dir)
            if len(existing_files) > 0:
                raise FileExistsError(
                    f"\nError: Artifacts already exist for artifact '{artifact_identifier}' with description '{artifact_description}' at '{artifact_dir}'\n"
                    f"Found {len(existing_files)} existing file(s).\n"
                    "To prevent overwriting existing results, please either:\n"
                    "  - Use a different seed (--seed)\n"
                    "  - Use a different run_name in the config\n"
                    "  - Remove or backup the existing artifacts"
                )

        self.registered_artifacts.append(artifact_info)

        # Return a closure that captures artifact_dir and file_name_generator.
        def get_path(*args, ensure_dir: bool = True) -> str:
            filename = file_name_generator(*args)
            if ensure_dir:
                ensure_directory(artifact_dir)
            return os.path.join(artifact_dir, filename)

        return get_path

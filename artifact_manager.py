import os
from typing import Optional, Dict, Any

from util import normalize_path, ensure_directory

class ArtifactManager:
    """Manages training artifacts."""

    def __init__(self, config: Dict[str, Any], seed: int):
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
        self.model_filepath = os.path.join(self.output_path, str(seed), "models")
        self.actor_file = os.path.join(self.model_filepath, "actor.keras")
        self.critic_file = os.path.join(self.model_filepath, "critic.keras")

    def _setup_output_path(self, config: Dict[str, Any]) -> str:
        output_path = config.get("output_path", None)
        if output_path is None:
            raise ValueError("Config must contain an 'output_path' field.")

        output_path = normalize_path(output_path.replace("{run_name}", self.run_name))
        return output_path

    def check_artifacts_exist(self, agent) -> bool:
        """Check if any model artifacts already exist.

        Args:
            agent: The trained agent.

        Returns:
            True if any artifacts exist, False otherwise.
        """

        artifacts_exist = True
        try:
            for file in agent.get_save_file_names().values():
                file_path = os.path.join(self.model_filepath, file)
                if not os.path.exists(file_path):
                    artifacts_exist = False
                    break
        except NotImplementedError:
            pass

        return artifacts_exist

    def ensure_no_conflicts(self, agent) -> None:
        """Check for existing artifacts and raise an error if found.

        Args:
            agent: The trained agent.

        Raises:
            FileExistsError: If any model artifacts already exist.
        """

        if self.check_artifacts_exist(agent):
            error_msg = (
                f"\nError: Saved model artifacts already exist at '{self.model_filepath}'\n"
                "To prevent overwriting existing results, training is terminated.\n"
                "Please either:\n"
                "  - Use a different seed (--seed).\n"
                "  - Use a different run_name in the config.\n"
                "  - Remove or backup the existing artifacts."
            )
            raise FileExistsError(error_msg)

    def save_training_artifacts(self, agent) -> None:
        """Save the training artifacts.

        Args:
            agent: The trained agent.
        """

        ensure_directory(self.model_filepath)
        try:
            agent.save(self.model_filepath)
        except NotImplementedError:
            pass

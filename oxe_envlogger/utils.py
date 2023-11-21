from typing import Any, Dict, Optional, Tuple
from oxe_envlogger.envlogger import MetadataInfo


class MetadataLogger():
    """
    Helper class to log metadata for each step and episode.
    """

    def __init__(self,
                 step_metadata_info: Optional[MetadataInfo] = None,
                 episode_metadata_info: Optional[MetadataInfo] = None,):
        """
        args:
            step_metadata_info: dict of metadata info
            episode_metadata_info: dict of metadata info
        """
        self.step_metadata_info = step_metadata_info
        self.episode_metadata_info = episode_metadata_info
        self.step_metadata_elements = {}
        self.episode_metadata_elements = {}

    def log_step(self, key: str, value: Any):
        """Provide a key-value pair to be logged for each step."""
        assert self.step_metadata_info is not None, "step_metadata_info is None"
        assert key in self.step_metadata_info, f"{key} not in step_metadata_info"
        self.step_metadata_elements[key] = value

    def log_episode(self, key: str, value: Any):
        """Provide a key-value pair to be logged for each episode."""
        assert self.episode_metadata_info is not None, "episode_metadata_info is None"
        assert key in self.episode_metadata_info, f"{key} not in episode_metadata_info"
        self.episode_metadata_elements[key] = value

    def step_ref(self):
        """Provide a reference to the step_metadata in OXEEnvLogger()."""
        return self.step_metadata_info, self._get_step_metadata

    def episode_ref(self):
        """Provide a reference to the episode_metadata in OXEEnvLogger()."""
        return self.episode_metadata_info, self._get_episode_metadata

    def _get_step_metadata(self, *args, **kwargs) -> Dict[str, Any]:
        return self.step_metadata_elements

    def _get_episode_metadata(self, *args, **kwargs) -> Dict[str, Any]:
        return self.episode_metadata_elements

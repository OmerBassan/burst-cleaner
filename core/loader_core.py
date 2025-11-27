from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ImageLoaderInterface(ABC):
    """
    Abstract interface for platform-specific image loading / metadata access.
    Windows, Linux, Android will each implement a subclass.
    """

    @abstractmethod
    def scan_folder(self, folder_path: str) -> List[str]:
        """Return list of image paths/identifiers from folder."""
        pass

    @abstractmethod
    def extract_timestamp(self, image_identifier: str) -> float:
        """Return image timestamp in epoch seconds."""
        pass

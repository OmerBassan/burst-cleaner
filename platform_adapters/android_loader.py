from typing import List
from burst_cleaner.core.loader_core import ImageLoaderInterface

class AndroidImageLoader(ImageLoaderInterface):
    """
    This loader receives Android URIs instead of paths.
    The app layer (Kotlin/Java) passes the URIs into Python via bridge.
    """

    def scan_folder(self, folder_token: str) -> List[str]:
        """
        folder_token – a logical identifier provided by the Android app.
        The actual scanning is done at the Android side.
        """
        raise NotImplementedError("Must be provided by Android integration")

    def extract_timestamp(self, image_uri: str) -> float:
        """
        Android must provide timestamp directly via a bridge call.
        """
        raise NotImplementedError("Platform method required")

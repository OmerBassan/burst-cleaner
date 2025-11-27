import os
import exifread
import time
from typing import List
from burst_cleaner.core.loader_core import ImageLoaderInterface

class WindowsImageLoader(ImageLoaderInterface):

    def scan_folder(self, folder_path: str) -> List[str]:
        valid_ext = {".jpg", ".jpeg", ".png", ".heic"}
        files = []
        for f in os.listdir(folder_path):
            ext = os.path.splitext(f.lower())[1]
            if ext in valid_ext:
                files.append(os.path.join(folder_path, f))
        return sorted(files)

    def extract_timestamp(self, image_path: str) -> float:
        try:
            with open(image_path, "rb") as f:
                tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal")
            if "EXIF DateTimeOriginal" in tags:
                dt = str(tags["EXIF DateTimeOriginal"])
                # dt = "2024:01:02 12:03:04"
                struct = time.strptime(dt, "%Y:%m:%d %H:%M:%S")
                return time.mktime(struct)
        except:
            pass

        # fallback - file modified time
        return os.path.getmtime(image_path)

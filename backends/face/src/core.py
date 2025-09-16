# face/core.py

from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable, Any
from tqdm import tqdm
from . import config # Use our config

def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames_func: Callable[[str, List[str], Any], None], progress: Any = None) -> None:
    """Processes frames in parallel using a thread pool."""
    with ThreadPoolExecutor(max_workers=config.execution_threads) as executor:
        futures = [executor.submit(process_frames_func, source_path, [path], progress) for path in temp_frame_paths]
        for future in futures:
            future.result()

def process_video(source_path: str, frame_paths: list[str], process_frames_func: Callable[[str, List[str], Any], None]) -> None:
    """Sets up a progress bar and processes video frames."""
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    with tqdm(total=len(frame_paths), desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        progress.set_postfix_str(f"providers={config.execution_providers}, threads={config.execution_threads}")
        multi_process_frame(source_path, frame_paths, process_frames_func, progress)
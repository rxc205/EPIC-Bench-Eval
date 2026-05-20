import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import json
import os
import time
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
import requests as url_requests
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
import portalocker

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from dotenv import find_dotenv, load_dotenv
from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI
from PIL import Image

load_dotenv(verbose=True)


@register_model("openai_compatible")
class OpenAICompatible(lmms):
    def __init__(
        self,
        model_version: str = "grok-2-latest",
        timeout: int = 300,
        max_retries: int = 5,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        azure_openai: bool = False,
        fps: int = 2,
        max_num_frames: int = 32,
        temperature: float = None,
        top_p: float = None,
        max_new_tokens: int = None,
        thinking_budget: int = 2048,
        modality="video",
        max_workers=1,
        **kwargs,
    ) -> None:
        print(locals())
        super().__init__()
        self.model_version = model_version
        self.timeout = timeout
        self.max_retries = max_retries
        self.continual_mode = continual_mode
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.thinking_budget = thinking_budget
        self.modality = modality
        self.max_workers = max_workers

        if self.continual_mode:
            if response_persistent_folder is None:
                raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")
            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

            if os.path.exists(self.response_persistent_file):
                with portalocker.Lock(self.response_persistent_file, 'r', timeout=50, flags=portalocker.LOCK_SH) as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        self.client = (
            OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"), timeout=self.timeout)
            if not azure_openai
            else AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"), api_version=os.getenv("AZURE_OPENAI_API_VERSION"))
        )

    def encode_video(self, video_path, max_num_frames):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num, video_fps = len(vr), vr.get_avg_fps()
        for_get_frames_num = min(
            max_num_frames, int(total_frame_num / video_fps * self.fps), total_frame_num
        )
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_uuid(self, task, split, doc_id, gen_kwargs):
        temperature = gen_kwargs["temperature"]
        top_p = gen_kwargs["top_p"]
        max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
        api_uuid = (
            f"fps-{self.fps}###"
            f"max_num_frames-{self.max_num_frames}###"
            f"temperature-{temperature}###"
            f"top_p-{top_p}###"
            f"max_new_tokens-{max_new_tokens}"
        )
        example_uuid = f"task-{task}###split-{split}###doc_id-{doc_id}"
        return f"{api_uuid}######{example_uuid}"

    def generate_until(self, requests) -> List[str]:
        res = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        lock = threading.Lock()  # make tqdm/cache writes thread-safe

        def process_single(idx, reg):
            # Unpack
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = reg.args

            # Default kwargs
            gen_kwargs.setdefault("max_new_tokens", 8192)
            gen_kwargs.setdefault("temperature", 0)
            gen_kwargs.setdefault("top_p", 1.0)
            if self.max_new_tokens is not None:
                gen_kwargs["max_new_tokens"] = self.max_new_tokens
            if self.temperature is not None:
                gen_kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                gen_kwargs["top_p"] = self.top_p

            # Cache hit
            if self.continual_mode and self.cache_mode == "resume":
                uuid = self.generate_uuid(task, split, doc_id, gen_kwargs)
                if uuid in self.response_cache:
                    return self.response_cache[uuid]

            # === Visual processing ===
            from datasets import concatenate_datasets
            split = split[0].split(",")
            ds = concatenate_datasets([self.task_dict[task][s] for s in split])
            visuals = [doc_to_visual(ds[doc_id])]
            imgs = []
            if None not in visuals:
                visuals = self.flatten(visuals)
                for visual in visuals:
                    if self.modality == "video":
                        pass
                    else:
                        pass

            # === payload ===
            payload = {
                "model": self.model_version,
                "messages": [{"role": "user", "content": [{"type": "text", "text": contexts}]}],
                "max_tokens": gen_kwargs["max_new_tokens"],
                "temperature": gen_kwargs["temperature"],
            }

            # === Call model ===
            response_text = ""
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(**payload)
                    response_text = response.choices[0].message.content
                except Exception as e:
                    error_msg = str(e)
                    eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")
                    if attempt == self.max_retries - 1:
                        eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                        response_text = ""
                    else:
                        time.sleep(self.timeout)
            # Cache write under lock
            if self.continual_mode:
                uuid = self.generate_uuid(task, split, doc_id, gen_kwargs)
                with lock:  # thread-safe
                    self.response_cache[uuid] = response_text

                    # Ensure the cache file exists
                    if not os.path.exists(self.response_persistent_file):
                        with open(self.response_persistent_file, "w") as f:
                            json.dump({}, f)

                    # === Write with an exclusive lock ===
                    try:
                        with portalocker.Lock(self.response_persistent_file, mode="r+", flags=portalocker.LOCK_EX,
                                              timeout=30) as f:
                            # Try to read existing cache
                            try:
                                f.seek(0)
                                cache = json.load(f)
                            except json.JSONDecodeError:
                                cache = {}

                            # Update and write back
                            cache[uuid] = response_text
                            f.seek(0)
                            json.dump(cache, f)
                            f.truncate()
                            f.flush()
                            os.fsync(f.fileno())
                    except Exception as e:
                        print(f"[Cache write failed] {e}")
            with lock:
                pbar.update(1)

            return response_text

        # === Parallel execution ===
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_single, i, reg): i for i, reg in enumerate(requests)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    res[idx] = future.result()
                except Exception as e:
                    res[idx] = ""
                    print(f"[Thread-{idx}] failed: {e}")

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for OpenAI compatible models")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("TODO: Implement loglikelihood for OpenAI compatible models")

def test():
    def _adjust_config(lm, task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(lm, task_obj)},
                }
            else:
                task_obj = task_dict[task_name]
                if type(task_obj) == tuple:
                    group, task_obj = task_obj
                    if task_obj is None:
                        continue
                adjusted_task_dict[task_name] = task_obj
            lm.task_dict[task_name] = task_obj.dataset
        return adjusted_task_dict
    import collections

    # OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    # OPENAI_API_BASE = "https://api.openai.com/v1/"
    # model_version = "gpt-4o"

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # model_version = "gemini-2.0-flash"
    # model_version = "gemini-2.5-flash-preview-05-20"
    model_version = "gemini-2.5-pro"

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    task = "vsibench"
    # task = "mvbench"
    # task = "mvbench_action_sequence"
    # task = "egoschema_subset"
    # task = "egothink_activity"
    # task = "egoplan"
    from lmms_eval.tasks import TaskManager, get_task_dict
    from lmms_eval.evaluator import get_task_list
    openai_compatible = OpenAICompatible(model_version=model_version, max_new_tokens=4096)
    task_manager = TaskManager(model_name="openai_compatible")
    task_dict = get_task_dict(task, task_manager)
    _adjust_config(openai_compatible, task_dict)
    eval_tasks = get_task_list(task_dict)
    task = eval_tasks[0].task
    limit = 10
    task.build_all_requests(limit=limit)
    requests = task.instances

    openai_compatible.generate_until(requests)

if __name__ == "__main__":
    test()

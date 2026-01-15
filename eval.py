from __future__ import print_function, division

import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.multiprocessing as mp

import time
import numpy as np
import random
import json
from tqdm import tqdm

from obj_utils.net_util import ScalarMeanTracker
from runners import nonadaptivea3c_val


def main_eval(args, create_shared_model, init_agent):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    folder_path = "final_result_json"
    os.makedirs(folder_path, exist_ok=True)
    args.results_json = os.path.join(folder_path, args.results_json)

    model_to_open = args.load_model
    processes = []
    res_queue = mp.Queue()

    rank = 0
    max_count = 150

    for scene_type in args.scene_types:
        p = mp.Process(
            target=nonadaptivea3c_val,
            args=(
                rank,
                args,
                model_to_open,
                create_shared_model,
                init_agent,
                res_queue,
                max_count,
                scene_type,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
        rank += 1

    count = 0
    end_count = 0
    train_scalars = ScalarMeanTracker()

    train_scalars_ba = ScalarMeanTracker()
    train_scalars_be = ScalarMeanTracker()
    train_scalars_k = ScalarMeanTracker()
    train_scalars_l = ScalarMeanTracker()

    proc = len(args.scene_types)
    pbar = tqdm(total=max_count * proc)

    try:
        while end_count < proc:
            train_result = res_queue.get()
            pbar.update(1)
            count += 1
            if (args.scene_types[end_count] == 'bathroom'):
                train_scalars_ba.add_scalars(train_result)
            if (args.scene_types[end_count] == 'bedroom'):
                train_scalars_be.add_scalars(train_result)
            if (args.scene_types[end_count] == 'kitchen'):
                train_scalars_k.add_scalars(train_result)
            if (args.scene_types[end_count] == 'living_room'):
                train_scalars_l.add_scalars(train_result)
            if "END" in train_result:
                end_count += 1
                continue
            train_scalars.add_scalars(train_result)

        tracked_means = train_scalars.pop_and_reset()

        tracked_means_ba = train_scalars_ba.pop_and_reset()
        tracked_means_be = train_scalars_be.pop_and_reset()
        tracked_means_k = train_scalars_k.pop_and_reset()
        tracked_means_l = train_scalars_l.pop_and_reset()

    finally:
        for p in processes:
            time.sleep(0.1)
            p.join()

    def tensor_to_jsonable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: tensor_to_jsonable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_jsonable(v) for v in obj]
        else:
            return obj

    with open(args.results_json, "w") as fp:
        json.dump(tensor_to_jsonable(tracked_means), fp, sort_keys=True, indent=4)

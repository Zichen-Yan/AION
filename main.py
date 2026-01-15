from __future__ import print_function, division

import os
import ctypes
import setproctitle
import time

import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from obj_utils import flag_parser

from obj_utils.class_finder import model_class, optimizer_class
from agents.navigation_agent import NavigationAgent
from obj_utils.net_util import ScalarMeanTracker
from eval import main_eval
import json

from runners import nonadaptivea3c_train_mp

os.environ["OMP_NUM_THREADS"] = "1"

def main():
    setproctitle.setproctitle("Train/Test Manager")
    args = flag_parser.parse_arguments()

    create_shared_model = model_class(args.model)
    init_agent = NavigationAgent
    optimizer_type = optimizer_class(args.optimizer)

    if args.eval:
        main_eval(args, create_shared_model, init_agent)
        return

    start_time = time.time()
    local_start_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(start_time))

    if -1 not in args.gpu_ids:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn")

    tb_log_dir = os.path.join(args.log_dir, args.title + "-" + local_start_time_str)
    log_writer = SummaryWriter(log_dir=tb_log_dir)
    with open(os.path.join(tb_log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    shared_model = create_shared_model(args)
    if shared_model is not None:
        shared_model.share_memory()
        optimizer = optimizer_type(filter(lambda p: p.requires_grad, shared_model.parameters()), args)
        optimizer.share_memory()
    else:
        optimizer = None

    train_total_ep = 0
    n_frames = 0

    processes = []
    end_flag = mp.Value(ctypes.c_bool, False)
    train_res_queue = mp.Queue()

    for rank in range(0, args.workers):
        p = mp.Process(
            target=nonadaptivea3c_train_mp,
            args=(
                rank,
                args,
                shared_model,
                init_agent,
                optimizer,
                train_res_queue,
                end_flag,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)

    print("Train agents created.")

    train_thin = args.train_thin
    train_scalars = ScalarMeanTracker()

    try:
        while n_frames <= args.max_steps:
            train_result = train_res_queue.get()
            train_scalars.add_scalars(train_result)
            train_total_ep += 1
            n_frames += train_result["ep_length"]
            if (train_total_ep % train_thin) == 0:
                log_writer.add_scalar("n_frames", n_frames, train_total_ep)
                tracked_means = train_scalars.pop_and_reset()
                for k in tracked_means:
                    log_writer.add_scalar(k + "/train", tracked_means[k], train_total_ep)

        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        state_to_save = shared_model.state_dict()
        save_path = os.path.join(
            args.save_model_dir,
            "{0}_{1}_{2}_{3}.dat".format(
                args.title, n_frames, train_total_ep, local_start_time_str
            ),
        )
        torch.save(state_to_save, save_path)

        state = {
            'epoch': train_total_ep,
            'state_dict': shared_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        save_model_path = os.path.join(
            args.save_model_dir,
            "{0}_{1}_{2}.tar".format(args.title, train_total_ep, local_start_time_str),
        )
        torch.save(state, save_model_path)

    finally:
        log_writer.close()
        end_flag.value = True
        for p in processes:
            time.sleep(0.1)
            p.join()

if __name__ == "__main__":
    main()

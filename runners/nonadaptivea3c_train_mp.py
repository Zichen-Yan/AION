from __future__ import division
import time

from datasets.data import get_seen_data
import copy
import setproctitle
from models.model_io import ModelOptions

import random
import torch

from .train_util import (
    compute_loss,
    new_episode,
    run_episode,
    transfer_gradient_from_player_to_shared,
    end_episode,
    reset_player,
)


def nonadaptivea3c_train_mp(
    rank,
    args,
    shared_model,
    initialize_agent,
    optimizer,
    res_queue,
    end_flag,
):
    scenes, possible_targets, targets, rooms = get_seen_data(args.scene_types, args.train_scenes, args.split)

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    setproctitle.setproctitle("Training Agent: {}".format(rank))

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    print("GPU ID: {}".format(gpu_id))
    if gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        torch.cuda.manual_seed(args.seed + rank)

    idx = list(range(len(args.scene_types)))
    random.shuffle(idx)

    local_model = copy.deepcopy(shared_model)
    local_model = local_model.to(f"cuda:{gpu_id}")

    player = initialize_agent(local_model, args, rank, gpu_id=gpu_id)

    model_options = ModelOptions()
    j = 0

    while not end_flag.value:
        total_reward = 0
        total_collision_cnt = 0
        player.eps_len = 0

        new_episode(args, player, scenes[idx[j]], possible_targets, targets[idx[j]], rooms[idx[j]])
        player_start_time = time.time()

        # Train on the new episode.
        while not player.done:
            player.sync_with_shared(shared_model)
            total_reward, total_collision_cnt = run_episode(player, args, total_reward, total_collision_cnt, model_options, True)
            loss = compute_loss(args, player, gpu_id, model_options)

            # Compute gradient.
            player.model.zero_grad()
            loss["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(player.model.parameters(), 100.0)
            transfer_gradient_from_player_to_shared(player, shared_model, gpu_id)
            optimizer.step()

            if not player.done:
                reset_player(player)

        for k in loss:
            if k=="entropy":
                loss[k] = loss[k].item()
            else:
                loss[k] = loss[k].item() / len(player.rewards)

        if args.episode_type == "ExplorationTrainEpisode":
            total_cnt = player.episode.forward_cnt+player.episode.turn_cnt+player.episode.updown_cnt+1e-8
            forward_ratio = player.episode.forward_cnt/total_cnt
            turn_ratio = player.episode.turn_cnt/total_cnt
            updown_ratio = player.episode.updown_cnt/total_cnt

            end_episode(
                player,
                res_queue,
                loss=loss,
                title=args.scene_types[idx[j]],
                total_time=time.time() - player_start_time,
                total_reward=total_reward,
                total_collision_cnt=total_collision_cnt,
                forward_ratio=forward_ratio,
                turn_ratio=turn_ratio,
                updown_ratio=updown_ratio,
            )
        else:
            end_episode(
                player,
                res_queue,
                loss=loss,
                title=args.scene_types[idx[j]],
                total_time=time.time() - player_start_time,
                total_reward=total_reward,
                total_collision_cnt=total_collision_cnt,
            )
        reset_player(player)

        j = (j + 1) % len(args.scene_types)

    player.exit()


from __future__ import division

import time
import setproctitle
import copy
from datasets.data import name_to_num, get_seen_data, get_unseen_data

from models.model_io import ModelOptions

from .train_util import (
    new_episode,
    run_episode,
    end_episode,
    reset_player,
    compute_spl,
    get_bucketed_metrics
)
from .visualization_util import *

def nonadaptivea3c_val(
    rank,
    args,
    model_to_open,
    model_create_fn,
    initialize_agent,
    res_queue,
    max_count,
    scene_type,
):
    args.max_episode_length = 100
    if args.get_seen_data:
        scenes, possible_targets, targets, rooms = get_seen_data(args.scene_types, args.val_scenes, args.split)
    else:
        scenes, possible_targets, targets, rooms = get_unseen_data(args.scene_types, args.val_scenes, args.split)
    num = name_to_num(scene_type)
    scenes = scenes[num]
    targets = targets[num]
    rooms = rooms[num]

    setproctitle.setproctitle("Agent: {}".format(rank))

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    shared_model = model_create_fn(args)

    if model_to_open != "":
        try:
            saved_state = torch.load(model_to_open, map_location=lambda storage, loc: storage)
            saved_state.pop("goal_text_emb", None)
            shared_model.load_state_dict(saved_state, strict=False)
        except:
            shared_model.load_state_dict(model_to_open)

    local_model = copy.deepcopy(shared_model)
    local_model = local_model.to(f"cuda:{gpu_id}")

    player = initialize_agent(local_model, args, rank, gpu_id=gpu_id)
    player.sync_with_shared(shared_model)
    count = 0

    model_options = ModelOptions()
    j = 0

    while count < max_count:
        total_reward = 0
        total_collision = 0
        player.eps_len = 0

        new_episode(args, player, scenes, possible_targets, targets, rooms)
        player_start_state = copy.deepcopy(player.environment.controller.state)
        player_start_time = time.time()

        # Train on the new episode.
        with torch.no_grad():
            while not player.done:
                total_reward, total_collision = run_episode(player, args, total_reward, total_collision, model_options, False)
                if not player.done:
                    reset_player(player)

        spl, best_path_length = compute_spl(player, player_start_state)
        bucketed_spl = get_bucketed_metrics(spl, best_path_length, player.success)
        end_episode(
            player,
            res_queue,
            total_time=time.time() - player_start_time,
            total_reward=total_reward,
            total_collision=total_collision,
            spl=spl,
            **bucketed_spl,
        )

        if args.save_visuals:
            generate_visualizations(player, count, rank)
        if args.save_episode_data:
            save_episode_data(player, count, rank)

        count += 1
        reset_player(player)

        j = (j + 1) % len(args.scene_types)

    player.exit()
    res_queue.put({"END": True})

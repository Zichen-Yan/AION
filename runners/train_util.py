from __future__ import division

import torch


def run_episode(player, args, total_reward, total_collision, model_options, training):
    for step in range(args.rollout_steps if training else args.max_episode_length):
        player.action(model_options, training, timestep=step)
        total_reward = total_reward + player.reward
        total_collision += int(player.collision)
        if player.done:
            break
    return total_reward, total_collision


def new_episode(
    args,
    player,
    scenes,
    possible_targets=None,
    targets=None,
    rooms=None,
):
    player.episode.new_episode(args, scenes, possible_targets, targets, rooms)
    player.reset_hidden()
    player.done = False

def a3c_loss(args, player, gpu_id, model_options):
    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")

    if player.done:
        R = torch.zeros(1, device=device)
    else:
        with torch.no_grad():
            output = player.eval_at_state(model_options)
            R = output.value.squeeze().to(device)

    player.values.append(R)

    policy_loss = 0.0
    value_loss = 0.0
    gae = 0.0

    for i in reversed(range(len(player.rewards))):
        R = args.gamma * R + player.rewards[i]
        advantage = R - player.values[i]

        value_loss += 0.5 * advantage.pow(2)

        next_value = player.values[i + 1] if i + 1 < len(player.values) else torch.zeros_like(R)
        delta = player.rewards[i] + args.gamma * next_value - player.values[i]

        gae = gae * args.gamma * args.tau + delta

        policy_loss += -player.log_probs[i] * gae - args.beta * player.entropies[i]

    entropy_mean = torch.stack(player.entropies).mean()
    return policy_loss, value_loss, entropy_mean

def transfer_gradient_from_player_to_shared(player, shared_model, gpu_id):
    """ Transfer the gradient from the player's model to the shared model
        and step """
    for param, shared_param in zip(
        player.model.parameters(), shared_model.parameters()
    ):
        if shared_param.requires_grad:
            if param.grad is None:
                shared_param._grad = torch.zeros(shared_param.shape)
            elif gpu_id < 0:
                shared_param._grad = param.grad
            else:
                shared_param._grad = param.grad.cpu()

def get_params(shared_model, gpu_id):
    """ Copies the parameters from shared_model into theta. """
    theta = {}
    for name, param in shared_model.named_parameters():
        # Clone and detach.
        param_copied = param.clone().detach().requires_grad_(True)
        if gpu_id >= 0:
            theta[name] = param_copied.to(torch.device("cuda:{}".format(gpu_id)))
        else:
            theta[name] = param_copied
    return theta

def reset_player(player):
    # --- End of addition ---
    player.eps_len = 0
    player.clear_actions()
    player.repackage_hidden()

def compute_loss(args, player, gpu_id, model_options):
    policy_loss, value_loss, entropy = a3c_loss(args, player, gpu_id, model_options)
    total_loss = policy_loss + 0.5 * value_loss
    return dict(total_loss=total_loss, policy_loss=policy_loss, value_loss=value_loss, entropy=entropy)

def end_episode(
    player, res_queue=None, loss=None, title=None, **kwargs
):
    results = {
        "done_count": player.episode.done_count,
        "ep_length": player.eps_len,
        "move_length": player.episode.move_steps,
        "success": int(player.success),
    }
    if loss is not None:
        results.update({k: v for k, v in loss.items()})

    results.update(**kwargs)
    if res_queue is not None:
        res_queue.put(results)
    return results


def get_bucketed_metrics(spl, best_path_length, success):
    out = {}
    for i in [1, 5]:
        if best_path_length >= i:
            out["GreaterThan/{}/success".format(i)] = success
            out["GreaterThan/{}/spl".format(i)] = spl
    return out

def compute_spl(player, start_state):
    if not player.success:
        return 0, float('inf')

    target_state = player.environment.controller.state
    
    try:
        path_nodes, best_path_len = player.environment.controller.shortest_path_to_target(
            start_state, target_state
        )
    except Exception as e:
        print(f"[Warning] Failed to compute shortest path: {e}")
        return 0, float('inf')

    actual_path_len = float(player.episode.move_steps) 
    if best_path_len == float('inf') or actual_path_len == 0:
        return (1.0 if best_path_len == 0 else 0.0), best_path_len
    
    spl = best_path_len / max(actual_path_len, best_path_len)
    return spl, best_path_len



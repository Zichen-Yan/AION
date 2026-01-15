""" Base class for all Agents. """
from __future__ import division

import torch
import torch.nn.functional as F


class ThorAgent:
    """ Base class for all actor-critic agents. """

    def __init__(
        self, model, args, rank, episode=None, max_episode_length=1e3, gpu_id=-1):
        self.gpu_id = gpu_id

        self.args = args
        self.model = model
        self.episode = episode
        self.eps_len = 0
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.collision = False
        self.entropies = []
        self.done = True
        self.reward = 0
        self.env_done = False
        self.hidden = None
        self.actions = []
        self.max_episode_length = max_episode_length
        self.success = False
        torch.manual_seed(args.seed + rank)
        if gpu_id >= 0:
            torch.cuda.manual_seed(args.seed + rank)

        self.hidden_state_sz = args.hidden_state_sz
        self.action_space = args.action_space

        self.rgb_seq = []
        self.trajectory = []

    def sync_with_shared(self, shared_model):
        """ Sync with the shared model. """
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model.load_state_dict(shared_model.state_dict())
        else:
            self.model.load_state_dict(shared_model.state_dict())

    def eval_at_state(self, model_options):
        """ Eval at state. """
        raise NotImplementedError()

    @property
    def environment(self):
        """ Return the current environmnet. """
        return self.episode.environment

    @property
    def state(self):
        """ Return the state of the agent. """
        raise NotImplementedError()

    @state.setter
    def state(self, value):
        raise NotImplementedError()

    def print_info(self):
        """ Print the actions. """
        for action in self.actions:
            print(action)

    def _increment_episode_length(self):
        self.eps_len += 1
        if self.eps_len >= self.max_episode_length:
            self.env_done = True
        else:
            self.env_done = False

    def action(self, model_options, training, timestep=None):
        """ Train the agent. """
        if training:
            self.model.train()
        else:
            self.model.eval()
        model_options.timestep = timestep
        out = self.eval_at_state(model_options)
        self.hidden = out.hidden

        prob = F.softmax(out.logit, dim=1)
        action = torch.multinomial(prob, num_samples=1)

        log_prob = F.log_softmax(out.logit, dim=1)
        self.last_action = action
        self.last_action_probs = prob.detach()

        entropy = -(log_prob * prob).sum(dim=1)
        log_prob = log_prob.gather(1, action)
        self.reward, self.agent_done, task_success = self.episode.step(action[0][0])

        if self.args.eval:
            current_state = self.episode.environment.last_event.metadata['agent']
            self.trajectory.append({
                'state': {
                    'position': current_state['position'].copy(),
                    'rotation': current_state['rotation'].copy(),
                    'horizon': current_state['cameraHorizon']
                },
                'timestep': timestep,
                'action': self.episode.actions[action[0][0].item()]
            })

        self.entropies.append(entropy)
        self.values.append(out.value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.actions.append(action)
        self.collision = self.episode.environment.controller.collision
        self.episode.current_frame = self.state()
        self._increment_episode_length()

        if self.episode.strict_done and self.agent_done:
            self.success = task_success
            self.done = True
            return

        if self.env_done:
            self.done = True
            self.success = False
        else:
            self.done = False

    def reset_hidden(self, volatile=False):
        """ Reset the hidden state of the LSTM. """
        raise NotImplementedError()

    def repackage_hidden(self, volatile=False):
        """ Repackage the hidden state of the LSTM. """
        raise NotImplementedError()

    def clear_actions(self):
        """ Clear the information stored by the agent. """
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.rewards = []
        self.trajectory = []

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        raise NotImplementedError()

    def exit(self):
        """ Called on exit. """
        pass

    def reset_episode(self):
        """ Reset the episode so that it is identical. """
        return self.episode.reset()

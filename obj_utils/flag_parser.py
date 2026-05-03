import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="AION")

    parser.add_argument(
        "--title", type=str, default="default_title", help="Info for logging."
    )

    # env ======================================================
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=64,
        metavar="RS",
        help="number of steps to collect in each rollout (default: 64)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="W",
        help="how many training processes to use (default: 4)",
    )
    parser.add_argument(
        "--max_episode_length",
        type=int,
        default=30,
        metavar="M",
        help="maximum length of an episode (default: 30)",
    )
    parser.add_argument(
        "--grid_size",
        type=float,
        default=0.25,
        metavar="GS",
        help="The grid size used to discretize AI2-THOR maps.",
    )
    parser.add_argument(
        "--success_distance",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--get_seen_data",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--max_steps",
        type=float,
        default=1e7
    )
    parser.add_argument(
        "--train_scenes",
        type=str,
        default="[1-20]",
        help="scenes for training."
    )
    parser.add_argument(
        "--val_scenes",
        type=str,
        default="[21-30]",
        help="old validation scenes before formal split.",
    )
    parser.add_argument(
        "--possible_targets",
        type=str,
        default="FULL_OBJECT_CLASS_LIST",
        help="all possible objects.",
    )
    parser.add_argument(
        "--train_targets",
        type=str,
        default=None,
        help="specific objects for this experiment from the object list.",
    )
    parser.set_defaults(strict_done=True)
    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="The field of view to use."
    )
    parser.add_argument(
        "--vis",
        type=str2bool,
        default=True,
        help="whether to show UI",
    )
    parser.add_argument(
        "--snapToGrid",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--episode_type",
        type=str,
        default="NavTrainEpisode",
        help="Which type of episode.",
    )
    parser.add_argument(
        "--scene_types",
        nargs="+",
        default=["kitchen", "living_room", "bedroom", "bathroom"],
    )
    # env ======================================================

    # training ======================================================
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.00,
        metavar="T",
        help="parameter for GAE (default: 1.00)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1e-2,
        help="entropy regularization term"
    )
    parser.add_argument(
        "--optimizer",
        default="SharedAdam",
        metavar="OPT",
        help="shared optimizer choice of SharedAdam or SharedRMSprop",
    )
    parser.add_argument(
        "--save_model_dir",
        default="trained_models/",
        metavar="SMD",
        help="folder to save trained model",
    )
    parser.add_argument(
        "--log_dir",
        default="runs/",
        metavar="LG",
        help="folder to save logs"
    )
    parser.add_argument(
        "--gpu_ids",
        type=int,
        default=-1,
        nargs="+",
        help="GPUs to use [-1 CPU only] (default: -1)",
    )
    parser.add_argument(
        "--amsgrad",
        default=True,
        metavar="AM",
        help="Adam optimizer amsgrad parameter"
    )
    parser.add_argument(
        "--train_thin",
        type=int,
        default=250,
        help="How often to print"
    )
    parser.add_argument(
        "--offline_data_dir",
        type=str,
        default="./data/thor_v1_offline_data",  # thor_offline_data
        help="where dataset is stored.",
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default="metrics.json",
        help="Write the results."
    )
    # training ======================================================

    # Model ======================================================
    parser.add_argument(
        "--load_model",
        type=str,
        default="",
        help="Path to load a saved model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BaseModel",
        help="Model to use.")

    parser.add_argument(
        "--glove_dim",
        type=int,
        default=300,
        help="which dimension of the glove vector to use",
    )
    parser.add_argument(
        "--action_space",
        type=int,
        default=6,
        help="space of possible actions."
    )
    parser.add_argument(
        "--hidden_state_sz",
        type=int,
        default=512,
        help="size of hidden state of LSTM."
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="run the test code")
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.25,
        help="The dropout ratio to use (default is no dropout).",
    )
    parser.add_argument(
        "--test_or_val",
        default="test",
        help="test or val")
    parser.add_argument(
        "--add_clip_align",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--add_stats",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--add_rgb",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--add_depth",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--add_dis_reward",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--add_bbox_reward",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--add_parent_reward",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--add_collision_reward",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "--split",
        default="18/4",
        help="class split")
    # Model ======================================================

    # Explorer ======================================================
    parser.add_argument(
        '--scene', 
        type=str, 
        default='ithor', 
        choices=['ithor', 'procthor'],
        help='Which type of scene to use.'
    )
    # Explorer ======================================================

    parser.add_argument(
        '--save_visuals',
        type=str2bool,
        default=False,
        help='Save trajectory map and third-person video for the first few episodes.'
    )
    parser.add_argument(
        '--save_episode_data',
        type=str2bool,
        default=False,
        help='Save trajectory map and third-person video for the first few episodes.'
    )
    args = parser.parse_args()

    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

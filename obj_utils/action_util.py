def get_actions(args):
    if args.action_space == 6:
        return ["up", "down", "forward", "turn_left", "turn_right", "Done"]
    elif args.action_space == 5:
        return ["up", "down", "forward", "turn_left", "turn_right"]
    elif args.action_space == 3:
        return ["linear", "angular", "z", "Done"]
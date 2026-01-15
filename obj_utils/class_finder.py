import models
import episodes
import optimizers
import argparse

def model_class(class_name):
    if class_name not in models.__all__:
       raise argparse.ArgumentTypeError("Invalid model {}; choices: {}".format(
           class_name, models.__all__))
    return getattr(models, class_name)
def episode_class(class_name):
    if class_name not in episodes.__all__:
       raise argparse.ArgumentTypeError("Invalid episodes {}; choices: {}".format(
           class_name, episodes.__all__))
    return getattr(episodes, class_name)
def optimizer_class(class_name):
    if class_name not in optimizers.__all__:
       raise argparse.ArgumentTypeError("Invalid optimizer {}; choices: {}".format(
           class_name, optimizers.__all__))
    return getattr(optimizers, class_name)

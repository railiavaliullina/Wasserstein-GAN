from trainers.trainer import Trainer
from configs.train_config import cfg as train_cfg


class Executor(object):
    """
    Class for running main class methods which run whole algorithm.
    """
    @staticmethod
    def run():
        trainer = Trainer(train_cfg)
        trainer.train()


if __name__ == '__main__':
    executor = Executor()
    executor.run()

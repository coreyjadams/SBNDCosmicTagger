#/usr/bin/env python
import yaml
import sys

from trainer import uresnet_trainer


def main(params):
    with uresnet_trainer(params) as trainer:
        trainer.batch_process()

if __name__ == '__main__':

    if len(sys.argv) < 2:
        sys.stdout.write('Requires configuration file.  [python train.py config.yml]\n')
        sys.stdout.flush()
        exit()

    config = sys.argv[-1]

    with open(config, 'r') as f:
        params = yaml.load(f)

    main(params)
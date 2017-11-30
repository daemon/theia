from collections import ChainMap
import argparse

import kumon
import mnist

class ConfigBuilder(object):
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.default_config.items():
            key = "--{}".format(key)
            if isinstance(value, tuple):
                parser.add_argument(key, default=list(value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(key, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool) and not value:
                parser.add_argument(key, action="store_true")
            else:
                parser.add_argument(key, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        if not parser:
            parser = self.build_argparse()
        args = vars(parser.parse_known_args()[0])
        return ChainMap(args, self.default_config)

def main():
    global_conf = dict(no_cuda=False, input_file="output.pt")
    builder = ConfigBuilder(global_conf)
    parser = builder.build_argparse()
    parser.add_argument("--input", type=str)
    parser.add_argument("--type", type=str, choices=["kumon"], default="kumon")
    config = builder.config_from_argparse(parser)

    mnist.init_model(config["input_file"], not config["no_cuda"])
    document = kumon.Document(config["input"])

if __name__ == "__main__":
    main()
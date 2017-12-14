from collections import ChainMap
import argparse
import io

import cherrypy

import kumon.document as doc
import kumon.model as mod
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

class GradeEndpoint(object):
    exposed = True

    @cherrypy.tools.json_out()
    def POST(self, image):
        image_bytes = image.file.read()
        reader = io.BytesIO(image_bytes)
        document = doc.Document(reader)
        return dict(points=document.mark_dict)

def start_server(host, port):
    cherrypy.server.socket_host = host
    cherrypy.server.socket_port = port

    rest_conf = {"/": {
        "request.dispatch": cherrypy.dispatch.MethodDispatcher()
    }}
    cherrypy.tree.mount(GradeEndpoint(), "/grade", rest_conf)
    cherrypy.engine.start()
    cherrypy.engine.block()

def main():
    global_conf = dict(no_cuda=False, mnist_model="mnist.pt", kumon_model="kumon.pt", port=16888, host="0.0.0.0")
    builder = ConfigBuilder(global_conf)
    parser = builder.build_argparse()
    parser.add_argument("--type", type=str, choices=["kumon"], default="kumon")
    config = builder.config_from_argparse(parser)

    mnist.init_model(config["mnist_model"], not config["no_cuda"])
    mod.init_model(config["kumon_model"], not config["no_cuda"])
    start_server(config["host"], config["port"])

if __name__ == "__main__":
    main()
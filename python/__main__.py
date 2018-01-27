from collections import ChainMap
import argparse
import io
import json
import os
import uuid

import cherrypy
import numpy as np

from server.tag import start_tag_server
import db.base as base
import db.assignment as asst
import db.user
import dump
import kumon.document as doc
import kumon.model as mod
import mnist

def json_in(f):
    def merge_dicts(x, y):
        z = x.copy()
        z.update(y)
        return z
    def wrapper(*args, **kwargs):
        cl = cherrypy.request.headers["Content-Length"]
        data = json.loads(cherrypy.request.body.read(int(cl)).decode("utf-8"))
        kwargs = merge_dicts(kwargs, data)
        return f(*args, **kwargs)
    return wrapper

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

class AssignmentEndpoint(object):
    exposed = True
    @cherrypy.tools.json_out()
    @json_in
    def POST(self, **kwargs):
        print(kwargs)
        try:
            auth_token = kwargs["auth_token"]
            grade_tokens = kwargs["grade_tokens"]
            duration_ms = kwargs["duration_ms"]
            profile_id = kwargs["profile_id"]
        except:
            cherrypy.response.status = 400
            return
        user = db.user.session_store.get_user(auth_token)
        profile = db.user.Profile.find(id=profile_id)
        if profile.owner_id != user.id:
            raise ValueError
        a, wksts = asst.create_assignment(profile_id, grade_tokens, duration_ms)
        return dict(id=a.id, worksheet_ids=[w.id for w in wksts])

    @cherrypy.tools.json_out()
    def GET(self, **kwargs):
        try:
            auth_token = kwargs["token"]
        except:
            cherrypy.response.status = 400
            return
        try:
            user = db.user.session_store.get_user(auth_token)
            assignments = [a.dict() for a in asst.Assignment.find(user.id)]
            return dict(assignments=assignments)
        except:
            cherrypy.response.status = 403

class ProfileEndpoint(object):
    exposed = True
    @cherrypy.tools.json_out()
    @json_in
    def POST(self, **kwargs):
        try:
            name = kwargs["name"]
            token = kwargs["token"]
        except:
            cherrypy.response.status = 400
            return
        try:
            user = db.user.session_store.get_user(token)
            profile = db.user.Profile.create(name, user.id)
            return dict(profile_id=profile.id)
        except:
            cherrypy.response.status = 403

    @cherrypy.tools.json_out()
    def GET(self, **kwargs):
        try:
            token = kwargs["token"]
        except:
            cherrypy.response.status = 400
            return
        try:
            user = db.user.session_store.get_user(token)
            profiles = db.user.Profile.list_profiles(user.id)
            return dict(profiles=[p.dict() for p in profiles])
        except:
            cherrypy.response.status = 403

class UserEndpoint(object):
    exposed = True

    @cherrypy.tools.json_out()
    def GET(self, **kwargs):
        try:
            email = kwargs["email"]
            password = kwargs["password"]
        except:
            cherrypy.response.status = 400
            return
        user = db.user.User.find(email=email)
        if not user:
            return dict(success=False, message="Incorrect e-mail and password combination.")
        token = user.login(password)
        if not token:
            return dict(success=False, message="Incorrect e-mail and password combination.")
        return dict(success=True, token=token, message="Login successful.")

    @cherrypy.tools.json_out()
    @json_in
    def POST(self, **kwargs):
        try:
            email = kwargs["email"]
            password = kwargs["password"]
        except:
            cherrypy.response.status = 400
            return
        try:
            user, profile = db.user.User.register(email, password)
            token = user.login(password)
            return dict(success=True, message="Registration successful.", token=token, profile_id=profile.id)
        except db.user.UserExistsError:
            return dict(success=False, message="E-mail already exists.")

class GradeEndpoint(object):
    exposed = True
    def __init__(self, internal_save_path="local_data"):
        self.save_path = internal_save_path

    @cherrypy.tools.json_out()
    def POST(self, *args, **kwargs):
        try:
            image = kwargs["image"]
            tag = int(kwargs["tag"])
        except:
            cherrypy.response.status = 400
            return
        image_bytes = image.file.read()
        reader = io.BytesIO(image_bytes)
        document = doc.Document(reader)
        try:
            mark_dict, grade = document.grade()
        except:
            mark_dict = []
            grade = 0
        token = asst.AssignmentStore.create_token(grade)
        path = "{}.jpg".format(os.path.join(self.save_path, str(uuid.uuid4())))
        asst.WorksheetStore.set_path(token, path)
        asst.WorksheetStore.set_points_json(token, mark_dict)
        with open(path, "wb") as f:
            f.write(image_bytes)
        return dict(points=mark_dict, token=token, grade=grade, tag=tag)

class CheckTokenEndpoint(object):
    exposed = True
    @cherrypy.tools.json_out()
    def GET(self, token):
        token = db.user.session_store.get_user(token)
        return dict(success=token is not None)

class SyncAssignmentEndpoint(object):
    exposed = True
    @cherrypy.tools.json_out()
    def POST(self, **kwargs):
        try:
            token = kwargs["token"]
            assignments = kwargs["assignments"]
        except:
            cherrpy.response.status = 400
            return
        try:
            user = db.user.session_store.get_user(token)
            metadatas = asst.Assignment.sync(user, [asst.AssignmentMetadata.from_dict(a) for a in assignments])
            return dict(metadatas=metadatas)
        except:
            return 403

def init_db():
    with open("config.json") as f:
        cfg = json.loads(f.read())["db_config"]
    base.initialize(cfg)

def start_main_server(host, port):
    init_db()
    cherrypy.server.socket_host = host
    cherrypy.server.socket_port = port

    rest_conf = {"/": {
        "request.dispatch": cherrypy.dispatch.MethodDispatcher()
    }}
    cherrypy.tree.mount(AssignmentEndpoint(), "/assignment/", rest_conf)
    cherrypy.tree.mount(GradeEndpoint(), "/grade", rest_conf)
    cherrypy.tree.mount(UserEndpoint(), "/user/", rest_conf)
    cherrypy.tree.mount(CheckTokenEndpoint(), "/check_token/", rest_conf)
    cherrypy.tree.mount(ProfileEndpoint(), "/profile/", rest_conf)
    cherrypy.tree.mount(SyncAssignmentEndpoint(), "/sync_assignments/", rest_conf)
    cherrypy.engine.start()
    cherrypy.engine.block()

def debug_image(image):
    document = doc.Document(image, debug=True)

def main():
    global_conf = dict(no_cuda=False, mnist_model="mnist.pt", kumon_model="kumon.pt", port=16888, host="0.0.0.0")
    builder = ConfigBuilder(global_conf)
    parser = builder.build_argparse()
    parser.add_argument("--type", type=str, choices=["kumon"], default="kumon")
    parser.add_argument("command", type=str, choices=["start", "debug", "tag", "add_local", "dump_boxes"])
    config = builder.config_from_argparse(parser)

    mnist.init_model("single", config["mnist_model"], not config["no_cuda"])
    mod.init_model(config["kumon_model"], not config["no_cuda"])
    if config["command"] == "start":
        start_main_server(config["host"], config["port"])
    elif config["command"] == "tag":
        start_tag_server(config["host"], config["port"])
    elif config["command"] == "add_local":
        init_db()
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_dir", type=str, default=".")
        parser.add_argument("--profile_name", type=str)
        parser.add_argument("--email", type=str)
        flags, _ = parser.parse_known_args()
        asst.add_local(flags.image_dir, flags.email, flags.profile_name)
    elif config["command"] == "debug":
        parser = argparse.ArgumentParser()
        parser.add_argument("--image", type=str)
        flags, _ = parser.parse_known_args()
        debug_image(flags.image)
    elif config["command"] == "dump_boxes":
        init_db()
        parser = argparse.ArgumentParser()
        parser.add_argument("--format", type=str, default="raw", choices=["raw", "voc", "csv"])
        parser.add_argument("--output", type=str, default="output")
        args, _ = parser.parse_known_args()
        boxes = asst.dump_worksheet_boxes()
        dump.make_writer(args.format, args.output, boxes).write()

if __name__ == "__main__":
    main()

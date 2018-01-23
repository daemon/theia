from urllib.parse import urlencode
import json
import os

from mako.template import Template
from mako.lookup import TemplateLookup
import cherrypy
import db.assignment as asst
import mako

import db.base as base

def route_mako(path):
    def decorator(f):
        def wrapper(*args, **kwargs):
            result = f(*args, **kwargs)
            try:
                return f.template.render(data=result, request=cherrypy.request, response=cherrypy.response)
            except AttributeError:
                f.template = args[0].template_lookup.get_template(path)
                return f.template.render(data=result, request=cherrypy.request, response=cherrypy.response)
        return wrapper
    return decorator

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

class SubmitEndpoint(object):
    exposed = True
    @cherrypy.tools.json_out()
    @json_in
    def POST(self, **kwargs):
        try:
            rects = kwargs["rects"]
            wkst_id = kwargs["id"]
        except KeyError:
            cherrypy.response.code = 400
            return
        if len(rects) == 0:
            return
        asst.WorksheetBox.insert_all(wkst_id, rects)

class RootEndpoint(object):
    def __init__(self, template_lookup):
        self.template_lookup = template_lookup
        self.save_path = "local_data"

    @cherrypy.expose
    @route_mako("editor.mako")
    def editor(self, worksheet_id):
        worksheet = asst.Worksheet.find_by_id(worksheet_id)
        boxes = asst.WorksheetBox.find_all(worksheet_id)
        labels = asst.Label.list_all()
        return dict(boxes=boxes, worksheet=worksheet, labels=labels)

    @cherrypy.expose
    @route_mako("tagger.mako")
    def tag(self, page_no=0, email=None, profile=None):
        page_no = int(page_no)
        user_params = dict()
        if email:
            user_params["email"] = email
        if profile:
            user_params["profile"] = profile
        worksheets, count = asst.Worksheet.get_page(page_no=page_no)
        return dict(worksheets=worksheets, page_no=page_no, url_params=urlencode(user_params))

def start_tag_server(host, port):
    with open("config.json") as f:
        cfg = json.loads(f.read())["db_config"]
    base.initialize(cfg)
    cherrypy.server.socket_host = host
    cherrypy.server.socket_port = port
    cherrypy_conf = {
        "/assets": {
          "tools.staticdir.on": True,
          "tools.staticdir.dir": os.path.join(os.getcwd(), "assets")
        }
    }
    rest_conf = {"/": {
        "request.dispatch": cherrypy.dispatch.MethodDispatcher()
    }}

    template_lookup = TemplateLookup(directories=["templates"], input_encoding="utf-8",
        output_encoding="utf-8", encoding_errors="replace")
    cherrypy.tree.mount(RootEndpoint(template_lookup), "/", cherrypy_conf)
    cherrypy.tree.mount(SubmitEndpoint(), "/create_tag/", rest_conf)
    cherrypy.engine.start()
    cherrypy.engine.block()
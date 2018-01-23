import json
import os

import numpy as np
from sqlalchemy import *
from sqlalchemy.sql import func
import redis

from db.base import sec_random_gen
import db.base as base
import db.user as user

assignment = Table("assignments", base.Base.metadata,
    Column("id", Integer, primary_key=True),
    Column("profile_id", Integer, ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False),
    Column("grade", Float, nullable=False),
    Column("creation_time", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("duration_ms", Integer, nullable=False, server_default="0"))

worksheet = Table("worksheets", base.Base.metadata,
    Column("id", Integer, primary_key=True),
    Column("assignment_id", Integer, ForeignKey("assignments.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("image_path", String(4096), nullable=False),
    Column("points_json", String(8192), nullable=False, server_default='""'))

box_label = Table("box_labels", base.Base.metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(64), nullable=False))

worksheet_box = Table("worksheet_boxes", base.Base.metadata,
    Column("id", Integer, primary_key=True),
    Column("worksheet_id", Integer, ForeignKey("worksheets.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("label_id", Integer, ForeignKey("box_labels.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("box_data", String(64), nullable=False))

class AssignmentStore:
    store = redis.StrictRedis(host="127.0.0.1", port=6379, db=0)
    @classmethod
    def create_token(cls, result):
        token = sec_random_gen(16)
        cls.store.set(token, result, ex=3600 * 8)
        return token

    @classmethod
    def get_result(cls, token):
        return cls.store.get(token)

class WorksheetStore:
    store = redis.StrictRedis(host="127.0.0.1", port=6379, db=0)
    @classmethod
    def set_path(cls, token, path):
        token = token + "_path"
        cls.store.set(token, path, ex=3600 * 8)
        return token

    @classmethod
    def set_points_json(cls, token, points_json):
        token = token + "_json"
        cls.store.set(token, points_json, ex=3600 * 8)
        return token

    @classmethod
    def get_points_json(cls, token):
        return cls.store.get(token + "_json")

    @classmethod
    def get_path(cls, token):
        return cls.store.get(token + "_path")

@base.access_point()
def add_local(image_dir, email, profile_name, connection=base.INIT_CONNECTION):
    profiles = user.Profile.list_profiles(user.User.find(email=email, connection=connection).id, connection=connection)
    for p in profiles:
        if p.name.lower() == profile_name.lower():
            break
    full_paths = []
    for filename in os.listdir(image_dir):
        if filename.endswith("jpg"):
            full_paths.append(os.path.join(image_dir, filename))
    create_local_assignment(p.id, full_paths, connection=connection)

@base.access_point()
def create_local_assignment(profile_id, image_paths, connection=base.INIT_CONNECTION):
    assign = Assignment.create(profile_id, 0, 0, connection=connection)
    Worksheet.create_all(assign.id, image_paths, [""] * len(image_paths), connection=connection)

class WorksheetBox(object):
    def __init__(self, *args, label=None):
        base.init_from_row(self, base.column_names(worksheet_box), args)
        self.label = label

    def dict(self):
        return dict(box_data=json.loads(self.box_data), label_id=self.label.id)

    @classmethod
    @base.access_point()
    def insert_all(cls, worksheet_id, rects, connection=base.INIT_CONNECTION):
        c = connection.cursor()
        stmt = "INSERT INTO worksheet_boxes (worksheet_id, label_id, box_data) VALUES "
        def bbox_str(rect):
            del rect["label_id"]
            return json.dumps(rect)
        rows = [(worksheet_id, rect["label_id"], bbox_str(rect)) for rect in rects]
        stmt += base.bulk_insert_str(c, rows)
        c.execute(stmt)

    @classmethod
    @base.access_point()
    def find_all(cls, worksheet_id, connection=base.INIT_CONNECTION):
        c = connection.cursor()
        stmt = "SELECT * FROM worksheet_boxes JOIN box_labels ON box_labels.id=label_id WHERE worksheet_id=%s"
        c.execute(stmt, (worksheet_id,))
        return [WorksheetBox(*row[:4], label=Label(*row[-2:])) for row in c.fetchall()]

class Label(object):
    def __init__(self, *args):
        base.init_from_row(self, base.column_names(box_label), args)

    def dict(self):
        return dict(id=self.id, name=self.name)

    @classmethod
    @base.access_point()
    def list_all(cls, connection=base.INIT_CONNECTION):
        c = connection.cursor()
        stmt = "SELECT * FROM box_labels"
        c.execute(stmt)
        return [Label(*row) for row in c.fetchall()]

class Worksheet(object):
    def __init__(self, *args):
        base.init_from_row(self, base.column_names(worksheet), args)

    @classmethod
    @base.access_point()
    def get_page(cls, page_no=0, connection=base.INIT_CONNECTION):
        worksheets = cls.list_all(connection=connection, page_no=page_no)
        tot_count =cls.count(connection=connection)
        return worksheets, tot_count

    @classmethod
    @base.access_point()
    def find_by_id(cls, worksheet_id, connection=base.INIT_CONNECTION):
        c = connection.cursor()
        stmt = "SELECT * FROM worksheets WHERE id=%s"
        c.execute(stmt, (worksheet_id,))
        return Worksheet(*c.fetchone())

    @classmethod
    @base.access_point()
    def find_all(cls, assignment_id, connection=base.INIT_CONNECTION):
        stmt = "SELECT * FROM worksheets WHERE assignment_id=%s"
        c = connection.cursor()
        c.execute(stmt, (assignment_id,))
        return [cls(*row) for row in c.fetchall()]

    @classmethod
    @base.access_point()
    def count(cls, connection=base.INIT_CONNECTION):
        stmt = "SELECT COUNT(id) FROM worksheets"
        c = connection.cursor()
        c.execute(stmt)
        return int(c.fetchone()[0])

    @classmethod
    @base.access_point()
    def list_all(cls, n_limit=24, page_no=0, connection=base.INIT_CONNECTION):
        stmt = "SELECT * FROM worksheets ORDER BY id DESC LIMIT %s OFFSET %s"
        c = connection.cursor()
        c.execute(stmt, (n_limit, page_no * n_limit))
        return [cls(*row) for row in c.fetchall()]

    @staticmethod
    @base.access_point()
    def create_all(assignment_id, paths, points_data, connection=base.INIT_CONNECTION):
        stmt = "INSERT INTO worksheets (assignment_id, image_path, points_json) VALUES {}"
        c = connection.cursor()
        append_stmt = base.bulk_insert_str(c, list(zip([assignment_id] * len(paths), paths, points_data)))
        stmt = stmt.format(append_stmt)
        c.execute(stmt)

@base.access_point()
def create_assignment(profile_id, grade_tokens, duration_ms, connection=base.INIT_CONNECTION):
    grades = [float(AssignmentStore.get_result(token)) for token in grade_tokens]
    grade = np.mean(grades)
    assign = Assignment.create(profile_id, grade, duration_ms=duration_ms, connection=connection)
    paths = [WorksheetStore.get_path(token).decode() for token in grade_tokens]
    points_data = [WorksheetStore.get_points_json(token).decode() for token in grade_tokens]
    Worksheet.create_all(assign.id, paths, points_data, connection=connection)
    worksheets = Worksheet.find_all(assign.id, connection=connection)
    return assign, worksheets

class Assignment(object):
    def __init__(self, *args):
        base.init_from_row(self, base.column_names(assignment), args)

    @classmethod
    @base.access_point()
    def find(cls, profile_id, connection=base.INIT_CONNECTION):
        c = connection.cursor()
        stmt = "SELECT * FROM assignments WHERE profile_id=%s ORDER BY creation_time DESC"
        c.execute(stmt, (profile_id,))
        return [cls(*row) for row in c.fetchall()]

    @classmethod
    @base.access_point()
    def find_all(cls, profile_id, limit=20, page_no=0, connection=base.INIT_CONNECTION):
        stmt = "SELECT * FROM assignments WHERE profile_id=%s LIMIT %s OFFSET %s"
        c = connection.cursor()
        c.execute(stmt, (profile_id, limit, page_no * limit))
        return [cls(*row) for row in c.fetchall()]

    @classmethod
    @base.access_point()
    def create(cls, profile_id, grade, duration_ms=0, connection=base.INIT_CONNECTION):
        stmt = "INSERT INTO assignments (profile_id, grade, duration_ms) VALUES (%s, %s, %s) RETURNING id, creation_time"
        c = connection.cursor()
        c.execute(stmt, (profile_id, grade, duration_ms))
        asst_id, creation_time = c.fetchone()
        return cls(asst_id, profile_id, grade, creation_time, duration_ms)

    def dict(self):
        return dict(id=self.id, profile_id=self.profile_id, grade=self.grade, date=self.creation_time.isoformat(), duration_ms=self.duration_ms)

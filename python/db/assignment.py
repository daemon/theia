import numpy as np
from sqlalchemy import *
from sqlalchemy.sql import func
import redis

from db.base import sec_random_gen
import db.base as base

assignment = Table("assignments", base.Base.metadata,
    Column("id", Integer, primary_key=True),
    Column("profile_id", Integer, ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False),
    Column("grade", Float, nullable=False),
    Column("creation_time", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("duration_ms", Integer, nullable=False, server_default="0"))

worksheet = Table("worksheets", base.Base.metadata,
    Column("id", Integer, primary_key=True),
    Column("assignment_id", Integer, ForeignKey("assignments.id", ondelete="CASCADE"), nullable=False),
    Column("image_path", String(4096), nullable=False),
    Column("points_json", String(8192), nullable=False, server_default='""'))

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

class Worksheet(object):
    def __init__(self, *args):
        base.init_from_row(self, base.column_names(worksheet), args)

    @classmethod
    @base.access_point()
    def find_all(cls, assignment_id, connection=base.INIT_CONNECTION):
        stmt = "SELECT * FROM worksheets WHERE assignment_id=%s"
        c = connection.cursor()
        c.execute(stmt, (assignment_id,))
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
def create_assignment(profile_id, grade_tokens, connection=base.INIT_CONNECTION):
    grades = [float(AssignmentStore.get_result(token)) for token in grade_tokens]
    grade = np.mean(grades)
    assign = Assignment.create(profile_id, grade, connection=connection)
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

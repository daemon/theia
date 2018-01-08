import base64
import datetime
import hashlib
import json
import random

from sqlalchemy import *
import redis

from db.base import sec_random_gen
import db.base as base

user = Table("users", base.Base.metadata,
    Column("id", Integer, primary_key=True),
    Column("password", String(44), nullable=False),
    Column("salt", String(16), nullable=False),
    Column("email", String(32), nullable=False, unique=True, index=True))

profile = Table("profiles", base.Base.metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(44), nullable=False, default="Default"),
    Column("owner_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False))

class session_store:
    store = redis.StrictRedis(host="127.0.0.1", port=6379, db=0)
    @staticmethod
    def create_token(user):
        token = sec_random_gen(24)
        session_store.store.set(token, user.id, ex=3600 * 24 * 14)
        return token

    @staticmethod
    def get_user(token):
        user_id = session_store.store.get(token)
        if not user_id:
            return None
        return User.find(id=int(user_id))

    @staticmethod
    def delete_token(token):
        session_store.store.delete(token)

class UserExistsError(Exception):
    pass

class Profile(object):
    def __init__(self, *args):
        base.init_from_row(self, base.column_names(profile), args)

    @classmethod
    @base.access_point()
    def find(cls, **kwargs):
        connection = kwargs["connection"]
        profile_id = kwargs["id"]
        c = connection.cursor()
        stmt = "SELECT * FROM profiles WHERE id=%s"
        c.execute(stmt, (profile_id,))
        return cls(*c.fetchone())

    @classmethod
    @base.access_point()
    def list_profiles(cls, user_id, connection=base.INIT_CONNECTION):
        c = connection.cursor()
        stmt = "SELECT * FROM profiles WHERE owner_id=%s"
        c.execute(stmt, (user_id,))
        return [cls(*row) for row in c.fetchall()]

    @classmethod
    @base.access_point()
    def create(cls, name, owner_id, connection=base.INIT_CONNECTION):
        stmt = "INSERT INTO profiles (name, owner_id) VALUES (%s, %s) RETURNING id"
        c = connection.cursor()
        c.execute(stmt, (name, owner_id))
        return cls(c.fetchone()[0], name, owner_id)

    def dict(self):
        return dict(id=self.id, name=self.name, owner_id=self.owner_id)

class User(object):
    def __init__(self, *args):
        base.init_from_row(self, ["id", "password", "salt", "email"], args)

    def login(self, password):
        if self.password.encode() == sha256x2(password, self.salt):
            return session_store.create_token(self)
        else:
            return None

    def logout(self, token):
        session_store.delete_token(token)

    @staticmethod
    @base.access_point()
    def find(**kwargs):
        connection = kwargs["connection"]
        c = connection.cursor()
        (conditions, params) = base.join_conditions(kwargs, "AND", ["id", "email"])
        stmt = "SELECT * FROM users WHERE " + conditions
        c.execute(stmt, params)
        for row in c.fetchall():
            return User(*row)
        return None

    @classmethod
    @base.access_point()
    def register(cls, email, password, connection=base.INIT_CONNECTION):
        user = cls.create(email, password, connection=connection)
        profile = Profile.create("Default", user.id, connection=connection)
        return user, profile

    @staticmethod
    @base.access_point()
    def create(email, password, connection=base.INIT_CONNECTION):
        salt = sec_random_gen(16)
        stmt = "INSERT INTO users (email, password, salt) VALUES (%s, %s, %s) RETURNING id"
        c = connection.cursor()
        password = sha256x2(password, salt).decode("utf-8")
        try:
          c.execute(stmt, (email, password, salt))
        except:
          raise UserExistsError
        user = User(c.fetchone()[0], password, salt, email)
        return user

def sha256x2(password, salt):
    image1 = ''.join([hashlib.sha256(password.encode()).hexdigest(), salt])
    image2 = base64.b64encode(hashlib.sha256(image1.encode()).digest())
    return image2

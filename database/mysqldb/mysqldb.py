#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/8/17 18:35 
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class MySqlDB:
    def __init__(self, host, username, password, port, database):
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.database = database
        self.url = "mysql+pymysql://{}:{}@{}:{}/{}".format(self.username, self.password,
                                                           self.host, self.port, self.database)
        self.engine = create_engine(self.url, echo=False)
        self.session = sessionmaker(bind=self.engine, expire_on_commit=False)
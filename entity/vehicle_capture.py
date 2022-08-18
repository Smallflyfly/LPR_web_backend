#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/8/17 19:00 
"""
from sqlalchemy import Column, BIGINT, String, Date
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class VehicleCapture(Base):
    __tablename__ = 'vehicle_capture'
    id = Column(BIGINT, primary_key=True, comment='id', autoincrement=True)
    number = Column(String(20))
    vehicle_image_url = Column(String(255))
    lp_url = Column(String(255))
    location = Column(String(255))
    capture_time = Column(Date())
    create_time = Column(Date())
    update_time = Column(Date())

# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import Farm, Field, Device
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/farms", tags=["Multi-Farm Management"])

class FarmCreateInput(BaseModel):
    name: str
    location: str
    owner_id: int = 1

class FieldCreateInput(BaseModel):
    name: str
    area_acres: float
    crop_type: str
    farm_id: int

def init_farms(db: Session):
    if db.query(Farm).count() == 0:
        f1 = Farm(name="North Grid Sector-A", location="Punjab Northern Belt", owner_id=1)
        f2 = Farm(name="Eastern Grain Hectare", location="Sikkim Valley Zone", owner_id=1)
        db.add_all([f1, f2])
        db.commit()
        db.refresh(f1)
        db.refresh(f2)
        
        # Seed fields
        field1 = Field(name="Punjab Wheat Field", area_acres=45.0, crop_type="Wheat", farm_id=f1.id)
        field2 = Field(name=" पंजाब Maize Field", area_acres=20.0, crop_type="Maize", farm_id=f1.id)
        field3 = Field(name="Sikkim Organic Tea", area_acres=15.0, crop_type="Rice", farm_id=f2.id)
        db.add_all([field1, field2, field3])
        db.commit()

@router.get("")
async def get_farms(db: Session = Depends(get_db)):
    init_farms(db)
    farms = db.query(Farm).all()
    res = []
    for f in farms:
        fields = db.query(Field).filter(Field.farm_id == f.id).all()
        res.append({
            "id": f.id,
            "name": f.name,
            "location": f.location,
            "fields": [{"id": fl.id, "name": fl.name, "cropType": fl.crop_type, "area": fl.area_acres} for fl in fields]
        })
    return res

@router.post("")
async def create_farm(payload: FarmCreateInput, db: Session = Depends(get_db)):
    farm = Farm(name=payload.name, location=payload.location, owner_id=payload.owner_id)
    db.add(farm)
    db.commit()
    db.refresh(farm)
    return farm

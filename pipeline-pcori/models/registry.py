#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
REGISTRY = {}

def register(name: str):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco

def create(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name](**kwargs)

def names():
    return sorted(list(REGISTRY.keys()))

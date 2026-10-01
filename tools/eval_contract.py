"""Explicit transport validity and descriptive, non-clinical evaluation statistics."""
from __future__ import annotations
import math


def wilson(successes:int,total:int,z:float=1.959963984540054):
    if type(total) is not int or type(successes) is not int or not 0<=successes<=total or total==0:
        raise ValueError('Wilson interval requires 0 <= successes <= total and total > 0')
    p=successes/total;denominator=1+z*z/total
    centre=(p+z*z/(2*total))/denominator
    delta=z*math.sqrt((p*(1-p)+z*z/(4*total))/total)/denominator
    return [0. if successes==0 else max(0.,centre-delta),1. if successes==total else min(1.,centre+delta)]


def completion_text(data):
    if not isinstance(data,dict) or data.get('error'):raise ValueError('endpoint returned an error or non-object')
    choices=data.get('choices')
    if not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict):raise ValueError('expected exactly one completion choice')
    message=choices[0].get('message')
    text=message.get('content') if isinstance(message,dict) else None
    if not isinstance(text,str) or not text.strip():raise ValueError('completion contains no nonempty text')
    return text


def usable_text(text):
    return isinstance(text,str) and bool(text.strip()) and not text.startswith(('HTTP_','ERROR_'))

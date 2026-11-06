"""Report assembly: one complete report string, plus a filtered console version."""
import os
from datetime import datetime

import config


def section(key,title,body):
    return dict(key=key,title=title,body=body.rstrip())


def render(sections,hidden=()):
    """Join sections into one text; sections whose key is in `hidden` are left out. Numbering is sequential."""
    parts=[]
    number=0
    for s in sections:
        if s["key"] in hidden:
            continue
        number+=1
        head=f"[{number}] {s['title']}"
        parts.append(head+"\n"+"-"*len(head)+"\n"+s["body"])
    return "\n\n".join(parts)+"\n"


def save_report(text,folder=None):
    """Write the full report to a new timestamped .txt file (never overwrites an earlier report)."""
    folder=folder or config.REPORT_DIR
    os.makedirs(folder,exist_ok=True)
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    path=os.path.join(folder,f"{config.REPORT_PREFIX}_{stamp}.txt")
    n=1
    while os.path.exists(path):
        path=os.path.join(folder,f"{config.REPORT_PREFIX}_{stamp}_{n}.txt")
        n+=1
    with open(path,"w",encoding="utf-8") as f:
        f.write(text)
    return path

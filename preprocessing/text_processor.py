import math
import re
from collections import Counter

import config

STOPWORDS=set("""a an the and or of to in on for with by from at as is are was were be been being this that these those
it its into than then so such can could should would may might will shall do does did have has had not no
explain define describe discuss write state list give briefly short note notes what which who whom whose how why
following example examples marks mark question questions answer using use also any each all both either
""".split())

PAGE_NO=re.compile(r"^\s*(?:-\s*)?(?:page\s*)?\d{1,3}(?:\s*(?:of|/)\s*\d{1,3})?(?:\s*-)?\s*$",re.I)
MARK_PATTERNS=[
    re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*(?:marks?|m)?\s*\]",re.I),
    re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*(?:marks?|m)\s*\)",re.I),
    re.compile(r"[-,]?\s*\(?(\d+(?:\.\d+)?)\s*marks?\)?\s*$",re.I),
    re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*\)\s*$"),
]

# Lines that are page furniture or table debris rather than question text.
NOISE_PATTERNS=[
    re.compile(r"^(?:no\.?\s*of\s*pages|page\s*no|course\s*code|roll\s*no|reg(?:ister)?\.?\s*no)\b",re.I),
    re.compile(r"^\(?\s*to be filled in by",re.I),
    re.compile(r"^\W*(?:[A-Z]{1,4}\s*/\s*[A-Z]{1,4}\s*)?/\s*END\s*/\W*$",re.I),   # "/END/", "FD/RL /END/"
    re.compile(r"^[A-Z]{1,4}\s*/\s*[A-Z]{1,4}$"),                                  # "FD/RL"
    re.compile(r"^CO\s*:\s*COURSE OUTCOME",re.I),
    re.compile(r"\bBTL\b\s*:",re.I),
    re.compile(r"^Question\s*(?:\(?\s*(?:BTL|Marks?|CO)\s*\)?\s*)*$",re.I),       # table header "Question (BTL)"
    re.compile(r"^\(?\s*BTL\s*\)?$",re.I),
    re.compile(r"^(?:Qn\.?\s*\d+\s*|CO\s*\.?\s*\d+\.?\s*)+$",re.I),              # "Qn.3 CO 3", "CO1."
    re.compile(r"^[\s.\u2026\d]*(?:\.{2,}|\u2026)[\s.\u2026\d]*$"),                  # "....", "3..3..", ellipsis rows
]
TAG_RE=re.compile(r"\(\s*(?:BTL\s*[-:]?\s*)?[LK]\s*[1-6]\s*\)|\(\s*CO\s*\.?\s*\d\s*\)",re.I)
LABEL_LIKE=re.compile(r"^\s*(?:\(?[a-h]\)|\(?(?:i{1,3}|iv|v|vi{0,3}|ix|x)\)|[\[(]\s*\d+(?:\.\d+)?\s*(?:marks?|m)?\s*[\])])",re.I)

TYPE_RULES=[
    ("Proof",re.compile(r"\b(prove|proof|show that)\b",re.I)),
    ("Derivation",re.compile(r"\b(derive|derivation|obtain an expression|deduce)\b",re.I)),
    ("Comparison",re.compile(r"\b(compare|differentiate|distinguish|contrast|differences? between|versus|vs\.?)\b",re.I)),
    ("Numerical / Problem Solving",re.compile(r"\b(calculate|compute|solve|determine the value|evaluate|numerical|how many|given that|find the)\b",re.I)),
    ("Definition",re.compile(r"^\s*(define|what is|what are|what do you mean|state|list|name|mention|give the definition)\b",re.I)),
    ("Application",re.compile(r"\b(apply|applications?|use cases?|real[- ]world|case study|design a|suggest|implement|scenario)\b",re.I)),
    ("Theory / Explanation",re.compile(r"\b(explain|describe|discuss|elaborate|outline|illustrate|notes? on|briefly|why|how does|how do|working of|with (?:the help of )?(?:a )?diagram)\b",re.I)),
]


def normalize_whitespace(text):
    text=text.replace("\u00a0"," ").replace("\u00ad","")
    text=re.sub(r"[ \t]+"," ",text)
    text="\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}","\n\n",text).strip()


def is_noise_line(line):
    return any(p.search(line.strip()) for p in NOISE_PATTERNS)


def strip_tags(line):
    """Remove Bloom-level / course-outcome tags such as (L4) or (CO2)."""
    return " ".join(TAG_RE.sub(" ",line).split())


def remove_headers_footers(page_texts):
    """Join page texts after dropping page numbers and repeated header/footer lines."""
    pages=[[l.strip() for l in t.split("\n") if l.strip()] for t in page_texts]
    n=len(pages)
    counts=Counter()
    for lines in pages:
        for key in {re.sub(r"\d+","#",l.lower()) for l in lines if len(l)<80}:
            counts[key]+=1
    repeat_min=max(2,math.ceil(0.6*n))
    kept=[]
    for lines in pages:
        for idx,l in enumerate(lines):
            if PAGE_NO.match(l) and (re.search(r"page|of|/|-",l,re.I) or idx in (0,len(lines)-1)):
                continue
            if idx==0 and re.fullmatch(r"\d{3,6}",l):      # paper / booklet code printed at the top of a page
                continue
            if is_noise_line(l):
                continue
            key=re.sub(r"\d+","#",l.lower())
            looks_like_question=re.match(r"^\s*(q\.?\s*)?\d",l,re.I) or LABEL_LIKE.match(l)
            if n>=2 and len(l)<80 and counts[key]>=repeat_min and not looks_like_question:
                continue
            kept.append(l)
        kept.append("")
    return normalize_whitespace("\n".join(kept))


def extract_marks(text):
    """Return (marks or None, text without the marks expression)."""
    for pat in MARK_PATTERNS:
        found=list(pat.finditer(text))
        if found:
            m=found[-1]
            value=float(m.group(1))
            if 0<value<=100:
                cleaned=(text[:m.start()]+" "+text[m.end():]).strip(" .-,")
                return value,normalize_whitespace(cleaned)
    return None,text


def embedding_ready(text):
    """Light cleaning for Sentence-BERT (keeps case and technical symbols)."""
    return " ".join(text.split())


def tfidf_ready(text):
    """Lower-case tokens without stopwords (keeps tokens such as k-means, c4)."""
    tokens=re.findall(r"[a-z0-9][a-z0-9_+#\-]*",text.lower())
    return " ".join(t for t in tokens if t not in STOPWORDS and len(t)>1)


def numeric_flag(text):
    return len(re.findall(r"\d+(?:\.\d+)?",text))>=2


def classify_question_type(text):
    """Rule-based question type (internal; the report only shows aggregates)."""
    for name,pattern in TYPE_RULES:
        if pattern.search(text):
            return name
    if len(re.findall(r"\d+(?:\.\d+)?",text))>=3:
        return "Numerical / Problem Solving"
    return "Other"

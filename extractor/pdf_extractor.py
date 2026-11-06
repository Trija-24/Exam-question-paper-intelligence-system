import os
import re

import fitz  # PyMuPDF

import config
from preprocessing import text_processor as tp

YEAR_RE=re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
MONTH_RE=re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",re.I)
ROMAN=r"(?:i{1,3}|iv|v|vi{1,3}|ix|x)"
# A line may start with any chain of labels: "1.", "a)", "i)", "1. a) i)" ... followed by the text.
LABEL_RE=re.compile(r"^\s*(?:(?i:Q(?:uestion)?)\.?\s*)?(?:(?P<q>\d{1,2})\s*[.):]\s*)?"
                    r"(?:\(?(?P<p>[a-h])\)\s*)?(?:\(?(?P<r>"+ROMAN+r")\)\s*)?(?P<rest>.*)$")
Q_PREFIX=re.compile(r"^\s*(?i:Q(?:uestion)?)\.?\s*(?P<q>\d{1,2})\s*[.):]?\s*(?P<rest>.*)$")
SECTION_RE=re.compile(r"^\s*((?i:section|part))\s*[-:]?\s*([A-E]|[IVX]{1,3}|\d{1,2})\b(.*)$")
INSTR_ITEM=re.compile(r"^(answer|attempt|all questions|each question|every question|subdivisions?|course outcomes?|"
                      r"use of|assume|note\b|candidates|write your|calculators?|read the|time allowed|do not|you may|"
                      r"internal choice|figures|marks|choose|instructions?)",re.I)
INSTR_HEAD=re.compile(r"^\s*instructions?\b",re.I)
INSTR_HINT=re.compile(r"\b(marks?|carr(?:y|ies)|compulsory|attempt|choose|answer|questions?|sections?|parts?|"
                      r"calculators?|permitted|allowed|candidates?)\b",re.I)
INSTRUCTION=re.compile(r"^(answer|attempt|all questions|each question|use of|assume|note\b|candidates|write your|"
                       r"calculators?|read the|time allowed|do not|you may|internal choice|figures)",re.I)
STATED_TOTAL=re.compile(r"(?:max(?:imum)?\.?\s*marks|total\s*marks)\s*[:\-]?\s*(\d+)",re.I)
# marks written at the END of a line: "(4)", "[5]", "(5 marks)", "5 marks"
LINE_MARKS=re.compile(r"(?:^|[\s,.;:-])(?:[\[(]\s*(\d{1,2}(?:\.\d+)?)\s*(?:marks?|m)?\s*[\])]|(\d{1,2}(?:\.\d+)?)\s*marks?\.?)\s*$",re.I)
# "Subdivisions (a)(i) and (a)(ii) carries 2 marks each, subdivision (b) carries 6 marks" -> label groups + marks
RULE_RE=re.compile(r"((?:\(\s*[a-h]\s*\)(?:\s*\(\s*"+ROMAN+r"\s*\))?\s*(?:,|and|&|or)?\s*)+)\s*(?:each\s+)?"
                   r"(?:carr(?:y|ies)|are\s+of|is\s+of|=|:)\s*(\d{1,3}(?:\.\d+)?)\s*marks?",re.I)
RULE_LABEL=re.compile(r"\(\s*([a-h])\s*\)(?:\s*\(\s*("+ROMAN+r")\s*\))?")
PER_QUESTION=re.compile(r"each\s+question\s+carr(?:y|ies)\s+(\d{1,3}(?:\.\d+)?)\s*marks?",re.I)

HEAD_RE=re.compile(r"^\s*(?:(?:unit|module|chapter|section|topic)\s*[-:.]?\s*(?:[ivxIVX]+|\d+)\s*[:.\-\u2013\u2014]?\s*(.*)"
                   r"|(\d+)\s*[.):\-\u2013]\s+(.*))$",re.I)
SKIP_HEAD=re.compile(r"^(text\s*books?|references?|reference books?|course (objectives?|outcomes?)|prerequisites?|"
                     r"total hours|credits?|evaluation|assessment|syllabus|marking scheme)\b",re.I)


def extract_pdf(path):
    """Read a PDF safely. Returns a dict; info['ok'] is True when text was found."""
    info=dict(path=path,name=os.path.basename(path),pages=0,page_texts=[],metadata={},
              issues=[],ok=False,text="")
    try:
        doc=fitz.open(path)
    except Exception as exc:
        info["issues"].append(f"Could not open PDF ({str(exc)[:80]})")
        return info
    try:
        if doc.needs_pass:
            info["issues"].append("PDF is password protected")
            return info
        info["pages"]=doc.page_count
        info["metadata"]=doc.metadata or {}
        for page in doc:
            try:
                info["page_texts"].append(page.get_text("text"))
            except Exception:
                info["page_texts"].append("")
    except Exception as exc:
        info["issues"].append(f"Error while reading pages ({str(exc)[:80]})")
    finally:
        doc.close()

    if info["pages"]==0:
        info["issues"].append("PDF has no pages")
        return info
    empty=sum(1 for t in info["page_texts"] if not t.strip())
    if empty==info["pages"]:
        info["issues"].append("No extractable text (likely scanned/image-only; OCR is not implemented)")
        return info
    if empty:
        info["issues"].append(f"{empty} page(s) had no extractable text (possibly scanned)")
    info["text"]=tp.remove_headers_footers(info["page_texts"])
    info["ok"]=bool(info["text"].strip())
    return info


def _single_year(text):
    years=set(YEAR_RE.findall(text))
    return int(years.pop()) if len(years)==1 else None


def detect_year(info):
    """Try filename, then the examination line (month + year), then first-page text, then PDF metadata."""
    year=_single_year(os.path.splitext(info["name"])[0])
    if year:
        return year,"filename"
    first=info["page_texts"][0][:1500] if info["page_texts"] else ""
    exam_lines=[l for l in first.split("\n") if MONTH_RE.search(l) or re.search(r"exam",l,re.I)]
    year=_single_year(" ".join(exam_lines))
    if year:
        return year,"PDF text"
    year=_single_year(first[:1000])
    if year:
        return year,"PDF text"
    meta=" ".join(str(info["metadata"].get(k,"") or "") for k in ("title","subject"))
    year=_single_year(meta)
    if year:
        return year,"PDF metadata"
    return None,None


def stated_total_marks(text):
    m=STATED_TOTAL.search(text)
    return float(m.group(1)) if m else None


def parse_marks_rules(text):
    """Read statements such as 'subdivision (b) carries 6 marks each' from the instructions.

    Returns (rules, per_question): rules maps (part, sub-part or None) -> marks.
    """
    rules={}
    for m in RULE_RE.finditer(text):
        for part,sub in RULE_LABEL.findall(m.group(1)):
            rules[(part.lower(),sub.lower() or None)]=float(m.group(2))
    pq=PER_QUESTION.search(text)
    return rules,(float(pq.group(1)) if pq else None)


def _new_unit(q,part,sub,section):
    return dict(q=q,part=part,sub=sub,section=section,lines=[],marks=None)


def _line_marks(line):
    """Split a trailing marks expression off a line -> (line without marks, marks or None)."""
    m=LINE_MARKS.search(line)
    if not m:
        return line,None
    value=float(m.group(1) or m.group(2))
    if not 0<value<=100:
        return line,None
    return line[:m.start()].rstrip(" ,.;:-"),value


def _resolve_marks(units,rules,per_question):
    """Fill missing marks: explicit marks > rule for the (part, sub-part) > share of the part's total > section default."""
    by_q={}
    for u in units:
        by_q.setdefault(u["q"],[]).append(u)
    for group in by_q.values():
        parts={}
        for u in group:
            parts.setdefault(u["part"],[]).append(u)
        for part,pu in parts.items():
            for u in pu:
                if u["marks"] is None and part is not None and (part,u["sub"]) in rules:
                    u["marks"]=rules[(part,u["sub"])]
            missing=[u for u in pu if u["marks"] is None]
            if missing and part is not None and (part,None) in rules:
                remaining=rules[(part,None)]-sum(u["marks"] for u in pu if u["marks"] is not None)
                if remaining>0:
                    for u in missing:
                        u["marks"]=remaining/len(missing)
            for u in pu:
                if u["marks"] is None:
                    u["marks"]=u.get("default_marks")
        if len(group)==1 and group[0]["marks"] is None:
            group[0]["marks"]=per_question


def parse_questions(text):
    """Split question-paper text into question units.

    Handles labels such as '1.', 'Q2', '3(a)', 'b) i)' whether they share a line with the text or stand alone,
    instructions, page headers, Bloom-level tags, and marks given inline or by a rule in the instructions.
    A part such as 'a) i) ... ii) ...' becomes one unit per sub-part; a part whose roman items follow a
    question stem ('c) <scenario> i) ... ii) ...') stays one unit.
    """
    section="Main"
    default_marks=None
    units=[]
    stems={}
    pre=[]                      # lines before the first question (instructions)
    cur=None
    last_q=None
    split={}                    # (q, part) -> True when the part is divided into roman sub-parts
    in_instructions=False
    instr_q=0

    def close():
        nonlocal cur
        if cur is not None and cur["lines"]:
            units.append(cur)
        cur=None

    for raw in text.split("\n"):
        line=tp.strip_tags(raw.strip())
        if not line or line.upper().strip(" .") in ("OR","OR.") or tp.is_noise_line(line):
            continue
        sec=SECTION_RE.match(line) if len(line)<70 else None
        if sec:
            close()
            section=f"{sec.group(1).title()} {sec.group(2).upper()}"
            each=re.search(r"(\d+(?:\.\d+)?)\s*marks?\s*each",sec.group(3),re.I)
            default_marks=float(each.group(1)) if each else None
            last_q=None
            continue
        if last_q is None and cur is None and INSTR_HEAD.match(line):
            in_instructions=True
            pre.append(line)
            continue

        full_line=line
        line,mk=_line_marks(line)
        lm=LABEL_RE.match(line)
        q=lm.group("q") and int(lm.group("q"))
        part=lm.group("p")
        rom=lm.group("r")
        rest=lm.group("rest").strip()
        has_label=bool(q or part or rom)
        if q is not None and not (last_q is None or q>last_q):
            has_label=False                      # a number that is not the next question: ordinary text
            q=part=rom=None
        if not q and not part and Q_PREFIX.match(line):          # "Question 3" with the number only
            qm=Q_PREFIX.match(line)
            q=int(qm.group("q"))
            rest=qm.group("rest").strip()
            has_label=bool(q) and (last_q is None or q>last_q)

        before_questions=last_q is None and cur is None
        if before_questions:
            real_start=has_label and bool(part or q)
            if real_start and q and not part and (INSTR_ITEM.match(rest) or
                                                  (in_instructions and q==instr_q+1 and INSTR_HINT.search(rest))):
                pre.append(full_line)            # a numbered instruction such as "2. Subdivision (b) carries 6 marks"
                instr_q=q
                continue
            if not real_start:
                pre.append(full_line)            # header text, course-outcome table, instruction continuation
                continue
            in_instructions=False

        if not has_label:
            if cur is not None:
                cur["lines"].append(line)
                if mk is not None:
                    cur["marks"]=(cur["marks"] or 0)+mk
            continue

        if q:                                    # a new question
            close()
            last_q=q
            cur=_new_unit(q,None,None,section)
            cur["default_marks"]=default_marks
            if part:
                cur=_new_unit(q,part,rom,section)
                cur["default_marks"]=default_marks
                cur["lines"]=[rest] if rest else []
                if rom:
                    split[(q,part)]=True
            elif rest:
                cur["lines"].append(rest)
        elif part:                               # a new part of the current question
            qn=cur["q"] if cur is not None else last_q
            if qn is None:                       # parts with no question number printed: assume question 1
                qn=last_q=1
            if cur is not None and cur["part"] is None and cur["lines"]:
                stems[qn]=" ".join(cur["lines"]).strip()
                cur=None                         # the stem belongs to the parts, it is not a question by itself
            close()
            cur=_new_unit(qn,part,rom,section)
            cur["default_marks"]=default_marks
            cur["lines"]=[rest] if rest else []
            if rom:
                split[(qn,part)]=True
        else:                                    # a roman label on its own: new sub-part or ordinary text
            if cur is not None and cur["part"] is not None and (split.get((cur["q"],cur["part"])) or not cur["lines"]):
                qn,pt=cur["q"],cur["part"]
                split[(qn,pt)]=True
                if cur["lines"]:
                    close()
                    cur=_new_unit(qn,pt,rom,section)
                    cur["default_marks"]=default_marks
                else:
                    cur["sub"]=rom
                cur["lines"]=[rest] if rest else []
            elif cur is not None:
                cur["lines"].append(line)
        if mk is not None and cur is not None:
            cur["marks"]=(cur["marks"] or 0)+mk
    close()

    rules,per_question=parse_marks_rules(" ".join(pre))
    _resolve_marks(units,rules,per_question)

    out=[]
    for u in units:
        body=" ".join(u["lines"]).strip()
        marks=u["marks"]
        if marks is None:
            marks,body=tp.extract_marks(body)
            if marks is None:
                marks=u.get("default_marks")
        stem=stems.get(u["q"],"")
        if u["part"] and stem and len(stem)<=200:
            body=stem+" "+body
        qid=f"{u['q']}{u['part'] or ''}"+(f"({u['sub']})" if u["sub"] else "")
        out.append(dict(qid=qid,section=u["section"],text=body.strip(),marks=marks,is_sub=u["part"] is not None))
    clean=[]
    for q in out:
        if len(q["text"].split())<3:
            continue
        if INSTRUCTION.match(q["text"]) and q["marks"] is None:
            continue
        clean.append(q)
    return clean


# ----------------------------- syllabus parsing -----------------------------

CAPS_HEAD=re.compile(r"^\s*([A-Za-z][A-Za-z0-9 &/,'\-]{2,90}?)\s*:\s*(.*)$")
HOURS_END=re.compile(r"\(\s*\d{1,3}\s*(?:hrs?|hours?|periods?)?\s*\)\s*[.,;]?\s*$",re.I)
HOURS_ANY=re.compile(r"\(\s*\d{1,3}\s*(?:hrs?|hours?|periods?)?\s*\)",re.I)
SMALL_WORDS={"and","of","on","in","for","the","to","a","an","vs","or","with","by","at"}
GENERIC_CONCEPTS={"introduction","overview","basics","conclusion","etc","others","and others","fundamentals"}
CONCEPT_SPLIT=re.compile(r"[,;\u2022\u00b7\u25cf\u25aa\u25cb|]|\s[-\u2013\u2014]+\s|\s[-\u2013\u2014]+(?=\S)|"
                         r"(?<=\S)[-\u2013\u2014]+\s|:\s*|(?<=[a-z])-(?=[A-Z])")


def _is_caps(text):
    letters=[c for c in text if c.isalpha()]
    return len(letters)>=4 and sum(c.isupper() for c in letters)/len(letters)>=0.8 and len(text.split())<=10


def _display_name(name):
    """ALL-CAPS headings are shown in title case (short acronyms stay upper case)."""
    name=" ".join(name.split()).strip(" .-:")
    if not _is_caps(name):
        return name
    words=[]
    for i,w in enumerate(name.split()):
        low=w.lower()
        if i>0 and low in SMALL_WORDS:
            words.append(low)
        elif len(w)<=3 and low not in SMALL_WORDS:
            words.append(w)
        else:
            words.append(w.capitalize())
    return " ".join(words)


def _split_concepts(text):
    """Split a run of syllabus text into short concept phrases."""
    text=HOURS_ANY.sub(" ",text)
    extra=[]

    def grab(m):
        inner=re.sub(r"^(?:case study|e\.?g\.?|i\.?e\.?|such as|like|including)\s*[:.,-]?\s*","",m.group(1).strip(),flags=re.I)
        if re.fullmatch(r"[A-Za-z0-9]{1,6}",inner) or not re.search(r"[A-Za-z]",inner):
            return " "                                  # an acronym or a number, not a concept
        extra.append(inner)
        return " "

    text=re.sub(r"\(([^()]*)\)",grab,text).replace("(","").replace(")","")
    out=[]
    for part in CONCEPT_SPLIT.split(text)+[x for e in extra for x in CONCEPT_SPLIT.split(e)]:
        p=re.sub(r"^\d+(?:\.\d+)*[.)]?\s*","",part.strip(" .-\u2013\u2014\u2022*\t,;"))
        p=" ".join(re.sub(r"^(?:and|&)\s+","",p,flags=re.I).strip(" .-").split())
        words=p.split()
        if not (1<=len(words)<=config.CONCEPT_MAX_WORDS and 3<=len(p)<=70 and re.search(r"[A-Za-z]",p)):
            continue
        if p.lower() in GENERIC_CONCEPTS or all(w.lower() in tp.STOPWORDS for w in words):
            continue
        out.append(p)
    return out


def _split_title(title):
    for sep in (":"," - "," \u2013 "):
        if sep in title:
            name,rest=title.split(sep,1)
            return name.strip(),rest.strip()
    if title.count(",")>=3:
        name,rest=title.split(",",1)
        return name.strip(),rest.strip()
    return title.strip(),""


def _structured_topics(lines):
    """'Unit 1: Title', 'Module 2 - Title', '3. Title' headings followed by concept lists."""
    topics=[]
    cur=None
    pending=False
    for line in lines:
        if pending and cur is not None:
            cur["name"]=line[:80]
            pending=False
            continue
        m=HEAD_RE.match(line)
        if m:
            title=((m.group(1) if m.group(1) is not None else m.group(3)) or "").strip()
            if SKIP_HEAD.match(title):
                cur=None
                continue
            name,rest=_split_title(title)
            cur=dict(name=name[:80],concepts=[])
            topics.append(cur)
            pending=not name
            if rest:
                cur["concepts"]+=_split_concepts(rest)
            continue
        if cur is not None:
            cur["concepts"]+=_split_concepts(line)
    return topics


def _paragraph_topics(lines):
    """'TITLE IN CAPS: concepts ... (hours)' blocks; a block without a title is named after its first concept.

    A block ends at its hours marker, e.g. '(6)', so an untitled block that follows is still recognised.
    """
    paras=[]
    cur=None
    for line in lines:
        head=CAPS_HEAD.match(line)
        if head and _is_caps(head.group(1)) and not SKIP_HEAD.match(head.group(1)):
            cur=dict(name=head.group(1),text=head.group(2),closed=False)
            paras.append(cur)
        elif cur is None or cur["closed"]:
            if not paras and len(line.split())<=8 and not re.search(r"[-,;:()]",line):
                continue                                # course code / title line before the first unit
            cur=dict(name=None,text=line,closed=False)
            paras.append(cur)
        else:
            cur["text"]+=" "+line
        if HOURS_END.search(line):
            cur["closed"]=True
    topics=[]
    for p in paras:
        concepts=_split_concepts(p["text"])
        name=_display_name(p["name"]) if p["name"] else None
        if name is None:
            if not concepts:
                continue
            name=concepts.pop(0)
        topics.append(dict(name=name[:80],concepts=concepts))
    return topics


def _drop_generic_single_words(topics):
    """A one-word concept that is also a word inside several other syllabus phrases is too generic to match on."""
    def stem(w):
        return re.sub(r"s$","",w.lower())
    counts={}
    for t in topics:
        for phrase in [t["name"]]+t["concepts"]:
            for w in set(stem(x) for x in re.findall(r"[A-Za-z]+",phrase)):
                counts[w]=counts.get(w,0)+1
    for t in topics:
        t["concepts"]=[c for c in t["concepts"] if len(c.split())>1 or counts.get(stem(c),0)<4]


def parse_syllabus_topics(text):
    """Detect topics and their concept phrases from the syllabus text (any of the supported layouts)."""
    lines=[l.strip() for l in text.split("\n") if re.search(r"[A-Za-z]",l)]
    topics=_structured_topics(lines)
    if len([t for t in topics if t["name"]])<2:
        topics=_paragraph_topics(lines)
    if len([t for t in topics if t["name"]])<2:
        # no recognisable structure: treat every reasonably short line as a topic
        topics=[]
        for line in lines:
            if 3<=len(line)<=200 and re.search(r"[A-Za-z]{3}",line):
                name,rest=_split_title(line)
                topics.append(dict(name=name[:80],concepts=_split_concepts(rest)))
    merged={}
    for t in topics:
        if not t["name"]:
            continue
        key=t["name"].lower()
        entry=merged.setdefault(key,dict(name=t["name"],concepts=[]))
        for c in t["concepts"]:
            if c.lower()!=key and c.lower() not in [x.lower() for x in entry["concepts"]]:
                entry["concepts"].append(c)
    result=list(merged.values())
    _drop_generic_single_words(result)
    return result

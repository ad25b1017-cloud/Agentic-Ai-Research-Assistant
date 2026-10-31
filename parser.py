"""PDF -> section-aware chunks with paper name + page number."""
import re
from dataclasses import dataclass
import fitz  # PyMuPDF
from config import CHUNK_CHARS

SECTIONS = {
    "abstract": r"abstract",
    "introduction": r"introduction",
    "related_work": r"related work|background|preliminaries",
    "methodology": r"method(s|ology)?|approach|proposed (method|model)|model|framework|architecture",
    "experiments": r"experiment(s|al)?( setup| settings)?|evaluation|implementation details|datasets?",
    "results": r"results?( and discussion)?|analysis|discussion|ablation(s| study)?",
    "limitations": r"limitations?|future work|threats to validity",
    "conclusion": r"conclusions?|concluding remarks",
    "references": r"references|bibliography",
}
CAPTION = re.compile(r"^(table|fig\.?|figure)\s*\d+", re.I)

@dataclass
class Chunk:
    id: int
    paper: str
    section: str
    page: int
    text: str

def _heading(text: str):
    t = text.strip().replace("\n", " ")
    if len(t) > 60 or len(t) < 3:
        return None
    t = re.sub(r"^([IVX]+\.|\d+(\.\d+)*\.?)\s*", "", t).strip().lower().rstrip(":")
    for name, pat in SECTIONS.items():
        if re.fullmatch(pat, t):
            return name
    return None

def parse_pdf(path: str, paper_name: str, start_id: int = 0):
    doc = fitz.open(path)
    section, blocks, held_caption = "front_matter", [], ""
    for pno, page in enumerate(doc, start=1):
        for b in sorted(page.get_text("blocks"), key=lambda b: (round(b[0] / 300), b[1])):
            if b[6] != 0:  # skip image blocks
                continue
            text = re.sub(r"-\n(\w)", r"\1", b[4]).replace("\n", " ").strip()
            if not text:
                continue
            h = _heading(text)
            if h:
                section = h
                continue
            if section == "references":
                continue
            if CAPTION.match(text):           # keep captions glued to their table/figure body
                held_caption = text
                continue
            if held_caption:
                text, held_caption = held_caption + " | " + text, ""
            if len(text) >= 40:
                blocks.append((section, pno, text))
    # merge paragraphs of the same section into ~CHUNK_CHARS chunks
    chunks, cur = [], None
    for sec, pno, text in blocks:
        if cur and cur[0] == sec and len(cur[2]) + len(text) < CHUNK_CHARS:
            cur[2] += " " + text
        else:
            if cur: chunks.append(cur)
            cur = [sec, pno, text]
    if cur: chunks.append(cur)
    return [Chunk(start_id + i, paper_name, s, p, t) for i, (s, p, t) in enumerate(chunks)]

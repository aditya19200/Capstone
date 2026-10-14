"""
services/pdf_extract.py — turn an uploaded PDF into documents the pipeline
can classify.

What this does and does not do
------------------------------
It extracts text and cuts it into classifiable units. It does not classify,
store, or normalise beyond what extraction requires — the caller runs the
text through utils.text.normalize() and hands it to the same batch pipeline
a CSV upload uses. A PDF is an input format, not a second pipeline.

Why chunking, instead of one document per page or per file
----------------------------------------------------------
The model truncates at settings.MAX_LENGTH (256) tokens. Anything past that
in a single document is silently discarded — it is never seen, never
explained, and never appears in the training data this system exists to
produce. A 20-page judgment submitted as one document would be classified on
its first page and a half.

Page boundaries are no better: they fall wherever the typesetter put them,
routinely mid-sentence.

So text is packed into chunks that fit the model's window, splitting at the
largest boundary that works: paragraph, then sentence, then word.

Worth knowing what the input actually looks like, because it decides which
of those does the work. pdfplumber's text layer is line-broken with single
newlines and, on every file tested here, contains no blank lines at all —
so paragraph splitting usually finds nothing to split on and the packing is
sentence-level in practice. The paragraph pass is kept because PDFs
exported from word processors do carry blank lines, and when they do, those
are the right boundaries to use.

Measured on datasets/v1.csv with the real tokeniser, legal English runs 4.26
characters per token at the 10th percentile, so 256 tokens is about 1090
characters for 90% of text; the default budget below is rounded down from
that. Callers that have the tokeniser to hand should pass token_counter
instead and get exact packing.

Scanned PDFs
------------
pdfplumber reads the text layer. A scanned judgment is a photograph of a
page with no text layer, and extraction returns empty strings rather than
failing. Ingesting that would queue a batch of blank documents, classify
them confidently as something, and quietly poison the annotation set — so
it is detected and refused. OCR would be the fix; that is not in scope here,
and claiming OCR we do not have would be worse than refusing.
"""

import io
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# Rounded down from the measured 10th-percentile 4.26 chars/token at
# MAX_LENGTH=256. Used only when the caller passes no token_counter.
DEFAULT_MAX_CHARS = 1000

# Below this many characters a chunk is not worth classifying — it is a
# running header, a page number, or a stray line. Mirrors settings.
# MIN_TEXT_CHARS, which exists for the same reason.
MIN_CHUNK_CHARS = 50

# A PDF averaging fewer than this many characters per page has no usable
# text layer. A genuine text PDF runs in the thousands; a scanned one
# returns 0, give or take a stray watermark or a page-number overlay.
MIN_CHARS_PER_PAGE = 50

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
# Sentence end followed by whitespace — used only to break up a paragraph
# that is on its own too long to fit the window.
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


class PdfExtractionError(ValueError):
    """Base for PDF problems that should become a 4xx, not a 500."""


class EncryptedPdfError(PdfExtractionError):
    """The file is password-protected."""


class ScannedPdfError(PdfExtractionError):
    """The file has no text layer — almost certainly scanned images."""


@dataclass
class PdfExtractResult:
    documents: List[str] = field(default_factory=list)
    page_count: int = 0
    pages_with_text: int = 0
    chars_extracted: int = 0


def extract_pages(raw: bytes) -> List[str]:
    """
    Return one string per page. Pages with no text layer come back empty.

    Raises EncryptedPdfError for password-protected files and
    PdfExtractionError for anything pdfplumber cannot open.
    """
    import pdfplumber                       # heavy import — kept out of startup
    from pdfminer.pdfdocument import PDFPasswordIncorrect

    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            return [(page.extract_text() or "") for page in pdf.pages]
    except PDFPasswordIncorrect as exc:
        raise EncryptedPdfError(
            "This PDF is password-protected. Remove the password and upload it again."
        ) from exc
    except PdfExtractionError:
        raise
    except Exception as exc:
        # pdfminer raises a wide and poorly-typed set of errors on malformed
        # files; the caller only needs to know the file was unreadable.
        raise PdfExtractionError(f"Could not read this PDF: {exc}") from exc


def _split_oversized(paragraph: str, fits: Callable[[str], bool]) -> List[str]:
    """
    Break a single paragraph that does not fit into sentence-sized pieces.

    Falls back to a hard character split only if one 'sentence' still does
    not fit, which happens with tables and unpunctuated blocks.
    """
    out: List[str] = []
    buf = ""
    for sentence in _SENTENCE_END.split(paragraph):
        candidate = f"{buf} {sentence}".strip() if buf else sentence
        if fits(candidate) or not buf:
            buf = candidate
        else:
            out.append(buf)
            buf = sentence
    if buf:
        out.append(buf)

    # Anything still too long has no sentence boundaries to use — tables,
    # unpunctuated headings, numbered schedules. Fall back to packing whole
    # words, which is the smallest unit worth keeping intact and, unlike
    # halving the string, cannot emit a piece that was never checked
    # against `fits`.
    final: List[str] = []
    for piece in out:
        if fits(piece):
            final.append(piece)
            continue
        buf = ""
        for word in piece.split():
            candidate = f"{buf} {word}".strip() if buf else word
            if fits(candidate) or not buf:
                buf = candidate
            else:
                final.append(buf)
                buf = word
        if buf:
            final.append(buf)
    return final


def split_documents(
    pages: List[str],
    max_chars: int = DEFAULT_MAX_CHARS,
    token_counter: Optional[Callable[[str], int]] = None,
    max_tokens: Optional[int] = None,
) -> List[str]:
    """
    Pack page text into documents that fit the model's window.

    Pages are joined with a single newline, so a sentence running across a
    page boundary stays intact. Paragraphs are then accumulated until the
    next one would not fit; a paragraph too long on its own is broken on
    sentence boundaries, and failing that on word boundaries.

    If token_counter is given, fitting is decided by real token count and
    max_chars is ignored. Otherwise the character budget stands in for it.
    """
    if token_counter is not None and max_tokens is not None:
        def fits(text: str) -> bool:
            return token_counter(text) <= max_tokens
    else:
        def fits(text: str) -> bool:
            return len(text) <= max_chars

    # Single newline, not a blank line: a page break is not a paragraph
    # break. Judgments routinely run a sentence across one — "...bench of
    # Orissa" / "High Court consisting of..." — and joining with a blank
    # line would insert a paragraph boundary into the middle of it.
    joined = "\n".join(p for p in pages if p.strip())
    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(joined) if p.strip()]

    documents: List[str] = []
    buf = ""
    for para in paragraphs:
        if not fits(para):
            if buf:
                documents.append(buf)
                buf = ""
            documents.extend(_split_oversized(para, fits))
            continue

        candidate = f"{buf}\n\n{para}" if buf else para
        if fits(candidate):
            buf = candidate
        else:
            documents.append(buf)
            buf = para
    if buf:
        documents.append(buf)

    return [d for d in (doc.strip() for doc in documents) if len(d) >= MIN_CHUNK_CHARS]


def extract(
    raw: bytes,
    max_chars: int = DEFAULT_MAX_CHARS,
    token_counter: Optional[Callable[[str], int]] = None,
    max_tokens: Optional[int] = None,
) -> PdfExtractResult:
    """
    Full extraction: bytes in, classifiable documents out.

    Raises ScannedPdfError when the file has no usable text layer, so the
    caller can say so instead of queueing blank documents.
    """
    pages = extract_pages(raw)
    if not pages:
        raise PdfExtractionError("This PDF has no pages.")

    chars = sum(len(p.strip()) for p in pages)
    with_text = sum(1 for p in pages if p.strip())

    if chars < MIN_CHARS_PER_PAGE * len(pages):
        raise ScannedPdfError(
            f"No text could be extracted from this PDF "
            f"({chars} characters across {len(pages)} page(s)). It is most "
            f"likely a scan rather than a digital document. Reading scanned "
            f"pages needs OCR, which this system does not do — upload a "
            f"digital PDF, or paste the text in directly."
        )

    documents = split_documents(
        pages, max_chars=max_chars, token_counter=token_counter, max_tokens=max_tokens
    )
    if not documents:
        raise PdfExtractionError(
            f"This PDF had text ({chars} characters) but none of it formed a "
            f"passage of at least {MIN_CHUNK_CHARS} characters."
        )

    logger.info(
        "[pdf_extract] %d page(s), %d with text, %d chars -> %d document(s)",
        len(pages), with_text, chars, len(documents),
    )
    return PdfExtractResult(
        documents=documents,
        page_count=len(pages),
        pages_with_text=with_text,
        chars_extracted=chars,
    )

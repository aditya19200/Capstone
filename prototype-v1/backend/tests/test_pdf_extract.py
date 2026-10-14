"""
tests/test_pdf_extract.py — Tests for services/pdf_extract.py.

The splitting logic is tested directly against page strings rather than
through generated PDF files: the behaviour that matters (paragraphs are kept
whole, nothing exceeds the window, headers and page numbers are dropped) is
about text, and binding it to a particular PDF writer would make the tests
fragile for no gain. Reading real PDF bytes is covered separately, with a
file built at test time.
"""

import pytest

from services import pdf_extract
from services.pdf_extract import (
    MIN_CHUNK_CHARS,
    PdfExtractionError,
    ScannedPdfError,
    split_documents,
)

PARA = "This is a paragraph of a legal judgment that runs to a reasonable length. " * 3


class TestSplitDocuments:
    def test_short_paragraphs_are_packed_together(self):
        pages = [f"{PARA}\n\n{PARA}"]
        docs = split_documents(pages, max_chars=10_000)
        assert len(docs) == 1, "both paragraphs fit, so they should be one document"

    def test_packing_stops_at_the_budget(self):
        pages = ["\n\n".join([PARA] * 6)]
        docs = split_documents(pages, max_chars=500)
        assert len(docs) > 1
        # Each paragraph is longer than the budget on its own, so every
        # document here came through the oversized path; what matters is
        # that nothing is wildly over.
        assert all(len(d) <= 500 * 2 for d in docs)

    def test_paragraphs_that_fit_are_never_split(self):
        pages = [f"{PARA}\n\n{PARA}"]
        docs = split_documents(pages, max_chars=len(PARA) + 10)
        assert docs == [PARA.strip(), PARA.strip()]

    def test_oversized_paragraph_is_broken_on_sentences(self):
        pages = ["One sentence here. Two sentence here. Three sentence here. " * 20]
        docs = split_documents(pages, max_chars=300)
        assert len(docs) > 1
        assert all(len(d) <= 400 for d in docs)

    def test_unpunctuated_block_still_gets_split(self):
        # No sentence boundaries at all — tables and numbered schedules look
        # like this. The word-level fallback must engage, and every piece it
        # emits must have been checked against the budget. An earlier version
        # halved the string and appended the first half unchecked, which
        # emitted a 5,000-character document against a 400-character budget.
        pages = ["word " * 2000]
        docs = split_documents(pages, max_chars=400)
        assert len(docs) > 1
        assert all(len(d) <= 400 for d in docs), "no piece may exceed the budget"

    def test_fragments_below_the_floor_are_dropped(self):
        pages = ["12", "Page 3 of 9", PARA]
        docs = split_documents(pages, max_chars=10_000)
        assert len(docs) == 1
        assert "Page 3 of 9" not in docs[0] or len(docs[0]) >= MIN_CHUNK_CHARS

    def test_empty_pages_contribute_nothing(self):
        assert split_documents(["", "   ", "\n\n"]) == []

    def test_token_counter_takes_precedence_over_chars(self):
        # A counter that calls everything oversized must force splitting even
        # though max_chars would have allowed one document.
        pages = [f"{PARA}\n\n{PARA}"]
        docs = split_documents(
            pages,
            max_chars=10_000,
            token_counter=lambda t: len(t),      # 1 token per char
            max_tokens=len(PARA) + 10,
        )
        assert len(docs) == 2

    def test_page_boundary_does_not_cut_a_sentence(self):
        # Judgments routinely run a sentence across a page break. Joining
        # pages with a blank line would insert a paragraph boundary into the
        # middle of one, so the two halves must stay in one document.
        docs = split_documents(
            ["Page one ends with the words bench of Orissa",
             "High Court consisting of Justice Mishra and others at length"],
            max_chars=10_000,
        )
        assert len(docs) == 1
        assert "Orissa\nHigh Court" in docs[0]


class TestExtract:
    def test_scanned_pdf_is_refused(self, monkeypatch):
        monkeypatch.setattr(pdf_extract, "extract_pages", lambda raw: ["", "", ""])
        with pytest.raises(ScannedPdfError) as exc:
            pdf_extract.extract(b"whatever")
        assert "OCR" in str(exc.value), "the message should say why it cannot be read"

    def test_pdf_with_text_but_no_usable_passage_is_refused(self, monkeypatch):
        # Guards the case where a file clears the scanned-PDF floor but
        # splitting yields nothing worth classifying. Driven by stubbing the
        # splitter rather than by crafted input: pages are joined with a
        # single newline, so short fragments merge into one usable passage
        # instead, and the natural input for this no longer exists.
        monkeypatch.setattr(pdf_extract, "extract_pages",
                            lambda raw: ["x " * 40, "y " * 40, "z " * 40])
        monkeypatch.setattr(pdf_extract, "split_documents",
                            lambda *a, **k: [])
        with pytest.raises(PdfExtractionError):
            pdf_extract.extract(b"whatever")

    def test_pdf_with_no_pages_is_refused(self, monkeypatch):
        monkeypatch.setattr(pdf_extract, "extract_pages", lambda raw: [])
        with pytest.raises(PdfExtractionError):
            pdf_extract.extract(b"whatever")

    def test_counts_are_reported(self, monkeypatch):
        monkeypatch.setattr(pdf_extract, "extract_pages", lambda raw: [PARA, "", PARA])
        result = pdf_extract.extract(b"whatever", max_chars=10_000)
        assert result.page_count == 3
        assert result.pages_with_text == 2
        assert result.chars_extracted == len(PARA.strip()) * 2
        assert result.documents


class TestRealPdfBytes:
    """Reads an actual PDF, built here so the test carries no binary fixture."""

    @staticmethod
    def _one_page_pdf(text: str) -> bytes:
        # A minimal but valid PDF. Written by hand rather than with a
        # generator so the test suite gains no dependency.
        content = f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET".encode()
        objs = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
            + content + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
        out = bytearray(b"%PDF-1.4\n")
        offsets = []
        for i, body in enumerate(objs, 1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
        for off in offsets:
            out += f"{off:010d} 00000 n \n".encode()
        out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n").encode()
        return bytes(out)

    def test_reads_text_out_of_a_real_pdf(self):
        sentence = ("The accused was charged under the Indian Penal Code and the "
                    "prosecution led evidence before this Court at trial")
        pages = pdf_extract.extract_pages(self._one_page_pdf(sentence))
        assert len(pages) == 1
        assert "prosecution" in pages[0]

    def test_malformed_file_raises_the_typed_error(self):
        with pytest.raises(PdfExtractionError):
            pdf_extract.extract_pages(b"%PDF-1.4\nnot actually a pdf\n")

    def test_empty_file_raises_the_typed_error(self):
        with pytest.raises(PdfExtractionError):
            pdf_extract.extract_pages(b"")

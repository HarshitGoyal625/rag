"""PDF document reader using PyMuPDF.

The reader converts PDF layout information into the common
DocumentStructure IR.

PDFs do not reliably expose semantic headings, so this reader uses
conservative layout-aware heuristics:

    PDF spans
        ↓
    visual line reconstruction
        ↓
    heading detection
        ↓
    paragraph reconstruction
        ↓
    DocumentStructure blocks

Important:
    Bold text alone is NOT considered a heading.

A heading is identified using a combination of:
    - larger font size
    - bold typography
    - short text
    - vertical isolation

Normal body text is reconstructed into paragraph blocks.
"""

from __future__ import annotations
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .base import DocumentReader
from .structure import Block, BlockType, DocumentStructure


class PdfReader(DocumentReader):
    """Reader for PDF files using PyMuPDF."""

    def read(self, file_path: Path) -> DocumentStructure:
        """Read a PDF and return a structured DocumentStructure."""

        import pymupdf

        with pymupdf.open(file_path) as doc:
            lines = self._extract_lines(doc)
            blocks = self._build_blocks(lines)
            metadata = self._extract_metadata(doc)

        return DocumentStructure(
            source=str(file_path),
            doc_type="pdf",
            blocks=blocks,
            metadata=metadata,
        )

    # ============================================================
    # METADATA
    # ============================================================

    @staticmethod
    def _extract_metadata(doc) -> dict:
        """Extract non-empty PDF metadata."""

        output: dict = {}

        metadata = doc.metadata or {}

        for key, value in metadata.items():
            if value not in (None, "", []):
                output[key] = value

        return output

    # ============================================================
    # EXTRACTION PIPELINE
    # ============================================================

    def _extract_lines(self, doc) -> list[dict[str, Any]]:
        """Extract and reconstruct visual lines from the PDF."""

        all_lines: list[dict[str, Any]] = []

        for page_num, page in enumerate(doc, start=1):

            page_dict = page.get_text("dict")

            page_lines: list[dict[str, Any]] = []

            for pdf_block in page_dict.get("blocks", []):

                # 0 = text block
                if pdf_block.get("type") != 0:
                    continue

                for pdf_line in pdf_block.get("lines", []):

                    line = self._build_line(
                        pdf_line,
                        page_num,
                    )

                    if line is not None:
                        page_lines.append(line)

            # ----------------------------------------------------
            # Merge PDF line fragments that are actually on the
            # same visual line.
            #
            # This is important because PDFs can represent pieces
            # of one visual line as separate text objects.
            # ----------------------------------------------------

            page_lines = self._merge_visual_line_fragments(
                page_lines
            )

            # ----------------------------------------------------
            # Calculate spacing between lines.
            # ----------------------------------------------------

            self._calculate_line_spacing(
                page_lines
            )

            all_lines.extend(page_lines)

        return all_lines

    # ============================================================
    # BUILD VISUAL LINE
    # ============================================================

    @staticmethod
    def _build_line(
        pdf_line: dict[str, Any],
        page_num: int,
    ) -> dict[str, Any] | None:
        """Convert a PyMuPDF line into our internal representation."""

        spans = pdf_line.get("spans", [])

        spans = [
            span
            for span in spans
            if span.get("text", "").strip()
        ]

        if not spans:
            return None

        # --------------------------------------------------------
        # Text
        # --------------------------------------------------------

        text = " ".join(
            span["text"].strip()
            for span in spans
            if span["text"].strip()
        ).strip()

        if not text:
            return None

        # --------------------------------------------------------
        # Font sizes
        # --------------------------------------------------------

        font_sizes = [
            round(
                float(span.get("size", 0)),
                2,
            )
            for span in spans
            if float(span.get("size", 0)) > 0
        ]

        if not font_sizes:
            return None

        # Most common font size.
        font_size = Counter(
            font_sizes
        ).most_common(1)[0][0]

        # --------------------------------------------------------
        # Boldness
        #
        # A line is considered bold only when >=80% of its spans
        # are bold.
        #
        # This prevents:
        #
        #   "Founder & CEO: Amit Bansal"
        #
        # from becoming a heading simply because the name/role
        # is bold.
        # --------------------------------------------------------

        bold_flags = [
            PdfReader._is_bold(
                int(span.get("flags", 0))
            )
            for span in spans
        ]

        bold_ratio = (
            sum(bold_flags) / len(bold_flags)
            if bold_flags
            else 0.0
        )

        mostly_bold = bold_ratio >= 0.80

        # --------------------------------------------------------
        # Font
        # --------------------------------------------------------

        fonts = [
            span.get("font", "")
            for span in spans
            if span.get("font")
        ]

        font = (
            Counter(fonts).most_common(1)[0][0]
            if fonts
            else ""
        )

        # --------------------------------------------------------
        # Bounding box
        # --------------------------------------------------------

        bboxes = [
            span.get(
                "bbox",
                [0, 0, 0, 0],
            )
            for span in spans
        ]

        x0 = min(
            bbox[0]
            for bbox in bboxes
        )

        y0 = min(
            bbox[1]
            for bbox in bboxes
        )

        x1 = max(
            bbox[2]
            for bbox in bboxes
        )

        y1 = max(
            bbox[3]
            for bbox in bboxes
        )

        return {
            "text": text,
            "font_size": font_size,
            "font": font,
            "bold": mostly_bold,
            "bold_ratio": bold_ratio,
            "bbox": [x0, y0, x1, y1],
            "page": page_num,

            "spacing_before": 0.0,
            "spacing_after": 0.0,

            "heading_level": None,
            "block_type": None,
        }

    # ============================================================
    # VISUAL LINE FRAGMENT MERGING
    # ============================================================

    @staticmethod
    def _merge_visual_line_fragments(
        lines: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge separate PDF objects occupying the same visual line."""

        if not lines:
            return []

        lines = sorted(
            lines,
            key=lambda line: (
                line["bbox"][1],
                line["bbox"][0],
            ),
        )

        merged: list[dict[str, Any]] = []

        for line in lines:

            if not merged:
                merged.append(line)
                continue

            previous = merged[-1]

            # ----------------------------------------------------
            # Same vertical position?
            # ----------------------------------------------------

            previous_y0 = previous["bbox"][1]
            previous_y1 = previous["bbox"][3]

            current_y0 = line["bbox"][1]
            current_y1 = line["bbox"][3]

            vertical_overlap = min(
                previous_y1,
                current_y1,
            ) - max(
                previous_y0,
                current_y0,
            )

            previous_height = (
                previous_y1 - previous_y0
            )

            current_height = (
                current_y1 - current_y0
            )

            min_height = min(
                previous_height,
                current_height,
            )

            same_visual_line = (
                vertical_overlap
                >= min_height * 0.60
            )

            # ----------------------------------------------------
            # Close horizontally?
            # ----------------------------------------------------

            horizontal_gap = (
                line["bbox"][0]
                - previous["bbox"][2]
            )

            close_horizontally = (
                -2.0
                <= horizontal_gap
                <= 8.0
            )

            # ----------------------------------------------------
            # Same font size?
            # ----------------------------------------------------

            same_font_size = abs(
                previous["font_size"]
                - line["font_size"]
            ) < 0.15

            if (
                same_visual_line
                and close_horizontally
                and same_font_size
            ):

                previous["text"] = (
                    previous["text"]
                    + " "
                    + line["text"]
                ).strip()

                previous["bbox"] = [
                    min(
                        previous["bbox"][0],
                        line["bbox"][0],
                    ),
                    min(
                        previous["bbox"][1],
                        line["bbox"][1],
                    ),
                    max(
                        previous["bbox"][2],
                        line["bbox"][2],
                    ),
                    max(
                        previous["bbox"][3],
                        line["bbox"][3],
                    ),
                ]

                # Recalculate boldness.
                previous["bold"] = (
                    previous["bold"]
                    and line["bold"]
                )

                continue

            merged.append(line)

        return merged

    # ============================================================
    # SPACING
    # ============================================================

    @staticmethod
    def _calculate_line_spacing(
        lines: list[dict[str, Any]],
    ) -> None:
        """Calculate vertical spacing between consecutive lines."""

        for index, line in enumerate(lines):

            if index == 0:
                line["spacing_before"] = 999.0
            else:

                previous = lines[index - 1]

                if previous["page"] != line["page"]:
                    line["spacing_before"] = 999.0

                else:
                    line["spacing_before"] = max(
                        0.0,
                        line["bbox"][1]
                        - previous["bbox"][3],
                    )

            if index == len(lines) - 1:
                line["spacing_after"] = 999.0
            else:

                next_line = lines[index + 1]

                if next_line["page"] != line["page"]:
                    line["spacing_after"] = 999.0

                else:
                    line["spacing_after"] = max(
                        0.0,
                        next_line["bbox"][1]
                        - line["bbox"][3],
                    )

    # ============================================================
    # BODY FONT DETECTION
    # ============================================================

    @staticmethod
    def _detect_body_font_size(
        lines: list[dict[str, Any]],
    ) -> float:
        """Detect the dominant body font size."""

        counts: Counter = Counter()

        for line in lines:

            size = round(
                line["font_size"],
                1,
            )

            # Weight by text length.
            weight = min(
                max(len(line["text"]), 1),
                100,
            )

            counts[size] += weight

        if not counts:
            return 12.0

        return float(
            counts.most_common(1)[0][0]
        )

    # ============================================================
    # HEADING DETECTION
    # ============================================================

    def _detect_headings(
        self,
        lines: list[dict[str, Any]],
    ) -> None:
        """Mark high-confidence headings."""

        body_size = self._detect_body_font_size(
            lines
        )

        # --------------------------------------------------------
        # Find larger font sizes.
        # --------------------------------------------------------

        larger_sizes = sorted(
            {
                round(
                    line["font_size"],
                    1,
                )
                for line in lines
                if line["font_size"]
                > body_size * 1.10
            },
            reverse=True,
        )

        for line in lines:

            text = line["text"].strip()

            if not text:
                continue

            size = round(
                line["font_size"],
                1,
            )

            # ----------------------------------------------------
            # Short text only.
            # ----------------------------------------------------

            if len(text) > 100:
                continue

            # ----------------------------------------------------
            # Heading candidates must be larger than body text.
            # ----------------------------------------------------

            if size <= body_size * 1.10:
                continue

            # ----------------------------------------------------
            # Require bold typography for normal section headings.
            #
            # Exception:
            # very large title-sized text can be accepted without
            # the bold flag.
            # ----------------------------------------------------

            size_ratio = (
                line["font_size"]
                / body_size
            )

            if not line["bold"]:

                if size_ratio < 1.40:
                    continue

            # ----------------------------------------------------
            # Isolation.
            #
            # Heading should have noticeably more whitespace
            # around it than ordinary body lines.
            # ----------------------------------------------------

            before = line[
                "spacing_before"
            ]

            after = line[
                "spacing_after"
            ]

            # Estimate normal body line spacing.
            normal_gaps = [
                other["spacing_before"]
                for other in lines
                if (
                    other["spacing_before"] > 0
                    and other["spacing_before"] < 20
                )
            ]

            if normal_gaps:

                normal_gap = statistics.median(
                    normal_gaps
                )

            else:
                normal_gap = 6.0

            large_gap = max(
                normal_gap * 1.6,
                normal_gap + 2.0,
            )

            isolated_before = (
                before >= large_gap
            )

            isolated_after = (
                after >= large_gap
            )

            # Must be isolated on at least one side.
            if not (
                isolated_before
                or isolated_after
            ):
                continue

            # ----------------------------------------------------
            # Avoid sentence-like headings.
            # ----------------------------------------------------

            if self._looks_like_sentence(text):
                if not (
                    isolated_before
                    and isolated_after
                ):
                    continue

            # ----------------------------------------------------
            # Determine heading level from font hierarchy.
            # ----------------------------------------------------

            if not larger_sizes:
                continue

            try:
                level = (
                    larger_sizes.index(size)
                    + 1
                )
            except ValueError:
                continue

            # Keep hierarchy manageable.
            level = min(level, 3)

            line["heading_level"] = level
            line["block_type"] = (
                BlockType.HEADING
            )

    # ============================================================
    # PARAGRAPH RECONSTRUCTION
    # ============================================================

    def _build_blocks(
        self,
        lines: list[dict[str, Any]],
    ) -> list[Block]:
        """Convert visual lines into semantic blocks."""

        if not lines:
            return []

        # First detect headings.
        self._detect_headings(lines)

        blocks: list[Block] = []

        current_paragraph: list[
            dict[str, Any]
        ] = []

        def flush_paragraph() -> None:

            if not current_paragraph:
                return

            text = self._join_paragraph_lines(
                current_paragraph
            )

            if text.strip():

                first = current_paragraph[0]
                last = current_paragraph[-1]

                blocks.append(
                    Block(
                        type=BlockType.PARAGRAPH,
                        text=text,
                        page=first["page"],
                        extra={
                            "font_size": first[
                                "font_size"
                            ],
                            "bbox": [
                                first["bbox"][0],
                                first["bbox"][1],
                                last["bbox"][2],
                                last["bbox"][3],
                            ],
                        },
                    )
                )

            current_paragraph.clear()

        for line in lines:

            # ----------------------------------------------------
            # Heading
            # ----------------------------------------------------

            if line["heading_level"] is not None:

                flush_paragraph()

                blocks.append(
                    Block(
                        type=BlockType.HEADING,
                        text=line["text"],
                        level=line["heading_level"],
                        page=line["page"],
                        extra={
                            "font_size": line[
                                "font_size"
                            ],
                            "font": line["font"],
                            "bold": line["bold"],
                            "bbox": line["bbox"],
                        },
                    )
                )

                continue

            # ----------------------------------------------------
            # Page number
            # ----------------------------------------------------

            if self._is_page_number(line):
                flush_paragraph()
                continue

            # ----------------------------------------------------
            # Table-like content
            #
            # For now we preserve the visual lines as paragraphs.
            # We will handle proper PDF tables separately once
            # normal paragraph extraction is stable.
            # ----------------------------------------------------

            if self._looks_like_table_content(
                line
            ):
                flush_paragraph()

                blocks.append(
                    Block(
                        type=BlockType.PARAGRAPH,
                        text=line["text"],
                        page=line["page"],
                        extra={
                            "font_size": line[
                                "font_size"
                            ],
                            "bbox": line["bbox"],
                        },
                    )
                )

                continue

            # ----------------------------------------------------
            # Normal paragraph
            # ----------------------------------------------------

            if not current_paragraph:
                current_paragraph.append(line)
                continue

            previous = current_paragraph[-1]

            if self._continues_paragraph(
                previous,
                line,
            ):
                current_paragraph.append(line)

            else:
                flush_paragraph()
                current_paragraph.append(line)

        flush_paragraph()

        return blocks

    # ============================================================
    # PARAGRAPH CONTINUATION
    # ============================================================

    @staticmethod
    def _continues_paragraph(
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> bool:
        """Determine whether current line continues previous text."""

        # Different pages = never.
        if previous["page"] != current["page"]:
            return False

        # A heading always breaks the paragraph.
        if current["heading_level"] is not None:
            return False

        # --------------------------------------------------------
        # Vertical gap.
        #
        # In the Aventro PDF:
        #
        # normal line gap ≈ 2 pt
        # paragraph gap ≈ 6-8 pt
        #
        # We use a conservative threshold.
        # --------------------------------------------------------

        gap = (
            current["bbox"][1]
            - previous["bbox"][3]
        )

        if gap > 8.0:
            return False

        # --------------------------------------------------------
        # Horizontal structure.
        #
        # Body paragraphs normally begin at roughly the same x.
        # --------------------------------------------------------

        previous_x = previous["bbox"][0]
        current_x = current["bbox"][0]

        x_difference = abs(
            current_x - previous_x
        )

        # If indentation changes substantially, treat it as a
        # new block/list item.
        if x_difference > 12.0:

            # Exception for wrapped lines that are indented.
            if current_x > previous_x:
                return True

            return False

        # --------------------------------------------------------
        # If previous line ends like a sentence, a small gap can
        # still mean the next line belongs to the same paragraph.
        # --------------------------------------------------------

        return True

    @staticmethod
    def _join_paragraph_lines(
        lines: list[dict[str, Any]],
    ) -> str:
        """Join PDF lines into a natural paragraph."""

        if not lines:
            return ""

        result = ""

        for index, line in enumerate(lines):

            text = line["text"].strip()

            if not text:
                continue

            if not result:
                result = text
                continue

            # ----------------------------------------------------
            # Hyphenated line break.
            #
            # Example:
            #
            # alterna-
            # tive
            #
            # becomes:
            #
            # alternative
            # ----------------------------------------------------

            if result.endswith("-"):

                result = (
                    result[:-1]
                    + text
                )

            else:

                result += " " + text

        return result.strip()

    # ============================================================
    # TABLE DETECTION
    # ============================================================

    @staticmethod
    def _looks_like_table_content(
        line: dict[str, Any],
    ) -> bool:
        """Conservative table-content detection.

        We deliberately don't convert arbitrary PDF text into TABLE
        blocks yet. This prevents normal paragraphs from being
        incorrectly classified.
        """

        text = line["text"].strip()

        # Year-column style values are a useful signal for this PDF,
        # but don't classify them as TABLE blocks automatically.
        if text in {
            "Year",
            "Milestone",
            "Annual",
            "Employees",
            "Notable",
        }:
            return True

        return False

    # ============================================================
    # PAGE NUMBER
    # ============================================================

    @staticmethod
    def _is_page_number(
        line: dict[str, Any],
    ) -> bool:
        """Detect simple footer page numbers."""

        text = line["text"].strip()

        if not text.isdigit():
            return False

        # Page numbers are normally near the bottom.
        y0 = line["bbox"][1]

        return y0 > 650

    # ============================================================
    # TEXT HEURISTICS
    # ============================================================

    @staticmethod
    def _looks_like_sentence(
        text: str,
    ) -> bool:
        """Check whether text looks like ordinary prose."""

        text = text.strip()

        if not text:
            return False

        if text.endswith(
            (".", "?", "!", ";")
        ):
            return True

        if len(text.split()) > 18:
            return True

        return False

    # ============================================================
    # BOLD
    # ============================================================

    @staticmethod
    def _is_bold(flags: int) -> bool:
        """Check PyMuPDF bold font flag."""

        # PyMuPDF bit 4 represents bold.
        return bool(flags & 16)

    # ============================================================
    # EXTENSIONS
    # ============================================================

    def supported_extensions(self) -> list[str]:
        """Return supported PDF extensions."""

        return [".pdf"]
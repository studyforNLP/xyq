from __future__ import annotations

import datetime as dt
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper"
WORD_DIR = PAPER_DIR / "word"
FIGURE_DIR = PAPER_DIR / "figures"

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"

CONTENT_WIDTH_DXA = 9026
EMU_PER_INCH = 914400


FIGURES = {
    1: "fig1_rql_framework.svg",
    2: "fig2_solution_quality.svg",
    3: "fig3_runtime_tradeoff.svg",
    4: "fig4_convergence.svg",
}


def xml_text(text: str) -> str:
    return escape(text, {'"': "&quot;"})


def tag(name: str) -> str:
    return f"w:{name}"


def run_xml(text: str, bold: bool = False, italic: bool = False, code: bool = False) -> str:
    props: list[str] = []
    if bold:
        props.append("<w:b/>")
    if italic:
        props.append("<w:i/>")
    if code:
        props.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="Consolas"/>')
        props.append("<w:color w:val=\"444444\"/>")
    props_xml = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    space = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
    return f"<w:r>{props_xml}<w:t{space}>{xml_text(text)}</w:t></w:r>"


def inline_runs(text: str) -> str:
    pattern = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")
    parts: list[str] = []
    pos = 0
    for match in pattern.finditer(text):
        if match.start() > pos:
            parts.append(run_xml(text[pos : match.start()]))
        token = match.group(0)
        if token.startswith("**"):
            parts.append(run_xml(token[2:-2], bold=True))
        elif token.startswith("`"):
            parts.append(run_xml(token[1:-1], code=True))
        pos = match.end()
    if pos < len(text):
        parts.append(run_xml(text[pos:]))
    return "".join(parts)


def paragraph_xml(
    text: str = "",
    style: str | None = None,
    align: str | None = None,
    before: int | None = None,
    after: int | None = None,
    first_line: int | None = None,
    page_break_before: bool = False,
    runs: str | None = None,
    num_id: int | None = None,
) -> str:
    ppr: list[str] = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if num_id is not None:
        ppr.append(
            f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{num_id}"/></w:numPr>'
        )
    if page_break_before:
        ppr.append("<w:pageBreakBefore/>")
    if before is not None or after is not None:
        ppr.append(
            f'<w:spacing w:before="{before or 0}" w:after="{after or 0}" w:line="360" w:lineRule="auto"/>'
        )
    if first_line is not None:
        ppr.append(f'<w:ind w:firstLine="{first_line}"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    ppr_xml = f"<w:pPr>{''.join(ppr)}</w:pPr>" if ppr else ""
    body_runs = runs if runs is not None else inline_runs(text)
    return f"<w:p>{ppr_xml}{body_runs}</w:p>"


def code_paragraph_xml(text: str) -> str:
    ppr = (
        '<w:pPr><w:pStyle w:val="CodeBlock"/>'
        '<w:shd w:val="clear" w:color="auto" w:fill="F5F5F5"/>'
        '<w:spacing w:before="80" w:after="80"/>'
        '<w:ind w:left="240" w:right="240"/></w:pPr>'
    )
    return f"{'<w:p>'}{ppr}{run_xml(text, code=True)}</w:p>"


def table_xml(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    col_count = max(len(row) for row in rows)
    col_width = max(900, CONTENT_WIDTH_DXA // col_count)
    grid_cols = "".join(f'<w:gridCol w:w="{col_width}"/>' for _ in range(col_count))
    tbl_pr = (
        '<w:tblPr><w:tblW w:w="9026" w:type="dxa"/>'
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="D9D9D9"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="D9D9D9"/>'
        '</w:tblBorders>'
        '<w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:left w:w="120" w:type="dxa"/>'
        '<w:bottom w:w="80" w:type="dxa"/><w:right w:w="120" w:type="dxa"/></w:tblCellMar>'
        "</w:tblPr>"
    )
    row_xml: list[str] = []
    for row_index, row in enumerate(rows):
        cells: list[str] = []
        for col_index in range(col_count):
            value = row[col_index] if col_index < len(row) else ""
            shading = '<w:shd w:val="clear" w:fill="DDEBF7"/>' if row_index == 0 else ""
            cell_pr = f'<w:tcPr><w:tcW w:w="{col_width}" w:type="dxa"/>{shading}</w:tcPr>'
            para = paragraph_xml(value, style="TableText", runs=inline_runs(value))
            cells.append(f"<w:tc>{cell_pr}{para}</w:tc>")
        row_xml.append(f"<w:tr>{''.join(cells)}</w:tr>")
    return f"<w:tbl>{tbl_pr}<w:tblGrid>{grid_cols}</w:tblGrid>{''.join(row_xml)}</w:tbl>"


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        raw = lines[i].strip()
        cells = [cell.strip() for cell in raw.strip("|").split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
            rows.append(cells)
        i += 1
    return rows, i


def svg_size(path: Path) -> tuple[int, int]:
    root = ET.parse(path).getroot()
    width = root.attrib.get("width", "1000")
    height = root.attrib.get("height", "600")
    def parse_number(value: str) -> float:
        match = re.search(r"[\d.]+", value)
        return float(match.group(0)) if match else 1000.0
    return int(parse_number(width)), int(parse_number(height))


def image_xml(rid: str, image_path: Path, doc_pr_id: int, title: str) -> str:
    src_w, src_h = svg_size(image_path)
    width_emu = int(6.1 * EMU_PER_INCH)
    height_emu = int(width_emu * src_h / max(src_w, 1))
    drawing = f"""
<w:drawing>
  <wp:inline distT="0" distB="0" distL="0" distR="0">
    <wp:extent cx="{width_emu}" cy="{height_emu}"/>
    <wp:effectExtent l="0" t="0" r="0" b="0"/>
    <wp:docPr id="{doc_pr_id}" name="{xml_text(title)}" descr="{xml_text(title)}"/>
    <wp:cNvGraphicFramePr>
      <a:graphicFrameLocks noChangeAspect="1"/>
    </wp:cNvGraphicFramePr>
    <a:graphic>
      <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
        <pic:pic>
          <pic:nvPicPr>
            <pic:cNvPr id="{doc_pr_id}" name="{xml_text(image_path.name)}"/>
            <pic:cNvPicPr/>
          </pic:nvPicPr>
          <pic:blipFill>
            <a:blip r:embed="{rid}"/>
            <a:stretch><a:fillRect/></a:stretch>
          </pic:blipFill>
          <pic:spPr>
            <a:xfrm><a:off x="0" y="0"/><a:ext cx="{width_emu}" cy="{height_emu}"/></a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
          </pic:spPr>
        </pic:pic>
      </a:graphicData>
    </a:graphic>
  </wp:inline>
</w:drawing>
""".strip()
    return paragraph_xml(align="center", runs=f"<w:r>{drawing}</w:r>", before=240, after=120)


def markdown_to_body(markdown: str) -> tuple[str, dict[int, Path]]:
    lines = markdown.splitlines()
    body: list[str] = []
    image_refs: dict[int, Path] = {}
    num_counter = 10
    in_figure_captions = False
    i = 0
    first_heading = True
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            body.append(code_paragraph_xml("\n".join(code_lines)))
            continue

        if stripped.startswith("|"):
            rows, next_i = parse_table(lines, i)
            body.append(table_xml(rows))
            i = next_i
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            if level == 1 and first_heading:
                body.append(paragraph_xml(text, style="Title", align="center", before=120, after=360))
                first_heading = False
            else:
                style = "Heading1" if level == 2 else "Heading2" if level == 3 else "Heading3"
                body.append(paragraph_xml(text, style=style, before=260, after=160))
            in_figure_captions = text.lower().startswith("figure captions") or text.startswith("图注")
            i += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            num_counter += 1
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                item = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                body.append(paragraph_xml(item, num_id=num_counter, before=40, after=40))
                i += 1
            continue

        paragraph_lines = [stripped]
        i += 1
        while i < len(lines):
            candidate = lines[i].rstrip()
            cand_stripped = candidate.strip()
            if (
                not cand_stripped
                or cand_stripped.startswith("#")
                or cand_stripped.startswith("|")
                or cand_stripped.startswith("```")
                or re.match(r"^\d+\.\s+", cand_stripped)
            ):
                break
            paragraph_lines.append(cand_stripped)
            i += 1
        text = " ".join(paragraph_lines)
        if in_figure_captions:
            fig_match = re.search(r"(?:Figure|图)\s*(\d+)", text)
            if fig_match:
                fig_no = int(fig_match.group(1))
                fig_path = FIGURE_DIR / FIGURES.get(fig_no, "")
                if fig_path.exists():
                    image_refs[fig_no] = fig_path
                    body.append(f"__IMAGE_PLACEHOLDER_{fig_no}__")
        is_caption = text.startswith("**Figure") or text.startswith("**图")
        body.append(paragraph_xml(text, style="Caption" if is_caption else None, first_line=None if is_caption else 420, before=80, after=80))

    return "\n".join(body), image_refs


def numbering_xml(max_num_id: int = 200) -> str:
    nums = []
    for num_id in range(1, max_num_id + 1):
        nums.append(f'<w:num w:numId="{num_id}"><w:abstractNumId w:val="0"/></w:num>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="{NS_W}">
  <w:abstractNum w:abstractNumId="0">
    <w:multiLevelType w:val="singleLevel"/>
    <w:lvl w:ilvl="0">
      <w:start w:val="1"/>
      <w:numFmt w:val="decimal"/>
      <w:lvlText w:val="%1."/>
      <w:lvlJc w:val="left"/>
      <w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr>
    </w:lvl>
  </w:abstractNum>
  {''.join(nums)}
</w:numbering>"""


def styles_xml(lang: str) -> str:
    east_asia = "SimSun" if lang == "zh" else "Times New Roman"
    body_font = "Times New Roman"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{NS_W}">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="120" w:line="360" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="{body_font}" w:hAnsi="{body_font}" w:eastAsia="{east_asia}"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="240" w:after="360"/><w:jc w:val="center"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="{body_font}" w:hAnsi="{body_font}" w:eastAsia="{east_asia}"/><w:b/><w:sz w:val="36"/><w:szCs w:val="36"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="Heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="300" w:after="180"/><w:outlineLvl w:val="0"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="{body_font}" w:hAnsi="{body_font}" w:eastAsia="{east_asia}"/><w:b/><w:sz w:val="30"/><w:szCs w:val="30"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="Heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="220" w:after="140"/><w:outlineLvl w:val="1"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="{body_font}" w:hAnsi="{body_font}" w:eastAsia="{east_asia}"/><w:b/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="Heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="180" w:after="100"/><w:outlineLvl w:val="2"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="{body_font}" w:hAnsi="{body_font}" w:eastAsia="{east_asia}"/><w:b/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Caption">
    <w:name w:val="Caption"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="80" w:after="160"/></w:pPr>
    <w:rPr><w:i/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="TableText">
    <w:name w:val="Table Text"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="0" w:after="0" w:line="300" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="CodeBlock">
    <w:name w:val="Code Block"/>
    <w:basedOn w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="Consolas"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>
  </w:style>
</w:styles>"""


def document_xml(body: str, image_refs: dict[int, Path]) -> tuple[str, str, list[tuple[str, Path]]]:
    relationships: list[tuple[str, Path]] = []
    body_with_images = body
    for idx, (fig_no, path) in enumerate(sorted(image_refs.items()), start=1):
        rid = f"rId{idx}"
        relationships.append((rid, path))
        body_with_images = body_with_images.replace(
            f"__IMAGE_PLACEHOLDER_{fig_no}__",
            image_xml(rid, path, idx, f"Figure {fig_no}"),
        )
    doc = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{NS_W}" xmlns:r="{NS_R}" xmlns:wp="{NS_WP}" xmlns:a="{NS_A}" xmlns:pic="{NS_PIC}">
  <w:body>
    {body_with_images}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
      <w:cols w:space="720"/>
      <w:docGrid w:linePitch="360"/>
    </w:sectPr>
  </w:body>
</w:document>"""
    rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        '<Relationship Id="rStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
        '<Relationship Id="rNumbering" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>',
        '<Relationship Id="rSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>',
    ]
    for rid, path in relationships:
        rels.append(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{path.name}"/>'
        )
    rels.append("</Relationships>")
    return doc, "\n".join(rels), relationships


def content_types_xml(relationships: list[tuple[str, Path]]) -> str:
    svg_default = '<Default Extension="svg" ContentType="image/svg+xml"/>' if relationships else ""
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {svg_default}
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def settings_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{NS_W}">
  <w:zoom w:percent="100"/>
  <w:defaultTabStop w:val="720"/>
  <w:compat/>
</w:settings>"""


def core_xml(title: str) -> str:
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{xml_text(title)}</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""


def app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>"""


def first_title(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "Manuscript"


def create_docx(markdown_path: Path, output_path: Path, lang: str) -> None:
    markdown = markdown_path.read_text(encoding="utf-8")
    title = first_title(markdown)
    body, image_refs = markdown_to_body(markdown)
    doc, document_rels, relationships = document_xml(body, image_refs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml(relationships))
        zf.writestr("_rels/.rels", root_rels_xml())
        zf.writestr("docProps/core.xml", core_xml(title))
        zf.writestr("docProps/app.xml", app_xml())
        zf.writestr("word/document.xml", doc)
        zf.writestr("word/styles.xml", styles_xml(lang))
        zf.writestr("word/numbering.xml", numbering_xml())
        zf.writestr("word/settings.xml", settings_xml())
        zf.writestr("word/_rels/document.xml.rels", document_rels)
        for _, image_path in relationships:
            zf.write(image_path, f"word/media/{image_path.name}")


def validate_basic_docx(path: Path) -> None:
    required = {
        "[Content_Types].xml",
        "_rels/.rels",
        "word/document.xml",
        "word/styles.xml",
        "word/numbering.xml",
        "word/settings.xml",
        "word/_rels/document.xml.rels",
    }
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        missing = required - names
        if missing:
            raise RuntimeError(f"{path} missing parts: {sorted(missing)}")
        for name in names:
            if name.endswith(".xml") or name.endswith(".rels"):
                ET.fromstring(zf.read(name))


def main() -> None:
    outputs = [
        (PAPER_DIR / "manuscript_revised_en.md", WORD_DIR / "RQL_manuscript_EN.docx", "en"),
        (PAPER_DIR / "manuscript_zh.md", WORD_DIR / "RQL_manuscript_ZH.docx", "zh"),
    ]
    for markdown_path, output_path, lang in outputs:
        create_docx(markdown_path, output_path, lang)
        validate_basic_docx(output_path)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

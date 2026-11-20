"""Finish the native Keynote export without changing historical artifacts.

Only stdlib OOXML edits are used. Text, tables, charts and diagrams remain native.
This is the local fallback for this host, where Artifact Tool is unavailable.
"""
from __future__ import annotations

import copy
import argparse
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as E
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "tmp/presentations/update_20261120"
FINAL = ROOT / "output/pptx/mi_progress_update_20261120.pptx"
PDF = ROOT / "output/pdf/mi_progress_update_20261120.pdf"
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
}
for prefix, uri in NS.items():
    E.register_namespace(prefix, uri)


def tag(name):
    prefix, local = name.split(":")
    return "{" + NS[prefix] + "}" + local


def add(parent, qualified_name, **attrs):
    return E.SubElement(parent, tag(qualified_name), {k: str(v) for k, v in attrs.items()})


def xml(root):
    return E.tostring(root, encoding="utf-8", xml_declaration=True)


def pt(value):
    return str(round(value * 12700))


def fill(parent, color):
    for node in list(parent):
        if node.tag.split("}")[-1] in {"solidFill", "noFill", "gradFill", "pattFill", "blipFill", "grpFill"}:
            parent.remove(node)
    solid = add(parent, "a:solidFill")
    add(solid, "a:srgbClr", val=color)
    if parent.tag.split("}")[-1] in {"rPr", "defRPr", "endParaRPr"}:
        fonts = [n for n in parent if n.tag.split("}")[-1] in {"latin", "ea", "cs", "sym"}]
        if fonts:
            parent.remove(solid)
            parent.insert(list(parent).index(fonts[0]), solid)


def position(shape, x, y, w, h):
    xf = shape.find("p:spPr/a:xfrm", NS)
    xf.find("a:off", NS).attrib.update(x=pt(x), y=pt(y))
    xf.find("a:ext", NS).attrib.update(cx=pt(w), cy=pt(h))
    body = shape.find("p:txBody/a:bodyPr", NS)
    body.attrib.update(lIns="0", rIns="0", tIns="0", bIns="0", anchor="t", wrap="square")
    for child in list(body):
        if child.tag.split("}")[-1] in {"spAutoFit", "normAutofit", "noAutofit"}:
            body.remove(child)
    add(body, "a:noAutofit")


def text_shape(tree, template, text, x, y, w, h, size=22, bold=False, center=False, box=False):
    sp = copy.deepcopy(template)
    ids = [int(node.attrib["id"]) for node in tree.findall(".//p:cNvPr", NS)]
    nv = sp.find("p:nvSpPr/p:cNvPr", NS)
    nv.attrib.update(id=str(max(ids + [1]) + 1), name=text.replace("\n", " ")[:60])
    position(sp, x, y, w, h)
    sppr = sp.find("p:spPr", NS)
    for node in list(sppr):
        if node.tag.split("}")[-1] in {"ln", "solidFill", "noFill", "effectLst", "extLst"}:
            sppr.remove(node)
    if box:
        fill(sppr, "F3F3F3")
        ln = add(sppr, "a:ln", w="12700")
        fill(ln, "505050")
        add(ln, "a:prstDash", val="solid")
    else:
        add(sppr, "a:noFill")
        add(add(sppr, "a:ln"), "a:noFill")
    body = sp.find("p:txBody", NS)
    for node in body.findall("a:p", NS):
        body.remove(node)
    for line in text.split("\n"):
        paragraph = add(body, "a:p")
        pp = add(paragraph, "a:pPr", algn="ctr" if center else "l")
        add(pp, "a:buNone")
        run = add(paragraph, "a:r")
        rp = add(run, "a:rPr", lang="en-US", sz=str(round(size * 100)), b="1" if bold else "0", dirty="0")
        fill(rp, "000000")
        add(rp, "a:latin", typeface="Times New Roman")
        add(run, "a:t").text = line
        add(paragraph, "a:endParaRPr", lang="en-US", sz=str(round(size * 100)))
    if box:
        bp = body.find("a:bodyPr", NS)
        bp.attrib.update(anchor="ctr", lIns=pt(8), rIns=pt(8))
    tree.append(sp)
    return sp


def connector(tree, x1, y1, x2, y2):
    ids = [int(node.attrib["id"]) for node in tree.findall(".//p:cNvPr", NS)]
    sp = add(tree, "p:cxnSp")
    nv = add(sp, "p:nvCxnSpPr")
    add(nv, "p:cNvPr", id=str(max(ids + [1]) + 1), name="Flow connector")
    add(nv, "p:cNvCxnSpPr")
    add(nv, "p:nvPr")
    pr = add(sp, "p:spPr")
    xf = add(pr, "a:xfrm")
    add(xf, "a:off", x=pt(min(x1, x2)), y=pt(min(y1, y2)))
    add(xf, "a:ext", cx=pt(abs(x2 - x1)), cy=pt(abs(y2 - y1)))
    add(add(pr, "a:prstGeom", prst="line"), "a:avLst")
    ln = add(pr, "a:ln", w="19050")
    fill(ln, "404040")
    add(ln, "a:prstDash", val="solid")
    add(ln, "a:headEnd", type="none", w="med", len="med")
    add(ln, "a:tailEnd", type="triangle", w="med", len="med")


def diagram(tree, template, kind):
    def box(text, x, y, w, h=72):
        return text_shape(tree, template, text, x, y, w, h, 22, center=True, box=True)
    def label(text, x, y, w, h=55, size=23):
        return text_shape(tree, template, text, x, y, w, h, size, center=True)
    if kind == "distribution":
        box("All original\nXYZ beats", 50, 104, 210)
        box("Per-beat feature\nencoding", 370, 104, 230)
        box("Mean, spread\nor quantiles", 710, 104, 200)
        connector(tree, 264, 140, 363, 140)
        connector(tree, 604, 140, 703, 140)
        label("Recording × window × feature projection × quantile", 65, 224, 830)
    elif kind == "bridge":
        box("Paired ECG\nteacher", 65, 103, 265)
        box("Measured XYZ\nstudent", 630, 103, 265)
        connector(tree, 335, 139, 624, 139)
        label("Training alignment", 345, 102, 275, 33, 22)
        label("Inference uses measured XYZ alone", 120, 224, 720)
    elif kind == "visits":
        box("Patient", 50, 108, 165)
        box("Recording A\nRecording B", 335, 108, 250)
        box("One patient\nprediction", 710, 108, 200)
        connector(tree, 220, 144, 328, 144)
        connector(tree, 590, 144, 703, 144)
        label("Which examination should support the prediction?", 70, 235, 820)
    elif kind == "readout":
        box("XYZ beat", 50, 108, 170)
        box("Frozen ResNet\nencoder", 330, 108, 235)
        box("64 pooled features or\n256 temporal features", 675, 108, 235)
        connector(tree, 225, 144, 323, 144)
        connector(tree, 570, 144, 668, 144)
        label("Patient pooling and training-only ridge readout", 90, 226, 780)
    elif kind == "pipeline":
        for text, x in [("Measured\nXYZ",50),("Waveforms\nor tensors",280),("Training-only\nmodel fit",510),("Patient\nmacro F1",740)]:
            box(text,x,108,170)
        for x1,x2 in [(225,273),(455,503),(685,733)]:
            connector(tree,x1,144,x2,144)
        label("Fixed patient folds and independent result replay",65,228,830)
    elif kind == "multiview":
        box("Raw signal\nR",65,104,230)
        box("Denoised signal\nD",365,104,230)
        box("Recording median\nM",665,104,230)
        connector(tree,300,140,358,140)
        connector(tree,600,140,658,140)
        label("R = D + (R−D)          D = M + (D−M)",70,224,820)


def notes_readable(notes):
    """Keep author notes legible even for compact ledger-style evidence strings."""
    notes = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", notes)
    notes = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", notes)
    return notes


def main():
    global BUILD, FINAL, PDF
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=BUILD)
    parser.add_argument('--pptx', type=Path, default=FINAL)
    parser.add_argument('--pdf', type=Path, default=PDF)
    parser.add_argument('--guide', type=Path, default=ROOT/'docs/progress_update_20261120_presenter_guide.md')
    args = parser.parse_args()
    BUILD, FINAL, PDF = args.build_dir.resolve(), args.pptx.resolve(), args.pdf.resolve()
    meta = json.loads((BUILD / "metadata.json").read_text())
    with ZipFile(BUILD / "candidate.pptx") as z:
        parts = {name: z.read(name) for name in z.namelist()}
    charts = []
    for index, spec in enumerate(meta["slides"], 1):
        name = f"ppt/slides/slide{index}.xml"
        slide = E.fromstring(parts[name])
        shapes = slide.findall("p:cSld/p:spTree/p:sp", NS)
        template = copy.deepcopy(shapes[0])
        tree = slide.find("p:cSld/p:spTree", NS)
        if spec.get("cover"):
            boxes = [(50,52,860,45),(50,112,860,82),(50,226,860,32),(50,280,860,100),(50,422,860,68)]
            for shape, coords in zip(shapes, boxes):
                position(shape, *coords)
                for par in shape.findall("p:txBody/a:p", NS):
                    pp = par.find("a:pPr", NS)
                    if pp is None:
                        pp = E.Element(tag("a:pPr"))
                        par.insert(0, pp)
                    pp.set("algn", "ctr")
        else:
            position(shapes[0], 36, 20, 888, 54)
            end = 105
            if "table" in spec:
                end = 100 + spec["rowH"] * len(spec["table"]) + 24
                frame = next(frame for frame in tree.findall("p:graphicFrame", NS) if frame.find(".//a:tbl", NS) is not None)
                xf = frame.find("p:xfrm", NS)
                xf.find("a:off", NS).attrib.update(x=pt(50), y=pt(100))
                xf.find("a:ext", NS).attrib.update(cx=pt(860), cy=pt(spec["rowH"]*len(spec["table"])))
            elif "chart" in spec:
                end = 93 + spec["chartH"] + 20
                charts.append(spec["chart"])
                frame = next(frame for frame in tree.findall("p:graphicFrame", NS) if frame.find(".//c:chart", NS) is not None)
                xf = frame.find("p:xfrm", NS)
                xf.find("a:off", NS).attrib.update(x=pt(45), y=pt(93))
                xf.find("a:ext", NS).attrib.update(cx=pt(870), cy=pt(spec["chartH"]))
            elif "diagram" in spec:
                end = 310 if spec["diagram"] == "visits" else 300
            if "image" in spec:
                picture = tree.find('p:pic', NS)
                xf = picture.find('p:spPr/a:xfrm', NS)
                x_img,y_img,w_img,h_img = spec['imageBox']
                xf.find('a:off', NS).attrib.update(x=pt(x_img),y=pt(y_img))
                xf.find('a:ext', NS).attrib.update(cx=pt(w_img),cy=pt(h_img))
            count = len(spec["lines"])
            h = min(120 if spec.get("side") else 77 if spec.get("sideLeft") else 82, (510-end)/count) if count else 0
            x = 685 if spec.get("side") else 50
            w = 225 if spec.get("side") else 520 if spec.get("sideLeft") else 860
            for j, shape in enumerate(shapes[1:]):
                position(shape, x, end+j*h, w, h-5)
            if "diagram" in spec:
                diagram(tree, template, spec["diagram"])
        # Booktabs-style tables. Preserve native table cells and source numbers.
        for table in slide.findall(".//a:tbl", NS):
            rows = table.findall("a:tr", NS)
            for ri, row in enumerate(rows):
                for cell in row.findall("a:tc", NS):
                    pr = cell.find("a:tcPr", NS)
                    fill(pr, "FFFFFF")
                    pr.attrib.update(marL=pt(6), marR=pt(6), marT=pt(3), marB=pt(3), anchor="ctr")
                    for edge in ("lnL", "lnR", "lnT", "lnB"):
                        node = pr.find("a:"+edge, NS)
                        if node is None:
                            node = add(pr, "a:"+edge)
                        node.clear()
                        if (edge == "lnT" and ri == 0) or (edge == "lnB" and ri in (0, len(rows)-1)):
                            node.set("w", "9500")
                            fill(node, "000000")
                            add(node, "a:prstDash", val="solid")
                        else:
                            add(node, "a:noFill")
                    for rp in cell.findall(".//a:rPr", NS) + cell.findall(".//a:defRPr", NS):
                        rp.set("sz", str(spec.get("font",22)*100))
                        fill(rp, "000000")
        parts[name] = xml(slide)
    # Preserve editable native charts/workbooks, changing presentation styling only.
    paths = sorted([name for name in parts if re.fullmatch(r"ppt/charts/chart\d+\.xml", name)], key=lambda n:int(re.search(r"chart(\d+)",n)[1]))
    assert len(paths) == len(charts)
    for name, spec in zip(paths, charts):
        chart = E.fromstring(parts[name])
        for node in chart.findall(".//a:latin", NS):
            node.set("typeface", "Times New Roman")
        for node in chart.findall(".//a:defRPr", NS):
            node.set("sz", "2100")
            fill(node, "000000")
        for axis in chart.findall(".//c:valAx", NS):
            scaling = axis.find("c:scaling", NS)
            for child in list(scaling):
                if child.tag.split("}")[-1] in {"min","max"}:
                    scaling.remove(child)
            add(scaling,"c:min",val="0")
            add(scaling,"c:max",val=spec["max"])
            fmt = axis.find("c:numFmt", NS)
            if fmt is not None:
                fmt.set("formatCode", "0.0")
        for axis in chart.findall(".//c:catAx", NS)+chart.findall(".//c:valAx", NS):
            for key in ("majorGridlines","minorGridlines"):
                child = axis.find("c:"+key, NS)
                if child is not None:
                    axis.remove(child)
        for series in chart.findall(".//c:ser", NS):
            for sppr in series.findall("c:spPr", NS)+series.findall("c:dPt/c:spPr", NS):
                fill(sppr,"707070")
            labels = series.find("c:dLbls", NS)
            if labels is not None:
                for key,value in [("showVal","1"),("dLblPos","outEnd")]:
                    child = labels.find("c:"+key,NS)
                    if child is not None:
                        child.set("val",value)
                fmt = labels.find("c:numFmt",NS)
                if fmt is not None:
                    fmt.set("formatCode","0.000")
        layout = chart.find("c:chart/c:plotArea/c:layout/c:manualLayout", NS)
        if layout is not None:
            for key,value in {"x":"0.37","y":"0.06","w":"0.53","h":"0.76"}.items():
                child=layout.find("c:"+key,NS)
                if child is not None:
                    child.set("val",value)
        parts[name] = xml(chart)
    FINAL.parent.mkdir(parents=True,exist_ok=True)
    PDF.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(FINAL,"w",ZIP_DEFLATED) as z:
        for name,data in parts.items():
            z.writestr(name,data)
    # Readable presenter guide also lives beside the project documentation.
    main_count = sum(not s.get('backup', False) for s in meta['slides'])
    appendix_count = len(meta['slides'])-main_count
    scope = meta.get('scopeDescription') or 'Based on the illustrated 17 November deck.'
    guide = ["# Presentation update, 20 November 2026\n", f"{main_count} main slides and {appendix_count} appendix slides. {scope} All study results are development evidence. The original presentation and numerical study artifacts are unchanged.\n", "The deck has no footer notes or slide-number fragments. Evidence and image attribution are in presenter notes. The appendix preserves detailed matched comparisons.\n"]
    for i,spec in enumerate(meta["slides"],1):
        guide.append(f"## {i}. {spec['title']}\n")
        guide.extend(line+"\n" for line in spec["lines"])
        guide.append("\nPresenter notes\n\n"+notes_readable(spec["notes"])+"\n")
        guide.append("\nEvidence\n\n"+"\n".join("- "+s for s in spec["sources"])+"\n")
    args.guide.parent.mkdir(parents=True, exist_ok=True)
    args.guide.write_text("\n".join(guide))
    export = f'''tell application "Keynote"
set d to open POSIX file "{FINAL}"
export d to POSIX file "{PDF}" as PDF
close d saving no
end tell
'''
    (BUILD/"export_pdf.applescript").write_text(export)
    print(json.dumps({"pptx":str(FINAL),"slides":len(meta["slides"]),"sha256":hashlib.sha256(FINAL.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()

/** Add sourced signal and experiment images to a copy of the formal deck.
 * Native text/tables/charts/notes in the source remain editable.
 * node scripts/illustrate_progress_presentation_20261117.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
const root=process.cwd();
const build=path.join(root,'tmp/presentations/progress_images_20261117');
fs.mkdirSync(build,{recursive:true});
const source=path.join(root,'output/pptx/mi_progress_since_october_2026.pptx');
const target=path.join(root,'output/pptx/mi_progress_since_october_2026_illustrated.pptx');
const assets=path.join(root,'output/pptx/assets/progress_illustrated_20261117');
fs.mkdirSync(assets,{recursive:true});
const ecg=path.join(assets,'ecg_cycle_yitzhaknat.png');
if(!fs.existsSync(ecg))fs.copyFileSync(path.join(build,'source/ecg_cycle.png'),ecg);
const py=String.raw`import copy,hashlib,json,posixpath,re,xml.etree.ElementTree as E
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
root=Path.cwd(); build=root/'tmp/presentations/progress_images_20261117'
src=root/'output/pptx/mi_progress_since_october_2026.pptx'
out=root/'output/pptx/mi_progress_since_october_2026_illustrated.pptx'
N={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','rel':'http://schemas.openxmlformats.org/package/2006/relationships','ct':'http://schemas.openxmlformats.org/package/2006/content-types'}
for k,v in N.items():E.register_namespace(k,v)
def tag(n):a,b=n.split(':');return '{'+N[a]+'}'+b
def sub(p,n,**kw):return E.SubElement(p,tag(n),kw)
def xml(r):return E.tostring(r,encoding='utf-8',xml_declaration=True)
with ZipFile(src) as z:parts={n:z.read(n) for n in z.namelist()}
types=E.fromstring(parts['[Content_Types].xml'])
if not any(n.attrib.get('Extension')=='png' for n in types):sub(types,'ct:Default',Extension='png',ContentType='image/png')
def override(part,content):sub(types,'ct:Override',PartName='/'+part,ContentType=content)
assets={
 'vcg':{'path':'output/latex/assets/vcg_processing_grayscale.png','credit':'Project figure from the earlier implementation deck. Public PTB record patient001/s0010_re. Recorded and denoised measured XYZ, detector peaks and one R-relative crop.','source':'https://physionet.org/content/ptbdb/1.0.0/'},
 'ecg':{'path':'output/pptx/assets/progress_illustrated_20261117/ecg_cycle_yitzhaknat.png','credit':'YitzhakNat, Normal EKG wave, 26 August 2022, Wikimedia Commons. CC BY-SA 4.0. Unmodified 960px PNG rendering. Educational schematic only, not a project recording or delineation output.','source':'https://commons.wikimedia.org/wiki/File:Normal_EKG_wave.svg','license':'https://creativecommons.org/licenses/by-sa/4.0/','download':'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Normal_EKG_wave.svg/960px-Normal_EKG_wave.svg.png'},
 'curve':{'path':'results/neural_learning_curve_v1/complete/learning_curve.png','credit':'Existing project aggregate figure. Error bars show sample SD across training seeds. Training-patient budgets and held-out target patients remain those of the completed study.','source':'results/neural_learning_curve_v1/complete/REPORT.md'},
 'source':{'path':'results/supervised_source_pilot_v1/learning_curves.png','credit':'Existing source-only aggregate figure. Patient-weighted validation macro AP, both declared source objectives and the constant baseline. This is not target six-class patient macro F1.','source':'results/supervised_source_pilot_v1/REPORT.md'}
}
rels_template=E.fromstring(parts['ppt/slides/_rels/slide15.xml.rels'])
layout=next(n.attrib['Target'] for n in rels_template if n.attrib['Type'].endswith('/slideLayout'))
notes_template=E.fromstring(parts['ppt/notesSlides/notesSlide15.xml'])
notes_rels_template=E.fromstring(parts['ppt/notesSlides/_rels/notesSlide15.xml.rels'])
body_template=E.fromstring(parts['ppt/slides/slide15.xml']).findall('.//p:sp',N)[1]
title_template=E.fromstring(parts['ppt/slides/slide15.xml']).findall('.//p:sp',N)[0]
number_template=E.fromstring(parts['ppt/slides/slide15.xml']).findall('.//p:sp',N)[-1]
def point(v):return str(round(v*12700))
def set_text(sp,text,x,y,w,h,size=20,bold=False,italic=False):
 off=sp.find('p:spPr/a:xfrm/a:off',N);off.set('x',point(x));off.set('y',point(y))
 ext=sp.find('p:spPr/a:xfrm/a:ext',N);ext.set('cx',point(w));ext.set('cy',point(h))
 tx=sp.find('p:txBody',N);body=tx.find('a:bodyPr',N)
 body.set('anchor','t')
 for k in ('lIns','rIns','tIns','bIns'):body.set(k,'0')
 for c in list(body):body.remove(c)
 sub(body,'a:noAutofit')
 for p in tx.findall('a:p',N):tx.remove(p)
 p=sub(tx,'a:p');pr=sub(p,'a:pPr',algn='l');sub(pr,'a:buNone')
 for j,line in enumerate(text.split('\n')):
  if j:sub(p,'a:br')
  run=sub(p,'a:r');rp=sub(run,'a:rPr',lang='en-US',sz=str(round(size*100)),b='1' if bold else '0',i='1' if italic else '0',dirty='0')
  sub(sub(rp,'a:solidFill'),'a:srgbClr',val='000000');sub(rp,'a:latin',typeface='Times New Roman');sub(run,'a:t').text=line
 return sp
def textbox(tree,text,x,y,w,h,size=20,bold=False,italic=False):
 sp=copy.deepcopy(body_template);ids=[int(n.attrib['id']) for n in tree.findall('.//p:cNvPr',N)]
 sp.find('p:nvSpPr/p:cNvPr',N).set('id',str(max(ids+[1])+1))
 sp.find('p:nvSpPr/p:cNvPr',N).set('name','Figure caption' if size<16 else 'Explanatory text')
 set_text(sp,text,x,y,w,h,size,bold,italic);tree.append(sp);return sp
def next_rid(rel):return 'rId'+str(max([int(n.attrib['Id'][3:]) for n in rel if n.attrib['Id'].startswith('rId')]+[0])+1)
def picture(slide,rel,key,x,y,w,h):
 spec=assets[key];data=(root/spec['path']).read_bytes();spec['sha256']=hashlib.sha256(data).hexdigest()
 media='ppt/media/progress_illustrated_'+key+'.png';parts[media]=data
 rid=next_rid(rel);sub(rel,'rel:Relationship',Id=rid,Type=N['r']+'/image',Target='../media/'+Path(media).name)
 tree=slide.find('p:cSld/p:spTree',N);ids=[int(n.attrib['id']) for n in tree.findall('.//p:cNvPr',N)]
 pic=sub(tree,'p:pic');nv=sub(pic,'p:nvPicPr');sub(nv,'p:cNvPr',id=str(max(ids+[1])+1),name=key+' image',descr=spec['credit'])
 locks=sub(nv,'p:cNvPicPr');sub(locks,'a:picLocks',noChangeAspect='1');sub(nv,'p:nvPr')
 fill=sub(pic,'p:blipFill');b=sub(fill,'a:blip');b.set(tag('r:embed'),rid);sub(sub(fill,'a:stretch'),'a:fillRect')
 sp=sub(pic,'p:spPr');xf=sub(sp,'a:xfrm');sub(xf,'a:off',x=point(x),y=point(y));sub(xf,'a:ext',cx=point(w),cy=point(h));sub(sub(sp,'a:prstGeom',prst='rect'),'a:avLst')
def add_notes(physical,notes):
 nr=copy.deepcopy(notes_template)
 sp=next(sp for sp in nr.findall('.//p:sp',N) if sp.find('p:nvSpPr/p:nvPr/p:ph',N) is not None and sp.find('p:nvSpPr/p:nvPr/p:ph',N).attrib.get('type')=='body')
 tx=sp.find('p:txBody',N)
 for p in tx.findall('a:p',N):tx.remove(p)
 p=sub(tx,'a:p');run=sub(p,'a:r');sub(run,'a:t').text=notes
 path=f'ppt/notesSlides/notesSlide{physical}.xml';parts[path]=xml(nr)
 rel=copy.deepcopy(notes_rels_template)
 for n in rel:
  if n.attrib['Type'].endswith('/slide'):n.set('Target',f'../slides/slide{physical}.xml')
 parts[f'ppt/notesSlides/_rels/notesSlide{physical}.xml.rels']=xml(rel)
 override(path,'application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml')
def append_notes(physical,extra):
 path=f'ppt/notesSlides/notesSlide{physical}.xml';r=E.fromstring(parts[path]);sp=next(sp for sp in r.findall('.//p:sp',N) if sp.find('p:nvSpPr/p:nvPr/p:ph',N) is not None and sp.find('p:nvSpPr/p:nvPr/p:ph',N).attrib.get('type')=='body')
 p=sub(sp.find('p:txBody',N),'a:p');sub(sub(p,'a:r'),'a:t').text=extra;parts[path]=xml(r)
def blank(physical,title,notes):
 r=E.fromstring(parts['ppt/slides/slide15.xml']);tree=r.find('p:cSld/p:spTree',N)
 for c in list(tree):
  if c.tag not in (tag('p:nvGrpSpPr'),tag('p:grpSpPr')):tree.remove(c)
 sp=copy.deepcopy(title_template);set_text(sp,title,20,24,920,45,29,True);tree.append(sp)
 sp=copy.deepcopy(number_template);sp.find('p:nvSpPr/p:cNvPr',N).set('id','3');tree.append(sp)
 rel=E.Element(tag('rel:Relationships'));sub(rel,'rel:Relationship',Id='rId1',Type=N['r']+'/slideLayout',Target=layout)
 sub(rel,'rel:Relationship',Id='rId2',Type=N['r']+'/notesSlide',Target=f'../notesSlides/notesSlide{physical}.xml')
 add_notes(physical,notes);override(f'ppt/slides/slide{physical}.xml','application/vnd.openxmlformats-officedocument.presentationml.slide+xml')
 return r,rel
signal_notes='MAIN PRESENTATION\n\nThe left panel reuses a public measured Frank XYZ example from the earlier implementation presentation. Gray and black show recorded and denoised signals. Dashed lines show detector peaks, and the shaded region marks one 651-sample R-relative crop. It is an example, not a representative class template or clinical diagnosis. The right panel is a generic ECG schematic for orientation. ECG is not the primary target input. Its labeled intervals do not validate our automatic boundaries.\n\nIMAGE CREDITS\n'+assets['vcg']['credit']+'\n'+assets['vcg']['source']+'\n'+assets['ecg']['credit']+'\n'+assets['ecg']['source']+'\n'+assets['ecg']['license']
r,rel=blank(34,'ECG illustration and measured VCG',signal_notes);tree=r.find('p:cSld/p:spTree',N)
textbox(tree,'Measured Frank XYZ example',50,88,570,30,21,True)
textbox(tree,'ECG schematic',650,88,260,30,21,True)
picture(r,rel,'vcg',50,133,570,570*988/1786)
picture(r,rel,'ecg',650,142,260,260*767/960)
textbox(tree,'Recorded and denoised XYZ. Dashed lines: detected R peaks.\nShading: one R-relative crop, 651 samples per lead.',50,460,570,42,14)
textbox(tree,'Illustration only.\nMeasured XYZ remains our primary input.\n\nYitzhakNat, CC BY-SA 4.0',650,365,260,100,16)
textbox(tree,'Signal example: public PTB database. Image sources and license links appear in speaker notes.',50,511,830,18,11,False,True)
parts['ppt/slides/slide34.xml']=xml(r);parts['ppt/slides/_rels/slide34.xml.rels']=xml(rel)
# Original diagnostics slide: retain the conclusion, with the actual completed learning curve.
name='ppt/slides/slide15.xml';r=E.fromstring(parts[name]);rel=E.fromstring(parts['ppt/slides/_rels/slide15.xml.rels']);tree=r.find('p:cSld/p:spTree',N)
shapes=tree.findall('p:sp',N)
lines=[
 '164 patients remain the independent sample units.',
 'Training-patient curve: approximately 0.407 to 0.550 macro F1.',
 '35 patients were wrong across all three selected ResNet seeds.',
 'Chronology errors persisted in IMI/ILMI, without a simple normalization mismatch.',
 'These diagnostics do not prove label noise, a ceiling or a required sample count.'
]
for j,line in enumerate(lines):set_text(shapes[j+1],'• '+line,625,103+j*76,285,72,18)
picture(r,rel,'curve',50,127,540,540*810/1296)
textbox(tree,'Completed patient-budget study. Error bars: training-seed SD.',50,480,550,28,12,False,True)
parts[name]=xml(r);parts['ppt/slides/_rels/slide15.xml.rels']=xml(rel)
append_notes(15,'\nILLUSTRATED REVISION\nAll original diagnostic interpretation remains above. The actual completed learning-curve figure illustrates the budget result. '+assets['curve']['credit']+'\nhttps://github.com/Kumaryan12/major_project/blob/main/'+assets['curve']['source'])
curve_notes='MAIN PRESENTATION\n\nThis is the completed source pilot learning curve, not the target transfer study. The two native annotation objectives both trained 15 epochs and selected epoch 13 by common source-validation macro average precision. The constant predictor uses training prevalence. Validation is model-selection data, not independent confirmation. Good source learning does not establish a target localization gain. The target transfer study remains in progress at presentation preparation.\n\nIMAGE CREDIT\n'+assets['source']['credit']+'\nhttps://github.com/Kumaryan12/major_project/blob/main/'+assets['source']['path']+'\nhttps://github.com/Kumaryan12/major_project/blob/main/'+assets['source']['source']
r,rel=blank(35,'Source-learning curves',curve_notes);tree=r.find('p:cSld/p:spTree',N)
picture(r,rel,'source',50,96,660,660*640/1120)
textbox(tree,'Both declared annotation objectives remain in the comparison.',742,126,170,112,19)
textbox(tree,'Best source checkpoint:\nepoch 13 of 15.',742,270,170,90,19)
textbox(tree,'Source validation AP selects checkpoints. Target localization results remain separate.',742,372,170,106,18)
textbox(tree,'Source-only multilabel diagnostic learning. This curve does not report target six-class macro F1.',50,492,850,28,13,False,True)
parts['ppt/slides/slide35.xml']=xml(r);parts['ppt/slides/_rels/slide35.xml.rels']=xml(rel)
# Add two slides to the existing presentation relationships and order.
pr=E.fromstring(parts['ppt/_rels/presentation.xml.rels']);pres=E.fromstring(parts['ppt/presentation.xml']);ids=pres.find('p:sldIdLst',N)
old=list(ids);old_physical={int(re.search(r'slide(\d+)\.xml',n.attrib['Target'])[1]):n.attrib['Id'] for n in pr if n.attrib['Type'].endswith('/slide')}
new={}
for physical in (34,35):
 rid=next_rid(pr);sub(pr,'rel:Relationship',Id=rid,Type=N['r']+'/slide',Target=f'slides/slide{physical}.xml')
 item=E.Element(tag('p:sldId'),{'id':str(1000+physical),tag('r:id'):rid});new[physical]=item
order=list(range(1,4))+[34]+list(range(4,19))+[35]+list(range(19,34))
for c in list(ids):ids.remove(c)
mapping={n.attrib[tag('r:id')]:n for n in old}
for physical in order:ids.append(new[physical] if physical in new else mapping[old_physical[physical]])
parts['ppt/presentation.xml']=xml(pres);parts['ppt/_rels/presentation.xml.rels']=xml(pr)
# Page markers and a portable companion guide follow actual presentation order.
guide=['# Illustrated implementation progress presentation','','24 main slides and 11 technical backup slides. The original formal deck remains unchanged. This revision adds four sourced images, two explanatory slides and the completed learning curve beside the diagnostics. All original result tables, charts and complete speaker notes remain.','','The ECG schematic is illustrative. The measured-Frank VCG example is a public PTB recording. Source-learning AP remains a separate endpoint from target macro F1. The current transfer study is in progress at preparation; check its monitor before presenting.','']
for display,physical in enumerate(order,1):
 name=f'ppt/slides/slide{physical}.xml';r=E.fromstring(parts[name]);tree=r.find('p:cSld/p:spTree',N)
 for sp in tree.findall('p:sp',N):
  off=sp.find('p:spPr/a:xfrm/a:off',N)
  if off is not None and int(off.attrib['x'])>880*12700 and int(off.attrib['y'])>505*12700:
   ts=sp.findall('p:txBody/a:p/a:r/a:t',N)
   if ts:ts[0].text=str(display)
 parts[name]=xml(r)
 title=''.join(tree.findall('p:sp',N)[0].find('p:txBody',N).itertext())
 nr=E.fromstring(parts[f'ppt/notesSlides/notesSlide{physical}.xml'])
 sp=next(sp for sp in nr.findall('.//p:sp',N) if sp.find('p:nvSpPr/p:nvPr/p:ph',N) is not None and sp.find('p:nvSpPr/p:nvPr/p:ph',N).attrib.get('type')=='body')
 note='\n'.join(t.text or '' for t in sp.findall('p:txBody/a:p/a:r/a:t',N))
 guide.extend([f'## {display}. {title}','',note,''])
# Keep the OPC content-types default namespace and schema-ordered chart styling.
# Only run-property order changes in the two inherited charts. Values/workbooks stay fixed.
E.register_namespace('',N['ct'])
parts['[Content_Types].xml']=xml(types)
for name in list(parts):
 if not re.fullmatch(r'ppt/charts/chart\d+\.xml',name):continue
 r=E.fromstring(parts[name])
 for pr in r.findall('.//a:defRPr',N):
  fills=[c for c in pr if c.tag.split('}')[-1] in ('noFill','solidFill','gradFill','blipFill','pattFill','grpFill')]
  for c in fills:pr.remove(c)
  pos=1 if len(pr) and pr[0].tag==tag('a:ln') else 0
  for c in fills:pr.insert(pos,c);pos+=1
 parts[name]=xml(r)
if 'docProps/app.xml' in parts:
 r=E.fromstring(parts['docProps/app.xml'])
 for n in r:
  if n.tag.split('}')[-1] in ('Slides','Notes'):n.text='35'
 parts['docProps/app.xml']=xml(r)
with ZipFile(out,'w',ZIP_DEFLATED) as z:
 for n,v in parts.items():z.writestr(n,v)
(root/'docs/progress_presentation_20261117_illustrated_notes.md').write_text('\n'.join(guide))
(root/'output/pptx/assets/progress_illustrated_20261117/credits.json').write_text(json.dumps(assets,indent=2)+'\n')
(build/'edit_map.json').write_text(json.dumps({'source':str(src),'output':str(out),'physical_slide_order':order,'changed_display_slides':[4,16,20],'main_slides':24,'backup_slides':11,'assets':assets},indent=2))
print(out)
`;
const r=spawnSync(path.join(root,'.venv/bin/python'),['-c',py],{cwd:root,encoding:'utf8',maxBuffer:8*1024*1024});
if(r.status!==0)throw new Error(r.stderr||r.stdout);
process.stdout.write(r.stdout);

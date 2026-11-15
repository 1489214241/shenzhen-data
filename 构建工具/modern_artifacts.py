"""Export/restore frozen Modern artifacts without changing historical bytes."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

PUBLIC=Path(__file__).resolve().parents[1]
CONTRACT=PUBLIC/'数据清单/Modern_K1补充_20261115/contract.json'

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()
def target(root,relative):
    p=PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe relative path: '+relative)
    dest=(root/p).resolve()
    if not dest.is_relative_to(root.resolve()): raise ValueError('Path escapes root')
    return dest
def check(root,rows):
    issues=[]
    for r in rows:
        p=target(root,r['path'])
        if not p.is_file(): issues.append(dict(path=r['path'],problem='missing'))
        elif sha(p)!=r['sha256']: issues.append(dict(path=r['path'],problem='hash_mismatch'))
    return issues
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def export(args,c):
    rows=[r for r in c['files'] if r['group']=='k1' or args.with_sources]
    issues=check(args.root,rows)
    write(args.out/'export_check.json',dict(passed=not issues,checked=len(rows),issues=issues))
    if issues: raise SystemExit('Missing/changed originals; no archive created. See export_check.json')
    names=['modern-k1-missing-47.zip']+(['modern-source-eight-16.zip'] if args.with_sources else [])
    for name in names:
        if (args.out/name).exists(): raise FileExistsError(args.out/name)
    assets=[]
    for name in names:
        group='k1' if name.startswith('modern-k1') else 'sources'
        chosen=[r for r in rows if r['group']==group and (group=='sources' or r['package_required'])]
        dest=args.out/name
        with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for row in chosen:
                info=zipfile.ZipInfo('model/'+row['path'],date_time=(2026,11,15,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                with target(args.root,row['path']).open('rb') as source,z.open(info,'w',force_zip64=True) as output:
                    shutil.copyfileobj(source,output,4*1024**2)
            z.writestr('manifest.json',json.dumps(dict(group=group,files=chosen),ensure_ascii=False,indent=2))
        with zipfile.ZipFile(dest) as z:
            assert z.testzip() is None
            for row in chosen:
                with z.open('model/'+row['path']) as stream:
                    h=hashlib.sha256()
                    for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b)
                assert h.hexdigest()==row['sha256'],row['path']
        assets.append(dict(name=name,bytes=dest.stat().st_size,sha256=sha(dest),files=len(chosen)))
        print('PACKED',name,len(chosen),dest.stat().st_size,flush=True)
    write(args.out/'assets.json',dict(assets=assets,target_gold_included=False))

def restore(args,c):
    bypath={r['path']:r for r in c['files']}
    payload={};archives=[]
    # Validate every payload and every existing destination before writing anything.
    for archive in args.archives:
        z=zipfile.ZipFile(archive);archives.append(z)
        m=json.loads(z.read('manifest.json'))
        expected={'manifest.json'}|{'model/'+r['path'] for r in m['files']}
        assert len(z.namelist())==len(expected) and set(z.namelist())==expected
        for row in m['files']:
            assert row==bypath[row['path']],row['path']
            target(args.root,row['path'])
            h=hashlib.sha256()
            with z.open('model/'+row['path']) as f:
                for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
            assert h.hexdigest()==row['sha256'],row['path']
            payload[row['path']]=(row,z)
    public={}
    for row in c['files']:
        if row['group']=='k1' and not row['package_required']:
            p=target(PUBLIC,row['public_path']);assert sha(p)==row['sha256'],row['public_path']
            public[row['path']]=(row,p)
    for row,_ in list(payload.values())+list(public.values()):
        p=target(args.root,row['path'])
        if p.exists() and (not p.is_file() or sha(p)!=row['sha256']):
            raise FileExistsError('Refuse to overwrite different existing file: '+str(p))
    for row,z in payload.values():
        p=target(args.root,row['path']);p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            with z.open('model/'+row['path']) as src,p.open('xb') as out:shutil.copyfileobj(src,out,4*1024**2)
    for row,src in public.values():
        p=target(args.root,row['path']);p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            with src.open('rb') as f,p.open('xb') as out:shutil.copyfileobj(f,out)
    for z in archives:z.close()
    rows=[row for row,_ in list(payload.values())+list(public.values())]
    assert not check(args.root,rows)
    print(json.dumps(dict(restored_verified=len(rows),root=str(args.root)),ensure_ascii=False))

def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('export');p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--with-sources',action='store_true')
    p=sub.add_parser('restore');p.add_argument('--root',type=Path,required=True);p.add_argument('archives',nargs='+',type=Path)
    p=sub.add_parser('verify');p.add_argument('--root',type=Path,required=True);p.add_argument('--with-sources',action='store_true')
    args=ap.parse_args();c=read(CONTRACT)
    if args.command=='verify':
        rows=[r for r in c['files'] if r['group']=='k1' or args.with_sources]
        issues=check(args.root,rows);print(json.dumps(dict(checked=len(rows),passed=not issues,issues=issues),ensure_ascii=False,indent=2))
        if issues:raise SystemExit(1)
    else:globals()[args.command](args,c)

if __name__=='__main__':main()

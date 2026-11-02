"""Back up every eligible workspace file, including gitignored artifacts.

Excludes only Git's own metadata, files above GitHub's 100 MiB limit, and
non-regular filesystem objects. Uses small commits/pushes to stay below 2 GiB.
Re-running after interruption skips already clean tracked files and retries the
unpushed HEAD before creating additional commits. No force push or deletion.
"""
import collections
import json
import os
from pathlib import Path
import stat
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'backup/20261102'
MAX_FILE=100*1024**2
MAX_BATCH=750*1024**2

def git(*args,input=None,capture=False):
    return subprocess.run(['git','-c','gc.auto=0',*args],cwd=ROOT,input=input,
        stdout=subprocess.PIPE if capture else None,check=True).stdout

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def push():
    for attempt in range(3):
        result=subprocess.run(['git','-c','gc.auto=0','-c','pack.threads=2','-c','pack.window=0',
            '-c','pack.compression=1','push','origin','main'],cwd=ROOT)
        if result.returncode==0:
            return
        if attempt<2:
            time.sleep(10)
    raise RuntimeError('Push failed; rerun this script to retry the committed HEAD')

def stage(paths):
    payload=b''.join(b':(literal)'+os.fsencode(p)+b'\0' for p in paths)
    git('add','-f','--pathspec-from-file=-','--pathspec-file-nul',input=payload)

def commit(message):
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        git('commit','--quiet','-m',message)
        push()

def main():
    import fcntl
    lock=(ROOT/'.git/workspace-backup.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert git('branch','--show-current',capture=True).strip()==b'main'
    remote=git('remote','get-url','origin',capture=True).decode().strip()
    assert remote in ('git@github.com:LinWang1225/test-V1.git','https://github.com/LinWang1225/test-V1.git')
    push()  # Resume an interrupted push before moving HEAD again.
    eligible=[];excluded=[]
    for base,dirs,files in os.walk(ROOT,followlinks=False):
        dirs[:]=[d for d in dirs if d!='.git']
        names=list(files)+[d for d in dirs if (Path(base)/d).is_symlink()]
        for name in names:
            p=Path(base)/name
            try:
                info=p.lstat()
            except FileNotFoundError:
                continue
            rel=str(p.relative_to(ROOT))
            if stat.S_ISLNK(info.st_mode):
                size=len(os.fsencode(os.readlink(p)))
                kind='symlink'
            elif stat.S_ISREG(info.st_mode):
                size=info.st_size;kind='file'
            else:
                excluded.append(dict(path=rel,reason='non_regular_filesystem_object'))
                continue
            if size>MAX_FILE:
                excluded.append(dict(path=rel,bytes=size,reason='exceeds_100_MiB'))
            else:
                eligible.append(dict(path=rel,bytes=size,kind=kind))
    eligible.sort(key=lambda r:(0 if r['path'].startswith(('scripts/','experiments/staged_algorithm_20261102/')) else
        1 if r['path'].startswith('experiments/') else 2 if r['path'].startswith('results/') else 3,r['path']))
    write(STATE/'excluded_files.json',excluded)
    write(STATE/'snapshot_manifest.json',dict(snapshot_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        max_file_bytes=MAX_FILE,eligible_count=len(eligible),eligible_bytes=sum(r['bytes'] for r in eligible),
        excluded_count=len(excluded),git_metadata_excluded=True,includes_ignored_files=True,files=eligible))
    known=set(os.fsdecode(p) for p in git('ls-files','-z',capture=True).split(b'\0') if p)
    changed=set(os.fsdecode(p) for p in git('diff','--name-only','-z','HEAD',capture=True).split(b'\0') if p)
    candidates=[r for r in eligible if r['path'] not in known or r['path'] in changed]
    candidates += [dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size) for p in
        (STATE/'excluded_files.json',STATE/'snapshot_manifest.json') if str(p.relative_to(ROOT)) not in {r['path'] for r in candidates}]
    # The snapshot includes results changing during experiments as observed at add
    # time. Further generation after this snapshot is not claimed as uploaded.
    batch=[];size=0;count=0;uploaded=0
    for entry in candidates:
        if batch and (size+entry['bytes']>MAX_BATCH or len(batch)>=12000):
            stage(batch);count+=1
            commit(f'Back up workspace files within GitHub size limit (batch {count})')
            uploaded+=len(batch)
            print(json.dumps(dict(batch=count,files_processed=uploaded,total_candidates=len(candidates)),ensure_ascii=False),flush=True)
            batch=[];size=0
        if (ROOT/entry['path']).exists() or (ROOT/entry['path']).is_symlink():
            batch.append(entry['path']);size+=entry['bytes']
    if batch:
        stage(batch);count+=1;commit(f'Back up workspace files within GitHub size limit (batch {count})')
    tracked=set(os.fsdecode(p) for p in git('ls-files','-z',capture=True).split(b'\0') if p)
    missing=[r['path'] for r in eligible if r['path'] not in tracked]
    assert not missing,missing[:20]
    local=git('rev-parse','HEAD',capture=True).decode().strip()
    remote_head=git('ls-remote','origin','refs/heads/main',capture=True).decode().split()[0]
    assert local==remote_head
    write(STATE/'completed.json',dict(complete=True,snapshot_eligible_files=len(eligible),
        eligible_bytes=sum(r['bytes'] for r in eligible),excluded_files=len(excluded),batches=count,
        snapshot_commit=local,remote=remote,missing_snapshot_paths=missing,
        completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        scope='All eligible paths in snapshot are tracked and pushed; later live experiment writes are not part of this snapshot.'))
    stage([str((STATE/'completed.json').relative_to(ROOT))]);commit('Record verified workspace backup completion')
    print('BACKUP_COMPLETE',git('rev-parse','HEAD',capture=True).decode().strip(),flush=True)

if __name__=='__main__':
    main()

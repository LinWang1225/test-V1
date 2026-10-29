"""Copied tested namespace sandbox; official HumanEval execution runs inside it."""
import json,os,shutil,signal,subprocess,tempfile
from pathlib import Path

SANDBOX_WRAPPER = r'''
import ctypes
import json
import os
import resource
import sys
import traceback

ROOT = os.path.realpath(sys.argv[1])
TIMEOUT = max(1, int(sys.argv[2]))

MS_RDONLY  = 1
MS_NOSUID  = 2
MS_NODEV   = 4
MS_NOEXEC  = 8
MS_REMOUNT = 32
MS_BIND    = 4096
MS_REC     = 16384
MS_PRIVATE = 1 << 18
PR_SET_NO_NEW_PRIVS = 38

libc = ctypes.CDLL(None, use_errno=True)
libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                       ctypes.c_ulong, ctypes.c_char_p]
libc.mount.restype = ctypes.c_int
libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                       ctypes.c_ulong, ctypes.c_ulong]
libc.prctl.restype = ctypes.c_int

class CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

class CapData(ctypes.Structure):
    _fields_ = [("effective", ctypes.c_uint32),
                ("permitted", ctypes.c_uint32),
                ("inheritable", ctypes.c_uint32)]

def b(x):
    return None if x is None else os.fsencode(x)

def mount(src, dst, fstype=None, flags=0, data=None):
    rc = libc.mount(b(src), b(dst), b(fstype), flags, b(data))
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), dst)

def bind_ro(src, dst):
    if not os.path.exists(src):
        return
    mount(src, dst, None, MS_BIND | MS_REC, None)
    mount(None, dst, None,
          MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV, None)

def drop_caps():
    # Linux capability ABI v3: clear effective/permitted/inheritable sets.
    hdr = CapHeader(0x20080522, 0)
    data = (CapData * 2)()
    capset = libc.capset
    if capset(ctypes.byref(hdr), ctypes.byref(data)) != 0:
        err = ctypes.get_errno()
        raise OSError(err, "capset failed: " + os.strerror(err))
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, "PR_SET_NO_NEW_PRIVS failed: " + os.strerror(err))

def setup_root():
    # Nothing mounted below may propagate back to the host namespace.
    mount(None, "/", None, MS_REC | MS_PRIVATE, None)

    # Runtime is visible read-only. /home, /etc and project files are not exposed.
    for src in ("/usr", "/lib", "/lib64", "/bin"):
        if os.path.exists(src):
            dst = os.path.join(ROOT, src.lstrip("/"))
            os.makedirs(dst, exist_ok=True)
            bind_ro(src, dst)

    # Private writable /tmp only.
    tmp = os.path.join(ROOT, "tmp")
    os.makedirs(tmp, exist_ok=True)
    mount("tmpfs", tmp, "tmpfs",
          MS_NOSUID | MS_NODEV | MS_NOEXEC,
          "size=64m,mode=1777")

    os.chroot(ROOT)
    os.chdir("/tmp")

def apply_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT, TIMEOUT + 1))
    mem = 512 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    fsz = 2 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsz, fsz))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except (ValueError, OSError):
        pass
    os.umask(0o077)

def main():
    req = json.load(sys.stdin)
    setup_root()
    apply_limits()
    # Mount/chroot setup is complete; generated code receives no namespace caps.
    drop_caps()

    program = req["program"]
    try:
        glb = {"__name__": "__main__", "__file__": "<humaneval>"}
        exec(compile(program, "<humaneval>", "exec"), glb, glb)
        print(json.dumps({"passed": True}), flush=True)
    except BaseException as exc:
        print(json.dumps({
            "passed": False,
            "exception": type(exc).__name__,
            "error": str(exc)[-4096:],
            "traceback": traceback.format_exc()[-8192:],
        }), flush=True)

try:
    main()
except BaseException as exc:
    print("SANDBOX_SETUP_ERROR:" + repr(exc), file=sys.stderr, flush=True)
    raise SystemExit(125)
'''

def make_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix="kvbench-human-root-"))
    for d in ("usr", "lib", "lib64", "bin", "tmp"):
        (root / d).mkdir(exist_ok=True)
    # Generated code must not be able to create files at chroot '/'.
    root.chmod(0o555)
    return root

def cleanup_root(root: Path) -> None:
    try:
        root.chmod(0o755)
    except OSError:
        pass
    shutil.rmtree(root, ignore_errors=True)

def sandbox_cmd(python_bin: Path, root: Path, timeout: int) -> list[str]:
    return [
        "unshare",
        "--user", "--map-root-user",
        "--net", "--mount", "--ipc",
        "--pid", "--fork", "--uts",
        str(python_bin), "-I", "-B", "-c", SANDBOX_WRAPPER,
        str(root), str(timeout),
    ]

def sandbox_run(program: str, python_bin: Path, timeout: int) -> dict:
    root = make_root()
    cmd = sandbox_cmd(python_bin, root, timeout)
    env = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
    }
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(
            json.dumps({"program": program}), timeout=timeout + 8
        )
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
        cleanup_root(root)
        return {
            "passed": False,
            "timeout": True,
            "returncode": proc.returncode,
            "stderr": "outer wall-clock timeout",
        }
    finally:
        cleanup_root(root)

    if proc.returncode == 125 or "SANDBOX_SETUP_ERROR:" in stderr:
        raise RuntimeError("sandbox infrastructure error:\n" + stderr[-8000:])

    lines = [line for line in stdout.splitlines() if line.strip()]
    if lines:
        try:
            result = json.loads(lines[-1])
            result["returncode"] = proc.returncode
            if stderr:
                result["stderr"] = stderr[-8192:]
            return result
        except json.JSONDecodeError:
            pass

    # os._exit(), abort(), signal termination, etc. are task failures.
    return {
        "passed": False,
        "runtime_abnormal": True,
        "returncode": proc.returncode,
        "stdout": stdout[-8192:],
        "stderr": stderr[-8192:],
    }

def full_namespace_check() -> None:
    cmd = [
        "unshare",
        "--user", "--map-root-user",
        "--net", "--mount", "--ipc",
        "--pid", "--fork", "--uts",
        "true",
    ]
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError("required namespaces unavailable:\n" + p.stderr)

def sandbox_self_test(python_bin: Path) -> None:
    program = r'''
import os
import socket

assert 2 + 2 == 4
assert not os.path.exists("/home"), "host /home is visible"

try:
    with open("/should_not_write", "w") as f:
        f.write("x")
    raise AssertionError("chroot root is writable")
except OSError:
    pass

s = socket.socket()
s.settimeout(0.2)
try:
    s.connect(("1.1.1.1", 80))
    raise AssertionError("external network reachable")
except OSError:
    pass
finally:
    s.close()
'''
    r = sandbox_run(program, python_bin, 3)
    if not r.get("passed"):
        raise RuntimeError(
            "sandbox self-test failed:\n" +
            json.dumps(r, indent=2, ensure_ascii=False)
        )
    print("sandbox self-test: PASS")

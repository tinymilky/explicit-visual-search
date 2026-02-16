#!/usr/bin/env python3
# parallel_extract_split_tar.py
#
# Parallel extraction of a split TAR archive (cot_images_00, cot_images_01, ...)
# without `cat` and without creating an intermediate .tar.
#
# Works best on fast local SSD/NVMe. On HDD/NFS it may be slower than plain `tar`.
#
# Install:
#   pip install -U tqdm
#
# Usage:
#   python parallel_extract_split_tar.py \
#     --in_dir /data/cot_images_tar_split \
#     --pattern "cot_images_*" \
#     --out_dir /data/out \
#     --workers 8 \
#     --strip_components 0 \
#     --no_metadata
#
import argparse
import glob
import os
import sys
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from tqdm import tqdm

BLOCK = 512


def roundup_512(n: int) -> int:
    return (n + BLOCK - 1) // BLOCK * BLOCK


def is_all_zero(b: bytes) -> bool:
    return all(x == 0 for x in b)


def parse_octal(field: bytes) -> int:
    field = field.split(b"\0", 1)[0].strip()  # strip NUL and spaces
    if not field:
        return 0
    try:
        return int(field, 8)
    except ValueError:
        return 0


def parse_tar_header(hdr: bytes) -> Dict:
    # USTAR header layout
    name = hdr[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
    mode = parse_octal(hdr[100:108])
    uid = parse_octal(hdr[108:116])
    gid = parse_octal(hdr[116:124])
    size = parse_octal(hdr[124:136])
    mtime = parse_octal(hdr[136:148])
    typeflag = hdr[156:157]  # b'0', b'5', b'2', b'1', b'L', b'x', ...
    linkname = hdr[157:257].split(b"\0", 1)[0].decode("utf-8", "replace")
    magic = hdr[257:263]
    prefix = hdr[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")

    path = name
    if prefix:
        path = f"{prefix}/{name}" if name else prefix

    return {
        "path": path,
        "mode": mode,
        "uid": uid,
        "gid": gid,
        "size": size,
        "mtime": mtime,
        "typeflag": typeflag,
        "linkname": linkname,
        "magic": magic,
    }


def parse_pax_records(data: bytes) -> Dict[str, str]:
    # PAX format: lines like "25 path=some/thing\n"
    out = {}
    i = 0
    n = len(data)
    while i < n:
        # read length prefix
        j = data.find(b" ", i)
        if j < 0:
            break
        try:
            length = int(data[i:j])
        except ValueError:
            break
        rec = data[j + 1 : i + length]
        # rec ends with '\n'
        eq = rec.find(b"=")
        if eq > 0:
            key = rec[:eq].decode("utf-8", "replace")
            val = rec[eq + 1 :].rstrip(b"\n").decode("utf-8", "replace")
            out[key] = val
        i += length
    return out


class MultiPartPread:
    """
    Provides pread-like reads over multiple concatenated part files without concatenating them.
    Each worker creates its own instance (no shared state).
    """

    def __init__(self, part_paths: List[str]):
        self.part_paths = part_paths
        self.fds: List[int] = []
        self.sizes: List[int] = []
        self.cum: List[int] = [0]  # cum[i] = start offset of part i
        total = 0
        for p in part_paths:
            st = os.stat(p)
            self.sizes.append(st.st_size)
            total += st.st_size
            self.cum.append(total)
        self.total_size = total
        for p in part_paths:
            self.fds.append(os.open(p, os.O_RDONLY))

    def close(self):
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds = []

    def _locate(self, offset: int) -> Tuple[int, int]:
        # find part idx such that cum[idx] <= offset < cum[idx+1]
        # linear scan is OK for ~dozens of parts; if hundreds, use bisect.
        if offset < 0 or offset >= self.total_size:
            raise EOFError
        lo = 0
        hi = len(self.cum) - 1
        # binary search
        import bisect
        idx = bisect.bisect_right(self.cum, offset) - 1
        in_part = offset - self.cum[idx]
        return idx, in_part

    def pread(self, offset: int, n: int) -> bytes:
        if n <= 0:
            return b""
        if offset >= self.total_size:
            return b""
        to_read = min(n, self.total_size - offset)
        out = bytearray()
        off = offset
        remain = to_read
        while remain > 0:
            idx, in_part = self._locate(off)
            can = min(remain, self.sizes[idx] - in_part)
            chunk = os.pread(self.fds[idx], can, in_part)
            if not chunk:
                break
            out.extend(chunk)
            got = len(chunk)
            off += got
            remain -= got
        return bytes(out)


@dataclass
class Task:
    kind: str            # "file", "dir", "symlink", "hardlink"
    path: str
    data_offset: int     # for files
    size: int            # for files
    mode: int
    mtime: int
    linkname: str = ""


def safe_join(out_dir: str, rel_path: str) -> str:
    out_dir_abs = os.path.abspath(out_dir)
    rel_path = rel_path.lstrip("/")  # do not allow absolute paths
    target = os.path.abspath(os.path.join(out_dir_abs, rel_path))
    if not (target == out_dir_abs or target.startswith(out_dir_abs + os.sep)):
        raise RuntimeError(f"Blocked path traversal: {rel_path!r}")
    return target


def producer_scan(parts: List[str], out_dir: str, strip_components: int, q: mp.Queue, progress: bool):
    r = MultiPartPread(parts)
    try:
        off = 0
        zero_blocks = 0

        pending_longname: Optional[str] = None
        pending_pax: Dict[str, str] = {}

        pbar = None
        if progress:
            pbar = tqdm(desc="Scanning headers", unit="hdr")

        while True:
            hdr = r.pread(off, BLOCK)
            if len(hdr) < BLOCK:
                break

            if is_all_zero(hdr):
                zero_blocks += 1
                off += BLOCK
                if zero_blocks >= 2:
                    break
                continue
            zero_blocks = 0

            h = parse_tar_header(hdr)
            size = h["size"]
            typeflag = h["typeflag"]

            data_off = off + BLOCK

            # read special records if needed
            if typeflag == b"L":  # GNU longname
                name_bytes = r.pread(data_off, size)
                pending_longname = name_bytes.split(b"\0", 1)[0].decode("utf-8", "replace").rstrip("\n")
                off = data_off + roundup_512(size)
                if pbar: pbar.update(1)
                continue

            if typeflag == b"x":  # PAX extended header
                pax_data = r.pread(data_off, size)
                pending_pax = parse_pax_records(pax_data)
                off = data_off + roundup_512(size)
                if pbar: pbar.update(1)
                continue

            path = h["path"]
            linkname = h["linkname"]

            # apply GNU longname / PAX overrides
            if pending_longname:
                path = pending_longname
                pending_longname = None
            if pending_pax:
                if "path" in pending_pax:
                    path = pending_pax["path"]
                if "linkpath" in pending_pax:
                    linkname = pending_pax["linkpath"]
                pending_pax = {}

            # strip components
            if strip_components > 0:
                comps = path.split("/")
                if len(comps) <= strip_components:
                    # skip entry that would become empty
                    off = data_off + roundup_512(size)
                    if pbar: pbar.update(1)
                    continue
                path = "/".join(comps[strip_components:])

            # enqueue tasks
            if typeflag in (b"0", b"\0"):  # regular file
                q.put(Task("file", path, data_off, size, h["mode"], h["mtime"], ""))
            elif typeflag == b"5":  # directory
                q.put(Task("dir", path.rstrip("/") + "/", 0, 0, h["mode"], h["mtime"], ""))
            elif typeflag == b"2":  # symlink
                q.put(Task("symlink", path, 0, 0, h["mode"], h["mtime"], linkname))
            elif typeflag == b"1":  # hard link
                q.put(Task("hardlink", path, 0, 0, h["mode"], h["mtime"], linkname))
            else:
                # ignore other types for simplicity
                pass

            # advance to next header
            off = data_off + roundup_512(size)

            if pbar: pbar.update(1)

        if pbar:
            pbar.close()
    finally:
        # send stop signals
        for _ in range(mp.cpu_count() * 2):
            q.put(None)
        r.close()


def worker_extract(worker_id: int, parts: List[str], out_dir: str, q: mp.Queue,
                   chunk_bytes: int, no_metadata: bool, stats_q: mp.Queue):
    r = MultiPartPread(parts)
    extracted = 0
    skipped = 0
    failed = 0
    try:
        while True:
            task = q.get()
            if task is None:
                break

            try:
                if task.kind == "dir":
                    dst = safe_join(out_dir, task.path)
                    os.makedirs(dst, exist_ok=True)

                elif task.kind == "symlink":
                    dst = safe_join(out_dir, task.path)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    try:
                        if os.path.lexists(dst):
                            # don't overwrite existing
                            skipped += 1
                        else:
                            os.symlink(task.linkname, dst)
                            extracted += 1
                    except OSError:
                        failed += 1

                elif task.kind == "hardlink":
                    dst = safe_join(out_dir, task.path)
                    src = safe_join(out_dir, task.linkname)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    try:
                        if os.path.exists(dst):
                            skipped += 1
                        else:
                            os.link(src, dst)
                            extracted += 1
                    except OSError:
                        failed += 1

                elif task.kind == "file":
                    dst = safe_join(out_dir, task.path)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)

                    if os.path.exists(dst) and os.path.getsize(dst) == task.size:
                        skipped += 1
                        continue

                    tmp = dst + f".tmp.{worker_id}"
                    with open(tmp, "wb") as f:
                        remaining = task.size
                        off = task.data_offset
                        while remaining > 0:
                            n = min(chunk_bytes, remaining)
                            buf = r.pread(off, n)
                            if not buf:
                                raise IOError("Unexpected EOF while reading payload")
                            f.write(buf)
                            got = len(buf)
                            off += got
                            remaining -= got

                    os.replace(tmp, dst)

                    if not no_metadata:
                        try:
                            os.chmod(dst, task.mode & 0o7777)
                        except OSError:
                            pass
                        try:
                            os.utime(dst, (task.mtime, task.mtime))
                        except OSError:
                            pass

                    extracted += 1

            except Exception:
                failed += 1

            # periodically report
            if (extracted + skipped + failed) % 2000 == 0:
                stats_q.put((extracted, skipped, failed))

    finally:
        r.close()
        stats_q.put((extracted, skipped, failed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--pattern", required=True, help='e.g. "cot_images_*"')
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--queue_size", type=int, default=20000)
    ap.add_argument("--chunk_mb", type=int, default=4)
    ap.add_argument("--strip_components", type=int, default=0)
    ap.add_argument("--no_metadata", action="store_true", help="Skip chmod/utime for speed")
    ap.add_argument("--progress", action="store_true", help="Show scanning progress + extraction stats")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    parts = sorted(glob.glob(os.path.join(in_dir, args.pattern)))
    if not parts:
        print(f"No parts matched: {os.path.join(in_dir, args.pattern)}", file=sys.stderr)
        sys.exit(1)

    ctx = mp.get_context("fork")  # Linux; fastest. If not available, switch to "spawn".
    q = ctx.Queue(maxsize=args.queue_size)
    stats_q = ctx.Queue()

    # start workers
    workers = []
    for wid in range(args.workers):
        p = ctx.Process(
            target=worker_extract,
            args=(wid, parts, out_dir, q, args.chunk_mb * 1024 * 1024, args.no_metadata, stats_q),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # start producer
    prod = ctx.Process(
        target=producer_scan,
        args=(parts, out_dir, args.strip_components, q, args.progress),
        daemon=True,
    )
    prod.start()

    # monitor
    if args.progress:
        pbar = tqdm(desc="Extracting (reported)", unit="files")
        last = (0, 0, 0)
        alive = True
        while alive:
            alive = prod.is_alive() or any(p.is_alive() for p in workers)
            try:
                while True:
                    ex, sk, fa = stats_q.get_nowait()
                    last = (ex, sk, fa)
                    pbar.set_postfix({"ok": ex, "skip": sk, "fail": fa})
                    pbar.update(0)
            except Exception:
                pass
            time.sleep(0.2)
        pbar.close()

    prod.join()
    for p in workers:
        p.join()

    # final drain
    ok = sk = fa = 0
    try:
        while True:
            ok, sk, fa = stats_q.get_nowait()
    except Exception:
        pass

    print(f"Done. ok={ok} skip={sk} fail={fa} -> {out_dir}")


if __name__ == "__main__":
    main()

'''
python extract_split_tar.py --in_dir /mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/datasets_to_train/visualcot/cot_images_tar_split --pattern "cot_images_*" --out_dir ./vis_cot --progress

python parallel_extract_split_tar.py \
--in_dir /mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/datasets_to_train/visualcot/cot_images_tar_split \
--pattern "cot_images_*" \
--out_dir ./vis_cot \
--workers 16 \
--strip_components 0 \
--no_metadata
'''
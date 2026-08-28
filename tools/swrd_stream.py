"""Stream members out of the 115GB SWRD zip on Google Drive via HTTP ranges."""
import io, json, os, re, sys, urllib.request, functools, collections, zipfile

FID = "1Hc9_de5YAdXg46F-GfjcRKMEBRF9XRQi"
URL = f"https://drive.usercontent.google.com/download?id={FID}&export=download&confirm=t"
BLOCK = 4 << 20  # 4MB blocks

class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])
        self.fetched = 0
    @functools.lru_cache(maxsize=64)
    def _block(self, i):
        import time, random
        lo = i * BLOCK; hi = min(self.size, lo + BLOCK) - 1
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    if "text/html" in (r.headers.get("Content-Type") or ""):
                        raise IOError("quota page")      # Drive throttle serves HTML
                    data = r.read()
                self.fetched += len(data)
                return data
            except Exception:
                if attempt == 5: raise
                time.sleep(2 ** attempt + random.random() * 3)
    def readable(self): return True
    def seekable(self): return True
    def seek(self, off, whence=0):
        self.pos = [off, self.pos + off, self.size + off][whence]
        return self.pos
    def tell(self): return self.pos
    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, o = divmod(self.pos, BLOCK)
            chunk = self._block(i)[o:o + n]
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return bytes(out)

if __name__ == "__main__":
    f = HttpFile(URL)
    print(f"file size: {f.size/1e9:.1f} GB", flush=True)
    z = zipfile.ZipFile(f)
    names = z.namelist()
    print(f"{len(names)} entries, central dir fetched with {f.fetched/1e6:.0f} MB")
    tops = collections.Counter(n.split("/")[0] for n in names)
    print("top-level:", dict(tops))
    exts = collections.Counter(os.path.splitext(n)[1].lower() for n in names)
    print("extensions:", dict(exts))
    # peek at a few paths per depth to reveal structure
    seen = set()
    for n in names:
        key = "/".join(n.split("/")[:2])
        if key not in seen:
            seen.add(key); print(" ", n)
        if len(seen) > 25: break
    json.dump(names, open("swrd_names.json", "w"))
    print("namelist saved -> swrd_names.json")

import io, zipfile, requests
URL = "https://zenodo.org/api/records/15340689/files/PT_DATASET_250314.zip/content"

class HTTPFile(io.RawIOBase):
    """Read-only file object that fetches bytes from a URL with Range requests."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        r = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True)
        self.size = int(r.headers["Content-Range"].split("/")[1]); r.close()
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def read(self, n=-1):
        end = self.size - 1 if n < 0 else min(self.pos + n - 1, self.size - 1)
        if self.pos > end: return b""
        r = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        self.pos += len(r.content); return r.content

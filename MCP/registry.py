# registry.py
from pycrdt import Doc
from jupyter_ydoc.ynotebook import YNotebook
import json

class NotebookRegistry:
    def __init__(self):
        self._docs = {}

    def get_or_load(self, path: str) -> YNotebook:
        if path not in self._docs:
            doc = Doc()
            nb = YNotebook(doc)
            with open(path, "r") as f:
                nb_json = json.load(f)
                nb.set(nb_json)
            self._docs[path] = nb
        return self._docs[path]

    def save(self, path: str) -> None:
        if path in self._docs:
            ynotebook = self._docs[path]
            with open(path, "w") as f:
                json.dump(ynotebook.get(), f, indent=2)

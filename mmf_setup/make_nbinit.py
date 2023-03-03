import os.path
from jinja2 import Template

_THEME_DIR = os.path.join("mmf_setup", "_data", "nbthemes")

with open("nbinit_tmpl.py") as f:
    t = Template(f.read())

_THEME = "mmf"
args = {}

for _type in ["css", "js", "html"]:
    with open(os.path.join(_THEME_DIR, ".".join([_THEME, _type]))) as f:
        args["theme_" + _type] = f.read()

with open("nbinit.py", "w") as f:
    f.write(t.render(**args))

try:
    import mmf_setup

    mmf_setup.nbinit()
except ImportError:
    import subprocess
    import sys
    from IPython.display import HTML, Javascript, display

    display(HTML(r"""<style>{{ theme_css }}</style>"""))
    display(Javascript(r"""{{ theme_js }}"""))
    display(HTML(r"""{{ theme_html }}"""))

    try:
        HGROOT = subprocess.check_output(["hg", "root"]).strip()
        if HGROOT not in sys.path:
            sys.path.insert(0, HGROOT)
    except subprocess.CalledProcessError:
        # Could not run hg or not in a repo.
        pass

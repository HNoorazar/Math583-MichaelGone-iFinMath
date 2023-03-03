import configparser
import os.path
import shlex
import sys


def set_path_from_file(filename="setup.cfg", ROOT=None, check=True):
    """Set the path from the `paths` variable in the [mmf_setup] section of the
    specified config file.  Each path entry should be on a separate line.
    I.e.::

        [mmf_setup]
        paths = .          # ROOT not included by default!  Add it here.
                src        # Paths relative to ROOT
                /abs/path  # Absolute paths are okay too

    Note: if you specify the paths this way, you need to explicitly include the
    root directory `.` as shown above.  All paths are relative to
    ROOT (defaults to `mmf_setup.ROOT`) unless specified as an absolute path.

    Arguments
    ---------
    filename : str
       Name of config file (default is 'setup.cfg').  If relative, then we look
       in `ROOT`.
    ROOT : str, None
       Root directory used for all relative paths.  If `None`, then we use
       `mmf_setup.ROOT`.
    check : bool
       If `True`, then only add paths if they actually exist.
    """
    if ROOT is None:
        import mmf_setup

        ROOT = getattr(mmf_setup, "ROOT", ".")

    # Now add any paths specified in system.cfg
    config = configparser.ConfigParser()
    if not os.path.isabs(filename):
        filename = os.path.join(ROOT, filename)

    config.read(filename)

    # Default includes only ROOT
    paths = [ROOT]

    if config.has_section("mmf_setup"):
        if config.has_option("mmf_setup", "paths"):
            # Remove default ROOT if this section is specified to allow the
            # user to NOT include ROOT if desired.
            paths = []

            for line in config.get("mmf_setup", "paths").split("\n"):
                # Strips out comments etc.  See
                # https://stackoverflow.com/a/27178714/1088938
                lex = shlex.shlex(line)
                lex.whitespace = ""  # if you want to strip newlines, use '\n'
                line = "".join(list(lex))
                if line:
                    paths.append(line.strip())

    for path in reversed(paths):
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)

        path = os.path.abspath(path)
        if (not check or os.path.exists(path)) and path not in sys.path:
            sys.path.insert(0, path)

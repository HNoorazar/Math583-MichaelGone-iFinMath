try:
    import mmf_setup

    mmf_setup.nbinit()
except ImportError:
    import subprocess
    import sys
    from IPython.display import HTML, Javascript, display

    display(
        HTML(
            r"""
<style>
.grade {
   background-color: #66FFCC;
}
</style>

<script id="MathJax-Element-48" type="math/tex">
\newcommand{\vect}[1]{\mathbf{#1}}
\newcommand{\uvect}[1]{\hat{#1}}
\newcommand{\abs}[1]{\lvert#1\rvert}
\newcommand{\norm}[1]{\lVert#1\rVert}
\newcommand{\I}{\mathrm{i}}
\newcommand{\ket}[1]{\left|#1\right\rangle}
\newcommand{\bra}[1]{\left\langle#1\right|}
\newcommand{\braket}[1]{\langle#1\rangle}
\newcommand{\Braket}[1]{\left\langle#1\right\rangle}
\newcommand{\op}[1]{\mathbf{#1}}
\newcommand{\mat}[1]{\mathbf{#1}}
\newcommand{\d}{\mathrm{d}}
\newcommand{\D}[1]{\mathcal{D}[#1]\;}
\newcommand{\pdiff}[3][]{\frac{\partial^{#1} #2}{\partial {#3}^{#1}}}
\newcommand{\diff}[3][]{\frac{\d^{#1} #2}{\d {#3}^{#1}}}
\newcommand{\ddiff}[3][]{\frac{\delta^{#1} #2}{\delta {#3}^{#1}}}
\newcommand{\floor}[1]{\left\lfloor#1\right\rfloor}
\newcommand{\ceil}[1]{\left\lceil#1\right\rceil}
\DeclareMathOperator{\Tr}{Tr}
\DeclareMathOperator{\erf}{erf}
\DeclareMathOperator{\erfi}{erfi}
\DeclareMathOperator{\sech}{sech}
\DeclareMathOperator{\sn}{sn}
\DeclareMathOperator{\cn}{cn}
\DeclareMathOperator{\dn}{dn}
\DeclareMathOperator{\sgn}{sgn}
\DeclareMathOperator{\order}{O}
\DeclareMathOperator{\diag}{diag}

\newcommand{\mylabel}[1]{\label{#1}\tag{#1}}
\newcommand{\degree}{\circ}
</script>
<i>
<p>This cell contains some definitions for equations and some CSS for styling
    the notebook.  If things look a bit strange, please try the following:
<ul>
  <li>Choose "Trust Notebook" from the "File" menu.</li>
  <li>Re-execute this cell.</li>
  <li>Reload the notebook.</li>
</ul>
</p>
</i>
"""
        )
    )

    try:
        HGROOT = subprocess.check_output(["hg", "root"]).strip()
        if HGROOT not in sys.path:
            sys.path.insert(0, HGROOT)
    except subprocess.CalledProcessError:
        # Could not run hg or not in a repo.
        pass

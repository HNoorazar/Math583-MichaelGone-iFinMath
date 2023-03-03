import nox


@nox.session(venv_backend="conda", python=["3.6", "3.7", "3.8", "3.9"])
def test_conda(session):
    session.install(".[test]")
    session.run("pytest")
    session.run("make", "test", external=True)

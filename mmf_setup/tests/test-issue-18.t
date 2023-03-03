  $ cat <<EOF >> $HGRCPATH
  > [ui]
  > ignore = $TESTDIR/hgignore  
  > %include $TESTDIR/hgrc
  > EOF

  $ hg init
  $ touch A
  $ hg add A
  $ hg com -m "Added A"
  $ echo "A" >> A
  $ cat <<EOF >> .hg/hgrc
  > [hooks]
  > pre-commit = false
  > EOF
  $ hg ccom -m "Modified A"
  Aborting... could not create checkpoint commmit!
  running hook pre-commit: false
  abort: pre-commit hook exited with status 1
  [255]
  $ hg lg
  @  0:d test Added A (1970-01-01)  tip
  

If the commit failed, then the modified files should still be there.
Issue #18 was that the file was removed.

  $ hg st
  M A
  $ cat A
  A

# Another possible result (if pre-commit hooks are ignored) is:
$ hg lg
@  1:d test Modified A (1970-01-01)  tip
|
o  0:d test Added A (1970-01-01)




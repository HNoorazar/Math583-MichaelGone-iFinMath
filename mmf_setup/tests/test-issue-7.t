  $ cat <<EOF >> $HGRCPATH
  > [ui]
  > ignore = $TESTDIR/hgignore  
  > %include $TESTDIR/hgrc
  > EOF

  $ hg init
  $ cp ${TESTDIR}/_data/n1_dirty.ipynb N.ipynb
  $ hg add N.ipynb
  $ hg com -m "Dirty N"
  $ hg glog
  @  0: test Dirty N (1970-01-01)   tip
  
  $ hg up -C 0
  0 files updated, 0 files merged, 0 files removed, 0 files unresolved
  $ hg cst
  cleaning output
  restoring output
  $ hg cst --clean-all
  cleaning output
  cleaning N.ipynb
  M N.ipynb
  restoring output

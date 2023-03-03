  $ cat <<EOF >> $HGRCPATH
  > %include $TESTDIR/hgrc
  > EOF

  $ hg init
  $ touch A
  $ touch B
  $ hg add A B
  $ hg ccom A B -m "Added A and B"
  cleaning output
  created new head
  no output to commit
  restoring output
  $ hg glog
  @  0: test Added A and B (1970-01-01)   tip
  

  $ cat <<EOF >> $HGRCPATH
  > %include $TESTDIR/hgrc
  > EOF

  $ hg init
  $ touch A
  $ touch B
  $ hg add A B
  $ hg ccom A B
  cleaning output
  restoring output
  abort: empty commit message
  [255]
  $ hg glog
  $ hg bookmarks
  no bookmarks set

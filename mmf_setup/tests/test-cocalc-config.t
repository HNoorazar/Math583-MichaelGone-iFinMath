  $ export PATH="$RUNTESTDIR/../bin/:$PATH"
  $ mmf_initial_setup --help
  Usage: mmf_initial_setup [options] dir1 dir2 ...
  
  Creates symlinks to files in the specified config directories (default "./").
  
  Options:
    -h, --help         show this help message and exit
    --home=<home>      use <home> rather than ~ for installation.(Used to
                       replace '~' in dest strings.)
    -v, --verbose      print lots of information
    -n, --no-action    don't do anything:only print commands that would be
                       executed
    -i, --interactive  prompt before taking action
    -a, --abs-path     Use absolute symlinks (defaults are relative to ~)
  
  Run from the desired config directory, and files with a second line like
  "dest = ~/.bashrc"  will be symlinked to the specified location.  If a file
  already exists, it will be backed up (copied to a file with a .bak extension).
  Existing symlinks will be overwritten.
  $ mmf_setup --help
  usage: source mmf_setup [-v]  OR  . mmf_setup [-v]  OR  mmf_setup cocalc
  
    -v : show variables that are set
  
  For initial setup on cocalc.com projects:
  
     mmf_setup cocalc
  
  [1]
  $ mkdir home
  $ touch home/.bash_aliases
  $ mmf_setup cocalc -n --home="./home"
  Installing hg-evolve for hg=*/bin/hg... (glob)
  */python -m pip install --user hg-evolve (glob)
  * (glob)
  Setting up config files for CoCalc...
  Using <home> = ./home
  Using dir = */site-packages/mmf_setup/_data/config_files/cocalc (glob)
  File ./home/.bash_aliases exists.
  backup('./home/.bash_aliases')
  os.symlink('*/config_files/cocalc/bash_aliases', './home/.bash_aliases') (glob)
  os.symlink('*/config_files/cocalc/mrconfig', './home/.mrconfig') (glob)
  os.symlink('*/config_files/cocalc/hgrc', './home/.hgrc') (glob)
  os.symlink('*/config_files/cocalc/inputrc', './home/.inputrc') (glob)
  os.symlink('*/config_files/cocalc/hgignore', './home/.hgignore') (glob)
  Configurations for your CoCalc project have been symlinked as described above.
  
  If you use version control, then to get the most of the configuration,
  please make sure that you set the following variables on your personal
  computer, and forward them when you ssh to the project:
  
      # ~/.bashrc or similar
      LC_HG_USERNAME=Your Full Name <your.email.address+hg@gmail.com>
      LC_GIT_USEREMAIL=your.email.address+git@gmail.com
      LC_GIT_USERNAME=Your Full Name
  
  To forward these, your SSH config file (~/.ssh/config) might look like:
  
      # ~/.ssh/config
      Host cc-project1
        User ff1cb986f...
      
      Host cc*
        HostName ssh.cocalc.com
        ForwardAgent yes
        SendEnv LC_HG_USERNAME
        SendEnv LC_GIT_USERNAME
        SendEnv LC_GIT_USEREMAIL
        SetEnv LC_EDITOR=vi
  $ mmf_setup cocalc --home="./home"
  Installing hg-evolve for hg=*/bin/hg... (glob)
  */bin/python -m pip install --user hg-evolve (glob)
  * (glob)
  Setting up config files for CoCalc...
  Using <home> = ./home
  Using dir = */site-packages/mmf_setup/_data/config_files/cocalc (glob)
  File ./home/.bash_aliases exists.
  backup('./home/.bash_aliases')
  os.rename('./home/.bash_aliases', './home/.bash_aliases.bak')
  os.symlink('*/config_files/cocalc/bash_aliases', './home/.bash_aliases') (glob)
  os.symlink('*/config_files/cocalc/mrconfig', './home/.mrconfig') (glob)
  os.symlink('*/config_files/cocalc/hgrc', './home/.hgrc') (glob)
  os.symlink('*/config_files/cocalc/inputrc', './home/.inputrc') (glob)
  os.symlink('*/config_files/cocalc/hgignore', './home/.hgignore') (glob)
  Configurations for your CoCalc project have been symlinked as described above.
  
  If you use version control, then to get the most of the configuration,
  please make sure that you set the following variables on your personal
  computer, and forward them when you ssh to the project:
  
      # ~/.bashrc or similar
      LC_HG_USERNAME=Your Full Name <your.email.address+hg@gmail.com>
      LC_GIT_USEREMAIL=your.email.address+git@gmail.com
      LC_GIT_USERNAME=Your Full Name
  
  To forward these, your SSH config file (~/.ssh/config) might look like:
  
      # ~/.ssh/config
      Host cc-project1
        User ff1cb986f...
      
      Host cc*
        HostName ssh.cocalc.com
        ForwardAgent yes
        SendEnv LC_HG_USERNAME
        SendEnv LC_GIT_USERNAME
        SendEnv LC_GIT_USEREMAIL
        SetEnv LC_EDITOR=vi
  $ ls -aF ./home
  ./
  ../
  .bash_aliases@
  .bash_aliases.bak
  .hgignore@
  .hgrc@
  .inputrc@
  .mrconfig@
  $ mmf_setup cocalc -v --home="./home"
  Installing hg-evolve for hg=*/bin/hg... (glob)
  */bin/python -m pip install --user hg-evolve (glob)
  * (glob)
  Setting up config files for CoCalc...
  Using <home> = ./home
  Using dir = */mmf_setup/_data/config_files/cocalc (glob)
  Symlink ./home/.bash_aliases exists.
  os.remove('./home/.bash_aliases')
  os.symlink('*/config_files/cocalc/bash_aliases', './home/.bash_aliases') (glob)
  Symlink ./home/.mrconfig exists.
  os.remove('./home/.mrconfig')
  os.symlink('*/config_files/cocalc/mrconfig', './home/.mrconfig') (glob)
  Symlink ./home/.hgrc exists.
  os.remove('./home/.hgrc')
  os.symlink('*/config_files/cocalc/hgrc', './home/.hgrc') (glob)
  Symlink ./home/.inputrc exists.
  os.remove('./home/.inputrc')
  os.symlink('*/config_files/cocalc/inputrc', './home/.inputrc') (glob)
  Symlink ./home/.hgignore exists.
  os.remove('./home/.hgignore')
  os.symlink('*/config_files/cocalc/hgignore', './home/.hgignore') (glob)
  Configurations for your CoCalc project have been symlinked as described above.
  
  If you use version control, then to get the most of the configuration,
  please make sure that you set the following variables on your personal
  computer, and forward them when you ssh to the project:
  
      # ~/.bashrc or similar
      LC_HG_USERNAME=Your Full Name <your.email.address+hg@gmail.com>
      LC_GIT_USEREMAIL=your.email.address+git@gmail.com
      LC_GIT_USERNAME=Your Full Name
  
  To forward these, your SSH config file (~/.ssh/config) might look like:
  
      # ~/.ssh/config
      Host cc-project1
        User ff1cb986f...
      
      Host cc*
        HostName ssh.cocalc.com
        ForwardAgent yes
        SendEnv LC_HG_USERNAME
        SendEnv LC_GIT_USERNAME
        SendEnv LC_GIT_USEREMAIL
        SetEnv LC_EDITOR=vi
  $ ls -aF ./home
  ./
  ../
  .bash_aliases@
  .bash_aliases.bak
  .hgignore@
  .hgrc@
  .inputrc@
  .mrconfig@

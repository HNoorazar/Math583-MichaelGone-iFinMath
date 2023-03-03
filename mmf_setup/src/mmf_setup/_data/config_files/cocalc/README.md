# Configuration on CoCalc

This directory contains files for setting up a project on CoCalc
(cocalc.com).  They are based on the following assumptions:

1. That the default `.bashrc` file sources `.bash_aliases`.  We put all of our
   customizations in this latter file.  This is [the default on
   CoCalc](https://doc.cocalc.com/howto/software-development.html?highlight=bash_aliases#is-bashrc-or-bash-profile-called-on-startup).
   
The following features are provided:

* **Terminal command history completion support.** Typing a few
  characters then using the up and down arrows, one should be able to
  scroll through previous terminal commands that have been issued
  starting with these characters.  This is done by providing an
  appropriate [`~/.inputrc`](inputrc) file.  On average, my commands
  are less than 20 characters long, more typically 13 characters.
  Thus, 500000 lines would take somewhere around 6MB.  This allows us
  to have a long history, which we set using `HISTFILESIZE=100000`.
* **Autocompletion.** We have added shell tab-completion for the
  following commands: `pip`, 
* **Mercurial configuration.**  In addition to what is provided by
  running the `mmf_setup` script, a global [`.hgignore`](hgignore) is
  provided with CoCalc-specific ignores and the mercurial username is
  set from the environmental variable `LC_HG_USERNAME`.  For this to
  function, you should set this on your personal computer and forward
  it in your `.ssh/config` file with something like:
  
  ```
  # ~/.ssh/config
  ...
  Host cocalc*
    HostName ssh.cocalc.com
    ForwardAgent yes
    SendEnv LC_HG_USERNAME
    SendEnv LC_GIT_USERNAME
    SendEnv LC_GIT_USEREMAIL
  ```

  The `LC_GIT_USER*` variables perform a similar function for `git`
  but are set using git itself when `mmf_setup` is run.


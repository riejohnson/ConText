  #***  Global variables  ***
  prep_exe=../bin/prepText
  exe=../bin/reNet

  tmpdir=temp
  csvdir=csv
  logdir=log
  outdir=out
  lddir=large-data
  if [ ! -e $tmpdir ]; then mkdir $tmpdir; fi
  if [ ! -e $csvdir ]; then mkdir $csvdir; fi
  if [ ! -e $logdir ]; then mkdir $logdir; fi
  if [ ! -e $outdir ]; then mkdir $outdir; fi

  shnm=$(basename $0)
  url=riejohnson.com/software/gz
  _cmd_=()  # Global variable for passing commands to do_jobs. 
  

  #***  Functions  ***
  #---  Download a file
  find_file () {
    local fnm=file_download
    local _dir=$1
    local _fn=$2
    if [ "$_dir" = "" ]; then echo $fnm: _dir is missing; return 1; fi    
    if [ "$_fn" = "" ]; then echo $fnm: _fn is missing; return 1; fi
    if [ -e "${_dir}/${_fn}" ]; then echo ${_dir}/${_fn} exists.; return 0; fi
    
    if [ "$_dir" != "for-semi" ]; then echo ${_dir}/${_fn} does not exist.  Generate it first.; return 1; fi

    #---  If directory is "for-semi", try downloading the file. 
    local _gzdir=gz

    if [ ! -e $_gzdir ]; then mkdir $_gzdir; fi
    local _fn=${_fn}.tar.gz
    rm -f ${_gzdir}/${_fn} # remove residue if any 
    echo $fnm: Downloading $_fn ... 
    wget --directory-prefix=${_gzdir} ${url}/${_fn} # download the file 
    if [ $? != 0 ]; then echo wget ${url}/${_fn} failed.; return 1; fi
    tar -xvf ${_gzdir}/${_fn}
    if [ $? != 0 ]; then tar -xvf ${_gzdir}/${_fn} failed.; return 1; fi
    rm -f ${_gzdir}/${_fn} # clean up ... 
    return 0
  }

  #---  Create and wait for background processes 
  #---  Input: global variable _cmd_ (array)
  do_jobs () {
    local fnm=do_jobs
    #---  parameter check 
    local _max_jobs=$1; local _num=$2; local _logfn=$3
    if [ "$_max_jobs" = "" ]; then echo $fnm: 1st arg is missing.; return 1; fi
    if [ "$_num"      = "" ]; then echo $fnm: 2nd arg is missing.; return 1; fi
    if [ "$_logfn"    = "" ]; then echo $fnm: 3rd arg is missing.; return 1; fi # used only if parallel
    local i; for (( i=0; i<_num; i++ )); do
      local cmd=${_cmd_[i]}
      if [ "$cmd" = "" ]; then echo $fnm: $i-th cmd is missing.; return 1; fi
    done

    #---  serial processing
    if [ $_max_jobs -le 1 ] || [ $_num -le 1 ]; then
      echo $fnm: serial ... 
      local i; for (( i=0; i<_num; i++ )); do
        ${_cmd_[i]}  # forward process
        if [ $? != 0 ]; then echo $fnm: Failed.; return 1; fi
      done
    #---  parallel processing 
    else
      echo $fnm: parallel ... 
      for (( i=0; i<_num; )); do
        #---  Create up to $_max_jobs background processes.
        local pid=(); local pnum=0
        local j; for (( j=0; j<_max_jobs; j++ )); do    
          local cx=$((i+j))
          if [ $cx -ge $_num ]; then break; fi

          local cmd=${_cmd_[$cx]}
          local mylog=${logdir}/${_logfn}.${cx}
          echo $fnm: Calling $cx-th command.  See $mylog for progress.
          $cmd > $mylog &  # background process
          if [ $? != 0 ]; then echo $fnm: Failed to invoke the $cx-th command.; return 1; fi

          pid[$pnum]=$!; pnum=$((pnum+1)) # save the process id.
        done

        #---  Wait for the background processes to complete. 
        echo Waiting for $pnum jobs ... 
        local _do_exit=0
        local j; for (( j=0; j<pnum; j++ )); do
          wait ${pid[$j]}
          if [ $? != 0 ]; then echo $fnm: Failed. See ${logdir}/${_logfn}.$((i+j)).; _do_exit=1; fi
        done
        if [ $_do_exit = 1 ]; then return 1; fi
        echo $fnm: $pnum jobs completed ...
        i=$((i+pnum))
      done
    fi
    return 0
  }
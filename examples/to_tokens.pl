 #  
 #  NOTE: This code was originally used for cleaning up/tokenizing html files and Wikipedia articles.  
 #        That's why it's a bit slow doing things unnecessary for relatively clean data like IMDB or RCV1.  
 # 
 
  use utf8; 
  use strict 'vars';  
  
  my $dlm = "\t"; 
  
  my $arg_num = $#ARGV + 1; 
  if ($arg_num != 3 && $arg_num != 4 && $arg_num != 5) {
    print STDERR "inp_fn reg_fn ext [ do_keep_url\(1\|0\) max_hyphenated ]\n"; 
    exit; 
  }

  my $argx = 0; 
  my $inp_fn = $ARGV[$argx++]; 
  my $reg_fn = $ARGV[$argx++]; 
  my $ext = $ARGV[$argx++]; 
  my $do_keep_url = 0;     # replace URL with $U$. 
  if ($argx < $arg_num) {
    $do_keep_url = $ARGV[$argx++]; 
  }
  my $max_hyphenated = 3;  # treat it as one token if up to this many words are concatenated by hyphens
  if ($argx < $arg_num) {
    $max_hyphenated = $ARGV[$argx++];
  }

  my $timestr = localtime;   
  print "$timestr\n";   
  my %nonword_hash = (); 
  my %hyphen_hash = (); 
  
  #----  read registered 
  my %hash_reg = ();  # original
  my @orgreg;  # original 
  my @reg;     # after applying spechar separation 
  open(REG, $reg_fn) or die("Can't open $reg_fn\n"); 
  while(<REG>) {
    my $myreg = $_; 
    chomp $myreg; 
    if ($myreg =~ /^\#\-\-\-/) { next; } # comment 
    if ($myreg =~ /^\s*(\S.*\S)\s*$/) { $myreg = $1; }
    if ($myreg =~ /\S\S/) { $hash_reg{$myreg} = length($myreg); }
  }
  close(REG); 
  
  #---  in the descending order of length ... 
  my $reg_num = 0; 
  foreach my $myreg (sort hash_reg_comp (keys(%hash_reg))) { 
    $orgreg[$reg_num] = $myreg; 
    $myreg = &separate_spechar($myreg); 
    $reg[$reg_num++] = $myreg; 
  }
  &prepare_registered(); 
  
  #-----
  if ($inp_fn =~ /\.lst$/) {
    if (!open(LST, $inp_fn)) {
      print STDERR "Can't open $inp_fn\n"; 
      exit; 
    }
    while(<LST>) {
      my $fn = $_; 
      chomp $fn; 
      &tokenize($fn); 
    }
    close(LST); 
  }
  else {
    &tokenize($inp_fn); 
  }

  for my $w ( keys %nonword_hash) {
    my $count = $nonword_hash{$w}; 
#    print STDERR "$w\t$count\n"; 
  }  
  for my $w ( keys %hyphen_hash) {
    my $count = $hyphen_hash{$w}; 
#    print STDERR "$w\t$count\n"; 
  }  
  my $timestr = localtime;   
  print "$timestr\n";   
  
#-----  
sub hash_reg_comp {
  $hash_reg{$b} <=> $hash_reg{$a}; 
}  
  
#-----  
sub tokenize {
  my($fn) = @_; 
  open(INP, $fn) or die("Can't open $fn\n");
  my $out_fn = "$fn$ext"; 
  open(OUT, ">$out_fn") or die("Can't open $out_fn\n"); 
  print "Generating $out_fn\n"; 
  
  my $lineno = 1; 
  while(<INP>) {
    my $line = $_; 
    chomp $line; 

    $line = " $line ";   
    $line =~ s/\s/ /gs; 

    $line =~ s/\<br \/\>/ /gs; 
    $line =~ s/\<p\>/ /gs; 
    $line =~ s/\&lt\;/\</gs; 
    $line =~ s/\&quot\;/\"/gs;
    $line =~ s/\&amp\;/\&/gs; 
    
    $line =~ s/\<a [^\>]+\>//gs;  # remove <a ...>
    $line =~ s/\<[A-Z_a-z]{1,8}\>//gs;  # remove <..>
    $line =~ s/\<\/[A-Z_a-z]{1,8}\>//gs; # remove </..>
    
    if ($do_keep_url != 1 ) {
      $line =~ s/http\:\/\/\S+/ \$U\$ /gs;  # URL -> $U$
      $line =~ s/https\:\/\/\S+/ \$U\$ /gs; # URL -> $U$
    }

    #---  separate special characters     
    $line = &separate_spechar($line); 
    #------------------------------
    my $out = &handle_registered($line); 
    #------------------------------
  
    my @tok = split(/\s+/, $out); 
    my $tok_num = $#tok+1; 
    $out = ''; 
    my $tx; 
    for ($tx = 0; $tx < $tok_num; ++$tx) { 
      if (&is_registered($tok[$tx]) == 1) {}
      else { 
        if ($tok[$tx] =~ /^([\d]+)(\S)([\d]+)$/) {}   # e.g., 1987-89 
        elsif ($tok[$tx] =~ /^[\d]+$/) {}    # integer such as year, age, date, month ...
        elsif ($tok[$tx] =~ /\d/ && $tok[$tx] =~ /^[\d\.\,\-]+$/) {} # 1,111, 1.111, 1/2, phone number, social security number, ...
        elsif ($tok[$tx] =~ /[A-Z_a-z]/ && $tok[$tx] =~ /\d/) {}  # alphabet and digits
        elsif ($tok[$tx] =~ /[A-Z_a-z]\-[A-Z_a-z]/) {
          $tok[$tx] = &handle_hyphenated_token($tok[$tx]); 
        }
        elsif ($tok[$tx] =~ /\w/ && $tok[$tx] =~ /\W/) { # other weird strings 
          ++$nonword_hash{$tok[$tx]}; 
        }
      }
      $out .= "$tok[$tx] "; 
    }
    $out = " $out "; 
    print OUT "$out\n"; 

    ++$lineno; 
  }
  close(OUT); 
  close(INP); 
}  

################
# input: $reg_num
# updated: @reg 
sub prepare_registered {  # make regular expressions 
  my $ix; 
  for ($ix = 0; $ix < $reg_num; ++$ix) {
    my $jx; 
    my $myreg = ''; 
    for ($jx = 0; $jx < length($reg[$ix]); ++$jx) {
      my $ch = substr($reg[$ix], $jx, 1); 
      if ($ch eq '\t') {
        $myreg .= '\\t'; 
      }
      elsif ($ch !~ /[A-Z_a-z_0-9]/) {
        $myreg .= "\\"; 
      }
      $myreg .= $ch; 
    }
    $reg[$ix] = $myreg;   
  }
}
  
################
# input: @reg, $reg_num
sub handle_registered {
  my($inp) = @_; 
  my $out = " $inp "; 
  my $ix; 
  for ($ix = 0; $ix < $reg_num; ++$ix) {
#    my $myreg = $reg[$ix]; 
#    $out =~ s/(\s)$myreg(\W)/$1 $orgreg[$ix] $2/gs;  
    $out =~ s/(\s)$reg[$ix](\W)/$1 $orgreg[$ix] $2/gs;      
  }
  return $out; 
}

#################
sub is_registered {
  my($inp) = @_; 
  if ($hash_reg{$inp} != 0) {
    return 1; 
  }
  return 0; 
}

###############
sub handle_hyphenated_token {
  my($inp) = @_; 
  my @tok = split(/\-/, $inp); 
  if ($#tok+1 <= $max_hyphenated) {
    ++$hyphen_hash{$inp}; 
    return $inp; 
  }
  else {
    my $out = $inp; 
    $out =~ s/\-/ /gs;     
    return $out; 
  }
}

#####
sub separate_spechar {
  my($inp) = @_; 
  my $line = $inp; 
  if ($max_hyphenated < 0) {
    $line =~ s/([a-z_A-Z])\-([a-z_A-Z])/$1$dlm-$dlm$2/gs; # separate hyphens
  }
    
  #  !"#$%&'()*+,-./  :;<=>?@  [\]^_`  {|}~       
  $line =~ s/\.{2,100}/$dlm\.\.$dlm/gs;  # "...." -> " .. "
 
  $line =~ s/\!+/\!/gs;  # !!!! -> !
  $line =~ s/\"+/\"/gs;  # """" -> "
  $line =~ s/\#+/\#/gs;  # #### -> #
  $line =~ s/\$+/\$/gs;  # $$$$ -> $
  $line =~ s/\%+/\%/gs;  # %%%% -> %  
  $line =~ s/\'+/\'/gs;  # '''' -> '
  $line =~ s/\*+/\*/gs;  # **** -> *
  $line =~ s/\++/\+/gs;  # ++++ -> +
  $line =~ s/\,+/\,/gs;  # ,,,, -> ,    
  $line =~ s/\-+/\-/gs;  # ---- -> -
  $line =~ s/\/+/\//gs;  # //// -> /
  $line =~ s/\:+/\:/gs;  # :::: -> :
  $line =~ s/\;+/\;/gs;  # ;;;; -> ;    
  $line =~ s/\=+/\=/gs;  # ==== -> =
  $line =~ s/\?+/\?/gs;  # ???? -> ?
  $line =~ s/\@+/\@/gs;  # @@@@ -> @    
  $line =~ s/\_+/\_/gs;  # ____ -> _  
  
  $line =~ s/([\)\(\!\%\^\&\*\+\#\@\~\`\;\?\=\"\[\]\{\}])/$dlm$1$dlm/gs;  # separate ()!%^&*+#@~`;?="[]{}   
  $line =~ s/([A-Z_a-z]{2,100})(\.)([A-Z_a-z]{2,100})/$1$dlm$2$dlm$3/gs;  # separate . between alphabets (more than one each)
  #---  be careful about $ ' _ .

  $line =~ s/(\s)/$1$1/gs; 
  $line =~ s/(\s\$)(.[^\$]\S*\s)/$1$dlm$2/gs; # separate $ at the beginning but keep $U$
  $line =~ s/(\s\S*[^\$].)(\$\s)/$1$dlm$2/gs; # separate $ at the end but keep $U$

  $line = &separate_dlm_shrink_repeat($line); # separate ".", ",", "/" and shrink repeat
  
  $line =~ s/(\s\')(\S)/$1$dlm$2/gs; # separate single quote at the beginning 
  $line =~ s/(\S)(\'\s)/$1$dlm$2/gs; # separate single quote at the end 
  $line =~ s/(\s)\-([A-Z_a-z]+)\-(\s)/$1$2$3/gs;   # -abcde- -> abcde 

  $line =~ s/$dlm{2,100}/$dlm/gs; 

  $line =~ s/^\s*(\S.*\S)\s*$/$1/; 
  $line =~ s/(\s)/$1$dlm/gs;   
  
  $line = " $line ";  # without this, can't separate "'s" etc. in the first/last word. 
  $line =~ s/(\s\S+)(\'(s|d|ll|re|ve|m)\s)/$1$dlm$2/gis;   # 's 'd 'll 're 've 'm

  $line =~ s/(\s\w\S*\w)([\!-\-_\:-\@_\/_\[-\`_\{-\~]+\s)/$1$dlm$dlm$2/gs;   # separate spechars at the end  
  $line =~ s/(\s[\!-\&_\(-\/_\:-\@_\[-\`_\{-\~]+)(\w\S*\w\s)/$1$dlm$dlm$2/gs;   # separate spechars at the beginning
  
  $line =~ s/^\s*(\S.*\S)\s*$/$1/;  # we need this for registered tokens 
  return $line; 
}  

################
sub separate_dlm_shrink_repeat {
  my($inp) = @_; 
 
  my $count = 0; 
  my $last_char = ''; 
 
  my $inp_padded = '  ' . $inp . '  ';  

  my $len = length($inp); 
  my $out = ""; 
  my $ix; 
  for ($ix = 0; $ix < $len; ++$ix) {
    my $char = substr($inp, $ix, 1); 
    my $do_dlm = 0; 
    if ($char eq '.') {
      my $ch3 = substr($inp_padded, $ix+1, 3);  # bef, char, aft 
      my $ch4 = substr($inp_padded, $ix, 4);    # bef2, bef, char aft
      if ($ch3 =~ /^[A-Z]\.[A-Z]/) {} # dot between upper cases 
      elsif ($ch4 =~ /^\.[A-Z]\./) {} # following .A 
      elsif ($ch3 =~ /^\d\.\d/ || $ch3 =~ /\.\./) {} # between digits or ".."
      else { $do_dlm = 1; }
    }
    elsif ($char eq ',' || $char eq '/' || $char eq ':') {
      my $ch3 = substr($inp_padded, $ix+1, 3);  # bef, char, aft 
      if ($ch3 =~ /\d.\d/) {} # between digits 
      else                 { $do_dlm = 1; }
    }

    if ($char eq $last_char && $do_dlm != 1 && $char !~ /\d/) {  # sooooo -> soo 
      ++$count; 
     if ($count > 1) { $char = ''; } 
    }
    else {
      $count = 0; 
      $last_char = $char; 
    }
    
    if ($do_dlm == 1) { $out .= "$dlm$char$dlm"; }
    else              { $out .= $char; }
  }

  return $out; 
}  

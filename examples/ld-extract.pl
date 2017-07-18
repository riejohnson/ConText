
  use strict 'vars'; 
  my $arg_num = $#ARGV+1; 
  if ($arg_num != 2 && $arg_num != 3) {
    print STDERR "inp_fn outnm [ do_dbpedia=\(0\|1\) ]\n"; 
    exit -1; 
  }

  my $argx = 0; 
  my $inp_fn = $ARGV[$argx++]; 
  my $outnm = $ARGV[$argx++]; 
  my $do_dbpedia = 0; 
  if ($argx < $arg_num) { $do_dbpedia = $ARGV[$argx++]; }
  
  open(INP, "$inp_fn") or die("Can't open $inp_fn\n"); 
  my $txt_fn = $outnm . '.txt'; 
  my $cat_fn = $outnm . '.cat'; 
  my $catdic = $outnm . '.catdic'; 
  open(TXT, ">$txt_fn") or die("Can't open $txt_fn\n"); 
  open(CAT, ">$cat_fn") or die("Can't open $cat_fn\n"); 
  open(CATDIC, ">$catdic") or die("Cant' open $catdic\n"); 

  my %hash = (); 

  while(<INP>) {
    my $line = $_; 
    chomp $line; 
    my $exp = '^\"([^\"]+)\"\,\"(.*)\"$'; 
    if ($do_dbpedia == 1) { $exp = '^([^\"]+)\,\"(.*)\"$'; }

    if ($line =~ /$exp/) {
      my $cat = $1; 
      my $txt = $2; 
      $txt = &cleanup($txt); 
      if ($hash{$cat} != 1) {
        $hash{$cat} = 1; 
        print CATDIC "$cat\n"; 
      }
      print TXT "$txt\n"; 
      print CAT "$cat\n"; 
    }
    else {
      print STDERR "unexpected line: $line\n"; 
    }
  }

  close(INP); 
  close(TXT);
  close(CAT); 
  close(CATDIC); 

#####
sub cleanup {
  my($inp) = @_; 

  my $out = $inp; 
  
  $out =~ s/\"\,\"/ \| /gs;   # delimiter between the summary and the text 
  $out =~ s/\\\"\"/\"/gs; 
  $out =~ s/\"\"/\"/gs; 
  $out =~ s/\\n/ /gs; 
  $out =~ s/\s/ /gs; 
  return $out; 
}
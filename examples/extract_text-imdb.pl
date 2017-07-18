  #---  for categorization (not for counting cooccurrences)
  #---  replace <br /> <p> by space
  
  use strict 'vars'; 
  my $arg_num = $#ARGV + 1; 
  if ($arg_num != 5) {
    print STDERR "lst_fn inp_dir cat out_fn cat_fn\n"; 
    exit; 

  }

  my $argx = 0; 
  my $lst_fn = $ARGV[$argx++]; 
  my $inp_dir = $ARGV[$argx++]; 
  my $cat = $ARGV[$argx++]; 
  my $out_fn = $ARGV[$argx++]; 
  my $cat_fn = $ARGV[$argx++]; 
  
  open(LST, $lst_fn) or die("Can't open $lst_fn\n"); 
  open(OUT, ">$out_fn") or die("Can't open $out_fn\n"); 
  open(CAT, ">$cat_fn") or die("Can't open $cat_fn\n"); 
  
  my $inc = 1000; 
  my $count = 0; 
  while(<LST>) {
    my $line = $_; 
    chomp $line; 
    my $inp_fn = $inp_dir . $line; 
    my $out = &proc($inp_fn); 
    ++$count; 

    print OUT "$out\n"; 
    print CAT "$cat\n"; 
  }
  print "$lst_fn ($count) ... \n"; 
  
  close(OUT); 
  close(LST); 
  
######  just remove newline 
sub proc {
  my($inp_fn) = @_; 

  open(INP, $inp_fn) or die("Can't open $inp_fn\n"); 
  my $out = ""; 
  while(<INP>) {
    my $line = $_; 
    chomp $line; 
    $out .= "$line "; 
  }
  close(INP); 
  return $out; 
}

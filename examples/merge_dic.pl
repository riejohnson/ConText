  use strict 'vars'; 

  my $arg_num = $#ARGV+1; 
  if ($arg_num < 2) {
    print STDERR "max_num dic1 dic2 ... \n"; 
    exit -1; 
  }
  my $argx = 0; 
  my $max_num = $ARGV[$argx++]; 

  my %hash = (); 
  for ( ; $argx < $arg_num; ++$argx) {
    my $fn = $ARGV[$argx]; 
    open(INP, $fn) or die("Can't open $fn\n"); 
    while(<INP>) {
      my $line = $_; 
      chomp $line; 
      if ($line =~ /^(\S.*)\t(\d+)$/) {
        my $word = $1; 
        my $count = $2; 
        if ($hash{$word} > 0) {
          print STDERR "duplicated word: $word\n"; 
          exit -1; 
        }
        $hash{$word} = $count; 
      }
      elsif ($line =~ /^(\S.*)$/) {
        my $word = $1; 
        my $count = 1; 
        if ($hash{$word} > 0) {
          print STDERR "duplicated word: $word\n"; 
          exit -1; 
        }
        $hash{$word} = $count;         
      }
      else {
        print STDERR "invalid line: $line\n"; 
        exit -1; 
      }
    }
    close(INP); 
  }
  
  my $num = 0; 
  foreach my $word (sort { $hash{$b} <=> $hash{$a} } keys %hash) {
    if ($max_num > 0 && $num >= $max_num) {
      last; 
    }
    my $count = $hash{$word}; 
    print "$word\t$count\n"; 
    ++$num; 
  }
  exit 0
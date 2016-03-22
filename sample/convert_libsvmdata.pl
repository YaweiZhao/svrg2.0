#---  convert libsvm format to libsvrg's sparse format 

  use strict 'vars'; 

  #======================================================
  #  libsvm labels => libsvrg labels (0,1,2,...)
  my %lab_hash=(
         '-1' => '0',   # Change this 
         '+1' => '1'    # Change this
     ); 
  #======================================================
  
  my $arg_num = $#ARGV+1; 
  if ($arg_num != 4) {
    print STDERR "Usage: #feat  input_fn  output_feature_fn  output_lab_fn\n"; 
    exit -1; 
  }
  
  my $argx = 0; 
  my $f_num=$ARGV[$argx++]; 
  my $inpfn=$ARGV[$argx++]; 
  my $x_fn=$ARGV[$argx++]; 
  my $y_fn=$ARGV[$argx++]; 

  open(INP, "$inpfn") or die("Can't open $inpfn\n");   
  open(OUTX, ">$x_fn") or die("Can't open $x_fn\n"); 
  open(OUTY, ">$y_fn") or die("Can't open $y_fn\n"); 
  
  print OUTX "sparse $f_num\n"; 
  
  my $lx = 0; 
  while(<INP>) {
    my $line = $_; 
    chomp $line; 

    my @tok = split(/\s+/, $line); 
    my $tok_num = $#tok+1; 
    if ($tok_num <= 0) {
      print "No tokens?!"; 
      exit -1; 
    }
    
    my $y = $lab_hash{$tok[0]}; 
    if ($y eq '') {
      my $line_no = $lx+1; 
      print "Unknown label in Line $line_no: $tok[0]\n"; 
      print "Update the hash table \%lab_hash at the beginning of this code if necessary.\n"; 
      exit -1; 
    }
    print OUTY "$y\n"; 
    
    my $out = "";     
    my $tx;
    for ($tx = 1; $tx < $tok_num; ++$tx) {
      my $fid = -1; 
      my $value = 1; 
      if ($tok[$tx] =~ /^(\d+)\:(\S+)$/) {
        $fid = $1 - 1; 
        $value = $2; 
      }
      elsif ($tok[$tx] =~ /^(\d+)$/) {
        $fid = $1 - 1; 
      }
      else {
        print STDERR "Unexpected token: $tok[$tx]\n"; 
        exit -1; 
      }
      if ($fid >= $f_num) {
        my $line = $lx+1; 
        print "Invalid feature id at line $line: $tok[$tx]\n"; 
        exit -1; 
      }
      $out .= "$fid\:$value ";
    }
    print OUTX "$out\n"; 
    ++$lx; 
  }
  my $num = $lx; 

  close(INP); 
  close(OUTX); 
  close(OUTY); 
  
  print "Converted $num data points ... \n"; 
  
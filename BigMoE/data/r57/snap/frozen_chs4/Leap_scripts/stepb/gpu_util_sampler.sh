#!/bin/sh
# GPU "Device Utilization %" sampler (ioreg, no sudo; ~30-50 Hz): "<unix time> <percent>" per line to $1 until killed.
exec perl -MTime::HiRes=time,sleep -e 'open(my $o, ">>", $ARGV[0]) or die; select($o); $|=1;
  while (1) { my $t = time; my $u = `ioreg -r -d 1 -w 0 -c AGXAccelerator 2>/dev/null`; my ($d) = $u =~ /"Device Utilization %"=(\d+)/;
  printf "%.3f %d\n", $t, defined $d ? $d : -1; sleep 0.015 }' "$1"

use strict;
use warnings;

my $path = shift @ARGV or die "usage: $0 FILE\n";
open my $input, '<:raw', $path or die "cannot read $path: $!\n";
my @lines = <$input>;
close $input;

for my $line (@lines) {
    if ($line =~ /Auxiliary campaign validity audit/) {
        $line = "\\caption{Auxiliary campaign validity audit.}\n";
    }
}

open my $output, '>:raw', $path or die "cannot write $path: $!\n";
print {$output} @lines;
close $output;

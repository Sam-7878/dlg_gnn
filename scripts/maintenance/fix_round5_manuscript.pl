use strict;
use warnings;

my $path = shift @ARGV or die "usage: $0 FILE\n";
open my $input, '<:raw', $path or die "cannot read $path: $!\n";
my @lines = <$input>;
close $input;

$lines[8] = "\\usepackage[hidelinks]{hyperref}\n";
$lines[47] = "\\title{A Validity-First Evaluation of Streaming Graph Neural Networks for Financial Fraud Detection under Temporal Distribution Shift}\n";
$lines[343] = "\\bibliographystyle{IEEEtran}\n";

open my $output, '>:raw', $path or die "cannot write $path: $!\n";
print {$output} @lines;
close $output;

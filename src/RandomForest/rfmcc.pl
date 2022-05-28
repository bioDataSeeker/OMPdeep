$tp=$tn=$fp=$fn=0;
open(IN,$ARGV[0])||die "can not open";
	while($line=<IN>){
		chomp($line);
		$line=~s/\\n//g;
		@wds = split(/\s+/,$line);
		push(@s,$wds[1]);
	}
close IN;

open(IN,$ARGV[0])||die "can not open";
        while($line=<IN>){
                chomp($line);
                $line=~s/\\n//g;
		@wds=split(/\s+/,$line);
                push(@label,$wds[0]);
        }
close IN;

$len1=@s;
$len2=@label;
print "len1=$len1 len2=$len2\n";

for($i=0;$i<@label;$i++){
		chomp($line);
		if($s[$i]>=0){
			if($label[$i]>0.5){
				$tp++;
			}else{
				$fp++;
			}
		}else{
			if($label[$i]<=0){
                                $tn++;
                        }else{
                                $fn++;
                        }
		}
}

$ac=($tp+$tn)/($tp+$fp+$tn+$fn);
$sn=$tp/($tp+$fn);
$sp=$tn/($tn+$fp);
$mcc=($tp*$tn-$fn*$fp)/sqrt(($tp+$fn)*($tn+$fp)*($tp+$fp)*($tn+$fn));
print "tp=$tp tn=$tn fp=$fp fn=$fn\n";
print "ac:$ac\n";
print "sn:$sn\n";
print "sp:$sp\n";
print "MCC:$mcc\n";


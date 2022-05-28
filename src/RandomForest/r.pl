for($i=1;$i<=25;$i++){
	print "file $i\n";
	$cmd="Rscript.exe C:\\Users\\yajiaomm\\Desktop\\randomforest20201129\\omp25\\rf\\omprf$i.R";
	system($cmd);
}
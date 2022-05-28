for($i=0;$i<25;$i++){
	print "$i\n";
	print `./svm_learn svmtrain$i.dat model_$i`;
	print `./svm_classify svmtest$i.dat model_$i output_$i`;
}



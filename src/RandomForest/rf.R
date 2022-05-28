library(randomForest)
group=c("01")
for(g in group){
	train<-read.csv(paste('C:\\Users\\yajiaomm\\Desktop\\randomforest20201129\\train',g,sep=''))
	long<-length(train)
	model<-randomForest(train[2:long],factor(train[[1]]),ntree=1000,prox=TRUE)
	save(model,file=paste('C:\\Users\\yajiaomm\\Desktop\\randomforest20201129\\model',g,sep=''))
	test<-read.csv(paste("C:\\Users\\yajiaomm\\Desktop\\randomforest20201129\\test",g,sep=""))
	result<-predict(model,test[2:long],type="prob")
	write(paste(test$id,result[,1],test$class,sep="\t"),file=paste('C:\\Users\\yajiaomm\\Desktop\\randomforest20201129\\result',g,sep=''))
	write.table(result,file="C:\\Users\\yajiaomm\\Desktop\\randomforest20201129\\result_table.txt",append=F)
}

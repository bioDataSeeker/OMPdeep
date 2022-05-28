library(randomForest)
group=c("01","02","03","04","05","06","07","08","09","10")
for(g in group){
	train<-read.csv(paste('train/train',g,sep=''))
	long<-length(train)
	model<-randomForest(train[2:long],train[[1]],ntree=4000,norm.votes=FALSE)
	save(model,file=paste('model/model',g,sep=''))
	test<-read.csv(paste("./test/test",g,sep=""))
	result<-predict(model,test[2:long],type="prob")
	write(paste(test$id,result[,1],test$class,sep="\t"),file=paste('rf_result/result',g,sep=''))
}

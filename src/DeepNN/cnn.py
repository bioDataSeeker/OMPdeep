import sys
import numpy as np
from tensorflow.keras import models,layers,optimizers,regularizers
import pandas as pd

nfeature=str(sys.argv[1])
nlabel=str(sys.argv[2])
testfeature=str(sys.argv[3])
noutput=str(sys.argv[4])

def txt_to_matrix1(filename):
    label=[]
    file=open(filename)
    lines=file.readlines()
    rows=len(lines)
    datamat=np.zeros((rows,1))
    row=0
    for line in lines:
        eachline = line.split()
        read_data = [float(x) for x in eachline[0:1]]
        label.append(read_data)

def txt_to_matrix546(filename):
    encoding=[]
    file=open(filename)
    lines=file.readlines()
    rows=len(lines)
    datamat=np.zeros((rows,620)) 
    row=0
    for line in lines:
        eachline = line.split()
        read_data = [float(x) for x in eachline[0:620]]
        x=np.array(read_data)
        result = x.reshape((20,31))  # 将一维数组变成4行5列  原数组不会被修改或者覆盖
        encoding.append(result)
    encoding=np.array(encoding)
    return encoding

def cnn(depth=31,l1=16,l2=512,gamma=1e-4,lr=1e-3,w1=3,w2=2):
    model=models.Sequential()
    model.add(layers.Conv1D(l1,w1,activation='relu',kernel_regularizer=regularizers.l1(gamma),input_shape=(20,31),padding='same'))
    model.add(layers.MaxPooling1D(w2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l1(gamma)))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['acc'])
    return model

def rnn(depth=21,l1=16,l2=128,gamma=1e-4,lr=1e-3,w1=3,w2=2):
    model=models.Sequential()
    model.add(layers.LSTM(l1,return_sequences=True,input_shape=(35,20)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l1(gamma)))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['acc'])
    return model

def readL(file):
    label=[]
    f=open(file,'r')
    for line in f:
        col=line.strip().split('\t')
        s=col[0]
        label.append(s)
    f.close()
    label=np.array(label).astype('float32')
    return label

def readF(file):
    encoding=[]
    label=[]
    f=open(file,'r')
    for line in f:
        eachline = line.split()
        encoding.append([float(x) for x in eachline[0:546]])
    f.close()
    encoding=np.array(encoding)
    return encoding

x=txt_to_matrix546(nfeature)
y=readL(nlabel)
trainFeature1=x
trainLabel=y
print(trainFeature1.shape)
#exit()
m1=cnn(l1=32,l2=256)
m1.fit(x,y,batch_size=100,epochs=1000,verbose=0)

test=txt_to_matrix546(testfeature)
a=m1.predict(test).reshape(len(test))
print(a)
output=open(noutput,'w+')
L=a
for i in range(len(L)):  
    output.write(str(L[i]))
    output.write('\n')      
output.close()
exit()
# M=trainFeature1
# output = open('raw_data.txt','w+')
# for i in range(len(M)):
    # for j in range(len(M[i])):
        # output.write(str(M[i][j]))
        # output.write(' ')   
    # output.write('\n')      
# output.close()
# exit()
# M=trainLabel
# output = open('checkL_data.txt','w+')
# for i in range(len(M)):
    # output.write(str(M[i]))   
    # output.write('\n')      
# output.close()


#m1.predict(trainFeature1).reshape(len(trainFeature1))
#exit()
k=5
np.random.seed(1234)
binaryCoding,label=binary_encode('./trainset')
aaindexCoding,label=index_encode('./trainset')
num=len(label)
mode1=np.arange(num/2)%k
mode2=np.arange(num/2)%k
np.random.shuffle(mode1)
np.random.shuffle(mode2)
mode=np.concatenate((mode1,mode2))
score_binary_cnn=np.zeros(num)
score_binary_rnn=np.zeros(num)
score_aaindex_cnn=np.zeros(num)
score_aaindex_rnn=np.zeros(num)

for fold in range(k): 
    trainLabel=label[mode!=fold]
    testLabel=label[mode==fold]
    trainFeature1=binaryCoding[mode!=fold]
    testFeature1=binaryCoding[mode==fold]

    trainFeature2=aaindexCoding[mode!=fold]
    testFeature2=aaindexCoding[mode==fold]
    #test_aaindex=trainCoding_aaindex[mode==fold]
    
    m1=cnn(l1=32,l2=256)
    m1.fit(trainFeature1,trainLabel,batch_size=100,epochs=50,verbose=0)
    score_binary_cnn[mode==fold]=m1.predict(testFeature1).reshape(len(testFeature1))
    print(trainFeature1)
    M=trainFeature1
    output = open('raw_data.txt','w+')
    for i in range(len(M)):
        for j in range(len(M[i])):
            output.write(str(M[i][j]))
            output.write(' ')   
        output.write('\n')      
    output.close()
    output = open('raw_Ldata.txt','w+')
    L=trainLabel
    for i in range(len(L)):  
        output.write(str(L[i]))
        output.write('\n')      
    output.close()
    exit()
 
    m2=rnn(l1=16,l2=512)
    m2.fit(trainFeature1,trainLabel,batch_size=100,epochs=50,verbose=0)
    score_binary_rnn[mode==fold]=m2.predict(testFeature1).reshape(len(testFeature1))



    m3=cnn(depth=31,l1=32,l2=128)
    m3.fit(trainFeature2,trainLabel,batch_size=100,epochs=100,verbose=0)
    score_aaindex_cnn[mode==fold]=m3.predict(testFeature2).reshape(len(testFeature2))

    m4=rnn(depth=31,l1=16,l2=512)
    m4.fit(trainFeature2,trainLabel,batch_size=100,epochs=50,verbose=0)
    score_aaindex_rnn[mode==fold]=m4.predict(testFeature2).reshape(len(testFeature2))
    

np.savez('cvscore.npz',cnn1=score_binary_cnn,rnn1=score_binary_rnn,
         cnn2=score_aaindex_cnn,rnn2=score_aaindex_rnn,label=label)



binaryCodingTest,labelTest=binary_encode('./testset')
aaindexCodingTest,_=index_encode('./testset')

m1=cnn(l1=32,l2=256)
m1.fit(binaryCoding,label,batch_size=100,epochs=50,verbose=0)
score_binary_cnn_test=m1.predict(binaryCodingTest).reshape(len(binaryCodingTest))
    
m2=rnn(l1=16,l2=512)
m2.fit(binaryCoding,label,batch_size=100,epochs=50,verbose=0)
score_binary_rnn_test=m2.predict(binaryCodingTest).reshape(len(binaryCodingTest))

m3=cnn(depth=31,l1=32,l2=128)
m3.fit(aaindexCoding,label,batch_size=100,epochs=100,verbose=0)
score_aaindex_cnn_test=m3.predict(aaindexCodingTest).reshape(len(aaindexCodingTest))

m4=rnn(depth=31,l1=16,l2=512)
m4.fit(aaindexCoding,label,batch_size=100,epochs=50,verbose=0)
score_aaindex_rnn_test=m4.predict(aaindexCodingTest).reshape(len(aaindexCodingTest))

np.savez('testscore.npz',cnn1=score_binary_cnn_test,rnn1=score_binary_rnn_test,
         cnn2=score_aaindex_cnn_test,rnn2=score_aaindex_rnn_test,label=labelTest)

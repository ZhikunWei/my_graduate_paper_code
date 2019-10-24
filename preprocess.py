# _*_coding:utf-8_*_
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def reformatData():
    user2data = {}
    cnt = 0

    with open("raw_data/hash_zj.txt") as f:
        for line in f:
            line = line.split('|')
            streamNumber = line[0]
            commitTime = line[1]
            serviceKey = line[2]
            callingNumber = line[3]
            calledNumber = line[4]

            callingVLR = line[5]
            callingHlr = line[6]
            calledHlr = line[7]
            ringTime = line[8]
            callBeginTime = line[9]
            callEndTime = line[10]
            callDuration = line[11]
            MSCAddress = line[12]
            callResult = line[13]
            cause = line[14]
            routPrefix = line[15]
            redirectNumber = line[16]
            originalNumber = line[17]
            singalFlag = line[18]
            illegalType = line[19]
            illegalDesc = line[20]
            if callingNumber not in user2data:
                user2data[callingNumber] = []
                cnt += 1
                print(cnt)
            tmp = []
            tmp.append(commitTime)
            tmp.append(callingHlr)
            tmp.append(calledHlr)
            tmp.append(ringTime)
            tmp.append(callDuration)
            tmp.append(callResult)
            tmp.append(cause)

            user2data[callingNumber].append(tmp)

    print(len(user2data))   # ~440000
    with open('data/user2data.pkl', 'wb') as f:
        pickle.dump(user2data, f)


def transformToArray():
    with open('data/user2data.pkl', 'rb') as f:
        user2data = pickle.load(f)
    with open('data/user2label.pkl', 'rb') as f:
        user2label = pickle.load(f)
    X = []
    y = []

    calledHlr2code = {'0580': 0, '0574': 1, ' ': 2, '0570': 3, '0576': 4, '0578': 5, '0577': 6,
                      '0579': 7, '0575': 8, '0572': 9, '0573': 10, '0571': 11}
    for user, records in user2data.items():
        if len(X) % 10000 == 0:
            print(len(X), len(user2data))
        if user not in user2label:
            continue
        clock_hour_vector = [0] * 12
        callededHlr_vector = [0] * 12
        ring_time_vector = [0] * 4
        duration_vector = [0] * 6
        call_result_vector = [0] * 3
        cause_vector = [0] * 5
        for record in records:
            clock = record[0]
            clock_hour = int(clock[8:10])
            clock_hour_vector[clock_hour//2] += 1

            callingHlr = record[1]

            calledHlr = record[2]
            if calledHlr in calledHlr2code:
                callededHlr_vector[calledHlr2code[calledHlr]] += 1

            ringTime = int(record[3])
            if ringTime == 0:
                ring_time_vector[0] += 1
            elif ringTime < 10:
                ring_time_vector[1] += 1
            elif ringTime < 30:
                ring_time_vector[2] += 1
            else:
                ring_time_vector[3] += 1

            call_duration = int(record[4])
            if call_duration == 0:
                duration_vector[0] += 1
            elif call_duration < 10:
                duration_vector[1] += 1
            elif call_duration < 30:
                duration_vector[2] += 1
            elif call_duration < 60:
                duration_vector[3] += 1
            elif call_duration < 180:
                duration_vector[4] += 1
            else:
                duration_vector[5] += 1

            callResult = int(record[5])
            call_result_vector[callResult] += 1

            cause = record[6]
            if cause == '08F90':
                cause_vector[0] += 1
            elif cause == '08590':
                cause_vector[1] += 1
            elif cause == '08090':
                cause_vector[2] += 1
            elif cause == '08180':
                cause_vector[3] += 1
            else:
                cause_vector[4] += 1

        clock_hour_vector = preprocessing.scale(clock_hour_vector)
        callededHlr_vector = preprocessing.scale(callededHlr_vector)
        ring_time_vector = preprocessing.scale(ring_time_vector)
        duration_vector = preprocessing.scale(duration_vector)
        call_result_vector = preprocessing.scale(call_result_vector)
        cause_vector = preprocessing.scale(cause_vector)

        x = np.concatenate((clock_hour_vector, callededHlr_vector, ring_time_vector, duration_vector,
                            call_result_vector, cause_vector))
        # print(x)
        
        if user2label[user] == '-1':
            # if len(y) > 3000:
            #     continue
            y.append(0)
        else:
            y.append(1)
        X.append(x)

    print(len(X), len(y), sum(y))
    with open('data/arrays/X.pkl', 'wb') as f:
        pickle.dump(X, f)
    with open('data/arrays/y.pkl', 'wb') as f:
        pickle.dump(y, f)
        

def raw2vocabulary():
    with open('data/user2label.pkl', 'rb') as f:
        user2label = pickle.load(f)
    vocabulary = []
    with open("raw_data/hash_zj.txt") as f:
        for line in f:
            line = line.split('|')
            streamNumber = line[0]
            commitTime = line[1]
            
            serviceKey = line[2]
            callingNumber = line[3]
            calledNumber = line[4]
            if callingNumber not in user2label:
                continue
            
            callingVLR = line[5]
            callingHlr = line[6]
            callingHlr = 'calling_hlr_' + callingHlr
            vocabulary.append(callingHlr)
            
            calledHlr = line[7]
            calledHlr = 'called_hlr_' + calledHlr
            vocabulary.append(calledHlr)
            
            callingNumber = 'user_' + callingNumber
            vocabulary.append(callingNumber)

            clock_hour = 'clock_hour_' + commitTime[8:10]
            vocabulary.append(clock_hour)
            
            if user2label[line[3]] != '-1':
                vocabulary.append('is_spam')
            else:
                vocabulary.append('is_not_spam')
                
            ringTime = line[8]
            ringTime = int(ringTime)
            if ringTime == 0:
                ringTime = 'ring_time_0'
            elif ringTime < 10:
                ringTime = 'ring_time_10'
            elif ringTime < 30:
                ringTime = 'ring_time_30'
            elif ringTime < 60:
                ringTime = 'ring_time_60'
            else:
                ringTime = 'ring_time_long'
            vocabulary.append(ringTime)  # 5
            
            callBeginTime = line[9]
            callEndTime = line[10]
            callDuration = line[11]
            call_duration = int(callDuration)
            if call_duration == 0:
                call_duration = 'call_duration_0'
            elif call_duration < 10:
                call_duration = 'call_duration_10'
            elif call_duration < 30:
                call_duration = 'call_duration_30'
            elif call_duration < 60:
                call_duration = 'call_duration_60'
            elif call_duration < 180:
                call_duration = 'call_duration_180'
            else:
                call_duration = 'call_duration_long'
            vocabulary.append(call_duration)  # 6
            
            MSCAddress = line[12]
            callResult = line[13]
            callResult = 'result_' + callResult
            vocabulary.append(callResult)
            
            cause = line[14]
            cause = 'cause_' + cause
            vocabulary.append(cause)
            
            routPrefix = line[15]
            redirectNumber = line[16]
            originalNumber = line[17]
            singalFlag = line[18]
            illegalType = line[19]
            illegalDesc = line[20]

    print(len(vocabulary))
    with open('data/vocabulary.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)
        
        
def transform2vocabulary():
    with open('data/user2data.pkl', 'rb') as f:
        user2data = pickle.load(f)
    with open('data/user2label.pkl', 'rb') as f:
        user2label = pickle.load(f)
    vocabulary = []
    X = []
    y = []
    for user in user2data:
        if user not in user2label:
            continue
        
        userID = 'user_' + user
        for record in user2data[user]:
            tmp = ''
            
            callingHlr = 'calling_hlr_' + record[1]
            vocabulary.append(callingHlr)  # 3
            tmp += callingHlr + ' '
    
            calledHlr = 'called_hlr_' + record[2]
            vocabulary.append(calledHlr)  # 4
            tmp += calledHlr + ' '
            
            vocabulary.append(userID)  # 1
            tmp += userID + ' '
            
            clock = record[0]
            clock_hour = 'clock_hour_' + clock[8:10]
            vocabulary.append(clock_hour)   # 2
            tmp += clock_hour + ' '
            
            if user2label[user] != '-1':    # 9
                vocabulary.append('is_spam')
                y.append(1)
            else:
                vocabulary.append('is_not_spam')
                y.append(0)
            
            ringTime = int(record[3])
            if ringTime == 0:
                ringTime = 'ring_time_0'
            elif ringTime < 10:
                ringTime = 'ring_time_10'
            elif ringTime < 30:
                ringTime = 'ring_time_30'
            else:
                ringTime = 'ring_time_60'
            vocabulary.append(ringTime)     # 5
            tmp += ringTime + ' '
            
            call_duration = int(record[4])
            if call_duration == 0:
                call_duration = 'call_duration_0'
            elif call_duration < 10:
                call_duration = 'call_duration_10'
            elif call_duration < 30:
                call_duration = 'call_duration_30'
            elif call_duration < 60:
                call_duration = 'call_duration_60'
            elif call_duration < 180:
                call_duration = 'call_duration_180'
            else:
                call_duration = 'call_duration_200'
            vocabulary.append(call_duration)    # 6
            tmp += call_duration + ' '
            
            callResult = 'call_result_' + record[5]
            vocabulary.append(callResult)       # 7
            tmp += callResult + ' '
            
            cause = 'cause_' + record[6]
            vocabulary.append(cause)        # 8
            tmp += cause + ' \n'
            
            vocabulary.append('\n')
            X.append(tmp)
    
    print(len(vocabulary))
    alldata = ' '.join(vocabulary).split('\n')
    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2)
    
    with open('data/vocabulary.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)
    
    

def transformToDocument():
    with open('data/user2data.pkl', 'rb') as f:
        user2data = pickle.load(f)
    with open('data/user2label.pkl', 'rb') as f:
        user2label = pickle.load(f)
    X = []
    y = []
    for user, records in user2data.items():
        x = ''
        for record in records:
            clock = record[0]
            clock_hour = 'clock_hour_' + clock[8:10]
            callingHlr = 'calling_hlr_' + record[1]
            calledHlr = 'called_hlr_' + record[2]
            ringTime = int(record[3])
            if ringTime == 0:
                ringTime = 'ring_time_0'
            elif ringTime < 10:
                ringTime = 'ring_time_10'
            elif ringTime < 30:
                ringTime = 'ring_time_30'
            else:
                ringTime = 'ring_time_60'

            call_duration = int(record[4])
            if call_duration == 0:
                call_duration = 'call_duration_0'
            elif call_duration < 10:
                call_duration = 'call_duration_10'
            elif call_duration < 30:
                call_duration = 'call_duration_30'
            elif call_duration < 60:
                call_duration = 'call_duration_60'
            elif call_duration < 180:
                call_duration = 'call_duration_180'
            else:
                call_duration = 'call_duration_200'

            callResult = 'call_result_' + record[5]

            cause = 'cause_' + record[6]

            x += clock_hour+' '+callingHlr+' '+calledHlr+' '+ringTime+' '+call_duration+' '+callResult+' '+cause+' '
        X.append(x)
        if user in user2label:
            y.append(1)
        else:
            y.append(0)
    with open('data/lda/X_doc.pkl', 'wb') as f:
        pickle.dump(X, f)
    with open('data/lda/y_doc.pkl', 'wb') as f:
        pickle.dump(y, f)


def preprocessLableData():
    user2label = {}
    for folderName in ['1', '2', '3', '4']:
        with open('raw_data/ground_truth/'+folderName+'/src_labels.csv', encoding='utf-8') as f:
            for line in f:
                line = line.split(',')
                userID = line[0]
                userLabelID = line[1]
                userLabelName = line[2]
                userReportedNum = line[3]
                # if userLabelID == '-1':
                #     continue

                user2label[userID] = userLabelID

    print(len(user2label))  # 1633, 11893
    with open('data/user2label.pkl', 'wb') as f:
        pickle.dump(user2label, f)


def checkspam():
    with open('data/arrays/X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('data/arrays/y.pkl', 'rb') as f:
        y = pickle.load(f)
    for x, yy in zip(X, y):
        if yy is 1:
            print(x)


def splitTrainTestSetArray():
    with open('data/arrays/X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('data/arrays/y.pkl', 'rb') as f:
        y = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    with open('data/arrays/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('data/arrays/X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open('data/arrays/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('data/arrays/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)


def splitTrainTestSetDoc():
    with open('data/lda/X_doc.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('data/lda/y_doc.pkl', 'rb') as f:
        y = pickle.load(f)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    with open('data/lda/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('data/lda/X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open('data/lda/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('data/lda/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)


def fraudArrayData():
    with open('data/arrays/X_train.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('data/arrays/y_train.pkl', 'rb') as f:
        y = pickle.load(f)
    X_fraud = []
    for x, y in zip(X, y):
        if y == 1:
            X_fraud.append(x)
    X_fraud = np.array(X_fraud)
    with open('data/arrays/X_fraud_train.pkl', 'wb') as f:
        pickle.dump(X_fraud, f)
    print(sum(X_fraud)/len(X_fraud))


def normalArrayData():
    with open('data/arrays/X_train.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('data/arrays/y_train.pkl', 'rb') as f:
        y = pickle.load(f)
    X_normal = []
    for x, yy in zip(X, y):
        if yy == 0:
            X_normal.append(x)
    X_normal = np.array(X_normal)
    with open('data/arrays/X_normal_train.pkl', 'wb') as f:
        pickle.dump(X_normal, f)
    print(sum(X_normal)/len(X_normal))


if __name__ == '__main__':
    # reformatData()
    # preprocessLableData()
    raw2vocabulary()
    # transform2vocabulary()
    # transformToArray()
    # checkspam()
    # transformToDocument()
    # splitTrainTestSetArray()
    # splitTrainTestSetDoc()
    # fraudArrayData()
    # normalArrayData()
    pass



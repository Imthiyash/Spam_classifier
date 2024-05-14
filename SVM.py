# SVM Algorithm Implementation

from sklearn.svm import SVC

X = np.array(train)
y = np.array(train_dataset_target)
print(X.shape,y.shape)

svm_model = SVC(kernel='linear', C = 1.0)
svm_model.fit(X, y)

j = 0;cnt = 0
for content in test_dataset_text:
    words = nltk.word_tokenize(content)
    x = []
    for i in my_dict:
        x.append(int(i in words))
    predicted_label = svm_model.predict(np.array(x).reshape(1,len(x)))
    cnt += (predicted_label[0] != test_dataset_target[j])
    j += 1
    print(j,cnt)
    
print('Accuracy is',100*(1-(cnt/len(test_dataset_text))),'%')
# Logistic Regression Implementation
import math

def g(w,x):
    a = np.dot(w.T,np.array(x))
    return 1/(1+math.exp(-a))

w_log_reg = np.zeros(len(x))

c = 0;nt = 0.001
while c<=20:
    sum = np.zeros(len(train[0]))
    for i in range(0,len(train_dataset_text)):
        sum += np.array(train[i])*(train_dataset_target[i]-g(w_log_reg,train[i]))
    w_log_reg = w_log_reg + nt*(sum)
    c += 1
    print(c)

print(w_log_reg)

j = 0;cnt = 0
for content in test_dataset_text:
    words = nltk.word_tokenize(content)
    x = []
    for i in my_dict:
        x.append(int(i in words))
    y_pred = 0
    if g(w_log_reg,np.array(x)) > 0.5:
        y_pred = 1
    cnt += (y_pred != test_dataset_target[j])
    j += 1
    print(j,cnt)

print('Accuracy is',100*(1-(cnt/len(test_dataset_text))),'%')
# Perceptron Algorithm Implementation

def pred(a):
    return int(a < 0)

def vec(words):
    x = []
    for i in my_dict:
        x.append(int(i in words))
    return np.array(x)

train = []
for i in range(0,len(train_dataset_target)):
    train.append(vec(nltk.word_tokenize(train_dataset_text[i])))

w = np.zeros(len(my_dict))
while True:
    change = np.array(w)
    for j in range(0,len(train_dataset_text)):
        if not(pred(np.dot(w,train[j])) == train_dataset_target[j]):
            y = 1 - 2*train_dataset_target[j]
            w += train[j]*y
    if list(w) == list(change):
        break

cnt = 0
j = 0
# directory = 'test'
# for filename in os.listdir(directory):
#     filepath = os.path.join(directory, filename)
#     if os.path.isfile(filepath):
#         with open(filepath, 'r') as file:
#             content = file.read()
for content in test_dataset_text:
    words = nltk.word_tokenize(content)
    x = []
    for i in my_dict:
        x.append(int(i in words))
    cnt += (pred(np.dot(w,x)) != test_dataset_target[j])
    j += 1

print('Accuracy is',100*(1-(cnt/len(test_dataset_text))),'%')
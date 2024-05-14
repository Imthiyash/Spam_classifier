# Naive-Bayes algorithm implementation

cnt = 0;j = 0
# directory = 'test'
# for filename in os.listdir(directory):
#     filepath = os.path.join(directory, filename)
#     if os.path.isfile(filepath):
#         with open(filepath, 'r') as file:
#             content = file.read()
for content in test_dataset_text:
    words = nltk.word_tokenize(content)
    p0 = c[0]/(c[0]+c[1]);p1 = c[1]/(c[0]+c[1])
    for i in my_dict:
        if i in words:
            p0 *= (my_dict[i][0])
            p1 *= (my_dict[i][1])
        else:
            p0 *= (1-my_dict[i][0])
            p1 *= (1-my_dict[i][1])
    cnt += (test_dataset_target[j] != (p0 <= p1))
    j += 1

print('Accuracy is ',100*(1-(cnt/len(test_dataset_text))))
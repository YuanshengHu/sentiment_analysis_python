import os, cPickle, time
from train import cur_file_dir, read_datas, classify_data, extract_features\
     , classify_data, negate_sequence


os.chdir(cur_file_dir())

# read in the test datas and check accuracy
# to add your own test datas, add them in the respective files
#pos_test = raw_input("Name of the file contains your positive datas: ")
#neg_test = raw_input("Name of the file contains your negative datas: ")

#test = read_datas(pos_test, 'positive')
#test.extend(read_datas(neg_test, 'negative'))
test = read_datas("pos_test.txt", 'positive')
test.extend(read_datas("neg_test.txt", 'negative'))

total = accuracy = float(len(test))

print("Testing the model...")
start = time.clock()
for data in test:
    if classify_data(data[0]) != data[1]:
        accuracy -= 1
end = time.clock()

print("Total accuracy: %f%%. Time elapsed for %f s" % (accuracy / total * 100, (end - start)))

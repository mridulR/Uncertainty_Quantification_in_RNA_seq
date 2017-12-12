import sys
import numpy as np
import pandas as pd
import csv
import re
import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score

def get_corelation(mean_map, transcript_quant, TranscriptInNumOfClassesDict):
    x_len = []
    x_effec_len = []
    x_tpm = []
    x_no_of_reads = []
    y = []
    y_eq_class = []
    x_num_eq_class = []

    for transcriptId in mean_map.keys():
        if transcriptId in TranscriptInNumOfClassesDict.keys():
            y_eq_class.append(mean_map[transcriptId][0])
            x_num_eq_class.append(TranscriptInNumOfClassesDict[transcriptId])
        if transcriptId in transcript_quant.keys():
            y.append(mean_map[transcriptId][0])
            x_len.append(transcript_quant[transcriptId][0])
            x_effec_len.append(transcript_quant[transcriptId][1])
            x_tpm.append(transcript_quant[transcriptId][2])
            x_no_of_reads.append(transcript_quant[transcriptId][3])
    #print("*******************************************************")
    #print("Data for covarience calculation : ", len(y))
    #print("Co-relation length", np.corrcoef(x_len, y)[0][1]) 
    #print("Co-relation effective length", np.corrcoef(x_effec_len, y)[0][1])
    #print("Co-relation tpm", np.corrcoef(x_tpm, y)[0][1])
    #print("Co-relation num of reads", np.corrcoef(x_no_of_reads, y)[0][1])

    #print("Equivalence class size ", len(y_eq_class))
    #print("Co-relation for eq class", np.corrcoef(x_num_eq_class, y_eq_class)[0][1])
    #print("*******************************************************")

    return (np.corrcoef(x_len, y)[0][1], np.corrcoef(x_effec_len, y)[0][1], \
            np.corrcoef(x_tpm, y)[0][1], np.corrcoef(x_no_of_reads, y)[0][1],\
            np.corrcoef(x_num_eq_class, y_eq_class)[0][1])

def get_mean_and_standard_deviation(bootstrap_transcript_ids, count_matrix):
    mean_map = {}
    for ind in range(0, len(bootstrap_transcript_ids)):
        col = count_matrix[bootstrap_transcript_ids[ind]]
        mean_map[bootstrap_transcript_ids[ind]] = [np.mean(col), np.std(col)]
    return mean_map

def generate_input_file(mean_map, corr, TranscriptInNumOfClassesDict, transcript_quant):
    data_file = open("input_data.csv", "w")
    writer = csv.writer(data_file)
    writer.writerow(["transcript_id", "count", "tpm", "num_of_reads"])

    for key in mean_map.keys():
        ##### For running on smaller input size

        if key not in TranscriptInNumOfClassesDict:
            continue

        data_row = [
            ##### equivalence class count for transcript_id
            str(key),
            int(TranscriptInNumOfClassesDict[key]), \
            #float(transcript_quant[key][0]), \      
            #float(transcript_quant[key][1]), \
            #### tpm #####
            float(transcript_quant[key][2]), \
            ### number_of_reads ####
            float(transcript_quant[key][3]) \
            ]
        writer.writerow(data_row)
    data_file.close()

def calculated_shifted_mean_values(mean_map, corr, test_data_count,
        TranscriptInNumOfClassesDict, transcript_truth_count, transcript_quant, error_value_map):
    for key in mean_map.keys():
        if key not in TranscriptInNumOfClassesDict:
            continue

    for key in mean_map.keys():
        ##### For running on smaller input size

        if key not in TranscriptInNumOfClassesDict:
            continue

        if key in transcript_truth_count:
            mu, sigma = mean_map[key] # mean and standard deviation
            meanValue = transcript_truth_count[key]
            errorValue = meanValue - mu
            error_value_map[key] = errorValue


def write_calculated_data_to_files(mean_map, corr, test_data_count,
        TranscriptInNumOfClassesDict, transcript_truth_count, transcript_quant, error_value_map):
    valid_transcripts = open("valid_transcripts", "w")
    invalid_transcripts = open("invalid_transcripts", "w")
    data_file = open("regression_data.csv", "w")
    writer = csv.writer(data_file)
    writer.writerow(["valid", "count", "tpm", "num_of_reads"])

    total_count = 0
    for key in mean_map.keys():
        if key not in TranscriptInNumOfClassesDict:
            continue
        if key in transcript_truth_count:
            total_count += 1

    current_count = 0
    failed_count = 0
    for key in mean_map.keys():
        ##### For running on smaller input size

        if key not in TranscriptInNumOfClassesDict:
            continue

        if key in transcript_truth_count:
            mu, sigma = mean_map[key] # mean and standard deviation
            meanValue = transcript_truth_count[key]
            errorValue = meanValue - mu
            error_value_map[key] = errorValue
            row = key + "\t" + str(errorValue) + "\t" + str(meanValue) + "\t" + str(mu - 2 * sigma) \
                    + "\t" + str(mu) + "\t" + str(mu + 2*sigma) + "\n"
            #print("Running for key - ", key, " with row - ", row)
            current_count += 1
            if current_count < total_count - test_data_count :
                data_row = [
                        ##### equivalence class count for transcript_id
                        int(TranscriptInNumOfClassesDict[key]), \
                        #float(transcript_quant[key][0]), \      
                        #float(transcript_quant[key][1]), \
                        #### tpm #####
                        float(transcript_quant[key][2]), \
                        ### number_of_reads ####
                        float(transcript_quant[key][3]) \
                        ]
                if meanValue > mu - 2*sigma and meanValue < mu + 2*sigma :
                    valid_transcripts.write(row)
                    success = [errorValue]
                    success.extend(data_row)
                    writer.writerow(success)
                else:
                    invalid_transcripts.write(row)
                    failed_count += 1
                    failure = [errorValue]
                    failure.extend(data_row)
                    writer.writerow(failure)
    data_file.close()
    if test_data_count != 0:
        print("\nTotal failed transcripts - ", str(failed_count))

def apply_classification_model(test_data_count):
    characters = pd.read_csv("regression_data.csv")
    character_labels = characters.valid
    labels = list(set(character_labels))

    character_labels = np.array([labels.index(x) for x in character_labels])
    all_features = characters.iloc[:,1:]
    train_features = all_features[:-test_data_count]
    train_features = np.array(train_features)

    classifier = svm.SVR()
    classifier.fit(train_features, character_labels[:-test_data_count])
    #classifier.predict(all_features[-test_data_count:])
    print("\n")
    print("score - ", classifier.score(train_features, character_labels[:-test_data_count]))

def get_quant_map(quant_file, dir_name):
    print("Parsing Quant File - ", quant_file)
    firstLine = True
    transcript_quant = {}
    for line in open(quant_file, "r"):
        if firstLine:
            # ignore
            firstLine = False
        else:
            row = line.strip().replace("\n", "").split("\t")
            length = int(row[1])
            effective_length = float(row[2])
            tpm = float(row[3])
            num_of_reads = float(row[4])
            transcript_quant[dir_name + row[0]] = (length, effective_length, tpm, num_of_reads)
    return transcript_quant

def get_bootstrap_transcript_info(boot_strap_file, dir_name):
    print("Parsing Boot Strap File - ", boot_strap_file)
    transcript_ids = []
    count_matrix = {}
    firstLine = True
    for line in open(boot_strap_file, "r"):
        if firstLine:
            firstLine = False
            transcript_ids = [dir_name + each.strip().replace("\n", "") for each in line.split("\t")]
        else:
            row = [float(each) for each in line.strip().replace("\n", "").split("\t")]
            for trans_ind in range(0, len(row)):
                if transcript_ids[trans_ind] not in count_matrix:
                    count_matrix[transcript_ids[trans_ind]] = []
                prev_rows = count_matrix[transcript_ids[trans_ind]]
                prev_rows.append(row[trans_ind])
                count_matrix[transcript_ids[trans_ind]] = prev_rows
        
    return transcript_ids, count_matrix

def get_poly_truth_count(truth_value_file, dir_name):
    print("Parsing Truth Value File - ", truth_value_file)
    firstLine = True
    transcript_truth_count = {}
    for line in open(truth_value_file, "r"):
        if firstLine:
            # Ignore this value
            firstLine = False
        else:
            pair = [each.strip().replace("\n", "") for each in line.split("\t")]
            transcript_truth_count[dir_name + pair[0]] = int(pair[1])
    return transcript_truth_count

#converts index given in equivalence classes to transcript ID
def ActualTranscriptMap(temp, Transcripts, dir_name):
    for i in  range(1, len(temp)-1):
        temp[i] = Transcripts[int(temp[i])-3]
    return temp

def get_equivalence_class(equivalence_class_file, dir_name):
    print("Parsing Equivalence classes File - ", equivalence_class_file)
    equivalence_file = open(equivalence_class_file ,"r")
    all_lines = [] 
    each_line = ''
    count=0
    for line in equivalence_file:
        each_line = ''
        each_line += line.rstrip()
        all_lines.append(each_line)
    equivalence_file.close()
    TranscriptInNumOfClassesDict={}
    #filling the transcripts
    Transcripts=[]
    lTranscripts= int(all_lines[0])
    lClasses = int(all_lines[1])

    Transcripts = all_lines[2:2+lTranscripts]
    equiValenceClasses= []
    count =0
    for i in range(2+lTranscripts, len(all_lines)):
        temp=[]
        temp = all_lines[i].split("\t");
        ActualTranscriptMap(temp, Transcripts, dir_name)
        for i in range(1, len(temp)-1):
            if (dir_name + temp[i]) in TranscriptInNumOfClassesDict:
                TranscriptInNumOfClassesDict[dir_name + temp[i]] = TranscriptInNumOfClassesDict[
                        dir_name + temp[i]] +1
            else:
                TranscriptInNumOfClassesDict[dir_name + temp[i]] = 1
        equiValenceClasses.append(temp)

    return(TranscriptInNumOfClassesDict,equiValenceClasses)


def shift_mean(mean_map, error_value_map):
    count = 0
    for key in mean_map:
        if key not in error_value_map:
            continue
        value = mean_map[key]
        mean = value[0] + error_value_map[key]
        value[0] = mean
        mean_map[key] = value 
        count += 1


def create_combined_model():

    transcript_quant = {}
    bootstrap_transcript_ids = []
    count_matrix = {}
    transcript_truth_count = {}
    TranscriptInNumOfClassesDict = {}
    equiValenceClasses = []

    boot_strap_file = ["poly_mo/quant_bootstraps.tsv", "poly_ro/quant_bootstraps.tsv"]
    truth_value_file = ["poly_mo/poly_truth.tsv", "poly_ro/poly_truth.tsv"]
    quant_file = ["poly_mo/quant.sf", "poly_ro/quant.sf"]
    equivalence_class_file = ["poly_mo/eq_classes.txt", "poly_ro/eq_classes.txt"]

    
    for ind in range(0, 2):
        dir_name = ""
        pt = re.compile(r'(.*)/(.*)')
        dir_name = pt.search(truth_value_file[ind]).group(1)

        curr_transcript_quant = get_quant_map(quant_file[ind], dir_name)
        curr_bootstrap_transcript_ids, curr_count_matrix = get_bootstrap_transcript_info(
                boot_strap_file[ind], dir_name)
        curr_transcript_truth_count = get_poly_truth_count(truth_value_file[ind], dir_name)
        curr_TranscriptInNumOfClassesDict, curr_equiValenceClasses = get_equivalence_class(
            equivalence_class_file[ind], dir_name)

        transcript_quant.update(curr_transcript_quant)
        bootstrap_transcript_ids.extend(curr_bootstrap_transcript_ids)
        count_matrix.update(curr_count_matrix)
        transcript_truth_count.update(curr_transcript_truth_count)
        TranscriptInNumOfClassesDict.update(curr_TranscriptInNumOfClassesDict)
        equiValenceClasses.append(curr_equiValenceClasses)

    print("Number of transcripts : ", len(bootstrap_transcript_ids))
    print("Given number of the true reads", len(transcript_truth_count))

    mean_map = get_mean_and_standard_deviation(bootstrap_transcript_ids, count_matrix)
    corr = get_corelation(mean_map, transcript_quant, TranscriptInNumOfClassesDict)
    error_value_map = {}

    calculated_shifted_mean_values(mean_map, corr, 0, TranscriptInNumOfClassesDict,
            transcript_truth_count, transcript_quant, error_value_map)

    shift_mean(mean_map, error_value_map)
    corr = get_corelation(mean_map, transcript_quant, TranscriptInNumOfClassesDict)
    error_value_map = {}
    write_calculated_data_to_files(mean_map, corr, 0, TranscriptInNumOfClassesDict,
            transcript_truth_count, transcript_quant, error_value_map)
    
    characters = pd.read_csv("regression_data.csv")
    character_labels = characters.valid
    labels = list(set(character_labels))

    character_labels = np.array([labels.index(x) for x in character_labels])
    all_features = characters.iloc[:,1:]
    train_features = np.array(all_features)
    classifier = svm.SVR()
    classifier.fit(all_features, character_labels)
    return classifier
    

def main(boot_strap_file, truth_value_file, quant_file,  equivalence_class_file):

    ########## Building dict by parsing input files ######################
    
    dir_name = ""
    pt = re.compile(r'(.*)/(.*)')
    dir_name = pt.search(truth_value_file).group(1)

    transcript_quant = get_quant_map(quant_file, dir_name)
    bootstrap_transcript_ids, count_matrix = get_bootstrap_transcript_info(boot_strap_file, dir_name)
    transcript_truth_count = get_poly_truth_count(truth_value_file, dir_name)
    TranscriptInNumOfClassesDict,equiValenceClasses = get_equivalence_class(equivalence_class_file,
            dir_name)

    dir_name = ""
    pt = re.compile(r'(.*)/(.*)')
    dir_name = pt.search(truth_value_file).group(1)

    print("Number of transcripts : ", len(bootstrap_transcript_ids))
    print("Given number of the true reads", len(transcript_truth_count))

    mean_map = get_mean_and_standard_deviation(bootstrap_transcript_ids, count_matrix)
    corr = get_corelation(mean_map, transcript_quant, TranscriptInNumOfClassesDict)
    test_data_count = 2000
    error_value_map = {}
    write_calculated_data_to_files(mean_map, corr, test_data_count, TranscriptInNumOfClassesDict,
            transcript_truth_count, transcript_quant, error_value_map)
    
    apply_classification_model(test_data_count)

    shift_mean(mean_map, error_value_map)

    corr = get_corelation(mean_map, transcript_quant, TranscriptInNumOfClassesDict)
    write_calculated_data_to_files(mean_map, corr, test_data_count, TranscriptInNumOfClassesDict,
            transcript_truth_count, transcript_quant, error_value_map)
    apply_classification_model(test_data_count)

    model = create_combined_model()
    
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    load_and_run_model(boot_strap_file, quant_file, equivalence_class_file)

     
def load_and_run_model(boot_strap_file, quant_file, equivalence_class_file):
    transcript_quant = get_quant_map(quant_file, "")
    bootstrap_transcript_ids, count_matrix = get_bootstrap_transcript_info(boot_strap_file, "")
    TranscriptInNumOfClassesDict,equiValenceClasses = get_equivalence_class(equivalence_class_file, "")

    print("calculating mean map .......")
    mean_map = get_mean_and_standard_deviation(bootstrap_transcript_ids, count_matrix)
    corr = get_corelation(mean_map, transcript_quant, TranscriptInNumOfClassesDict)

    print("Generating input file for trained model .......")
    generate_input_file(mean_map, corr, TranscriptInNumOfClassesDict, transcript_quant)


    print("Loading model .......")
    filename = 'finalized_model.sav'
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    

    characters = pd.read_csv("input_data.csv")
    all_features = characters.iloc[:,1:]
    print("Predicting label values .......")
    print("Errors - > ",loaded_model.predict(all_features[:]))

    

if __name__ == "__main__":
    
    ############ Default Input Files ######################
    boot_strap_file = "poly_mo/quant_bootstraps.tsv"
    truth_value_file = "poly_mo/poly_truth.tsv"
    quant_file = "poly_mo/quant.sf"
    equivalence_class_file = "poly_mo/eq_classes.txt"

    ## Show normal distribution of bootstrap experiment for each transcript_id ##
    if len(sys.argv) == 5:
        boot_strap_file = sys.argv[1]
        truth_value_file = sys.argv[2]
        quant_file = sys.argv[3]
        equivalence_class_file = sys.argv[4]

    ############ Calling main #############
    main(boot_strap_file, truth_value_file, quant_file, equivalence_class_file)

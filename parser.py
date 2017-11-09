import sys
import numpy as np
import pandas as pd
import csv
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
    print("*******************************************************")
    print("Data for covarience calculation : ", len(y))
    print("Co-relation length", np.corrcoef(x_len, y)[0][1]) 
    print("Co-relation effective length", np.corrcoef(x_effec_len, y)[0][1])
    print("Co-relation tpm", np.corrcoef(x_tpm, y)[0][1])
    print("Co-relation num of reads", np.corrcoef(x_no_of_reads, y)[0][1])

    print("Equivalence class size ", len(y_eq_class))
    print("Co-relation for eq class", np.corrcoef(x_num_eq_class, y_eq_class)[0][1])
    print("*******************************************************")

    return (np.corrcoef(x_len, y)[0][1], np.corrcoef(x_effec_len, y)[0][1], \
            np.corrcoef(x_tpm, y)[0][1], np.corrcoef(x_no_of_reads, y)[0][1],\
            np.corrcoef(x_num_eq_class, y_eq_class)[0][1])

def get_mean_and_standard_deviation(bootstrap_transcript_ids):
    mean_map = {}
    for ind in range(0, len(bootstrap_transcript_ids)):
        col = count_matrix[bootstrap_transcript_ids[ind]]
        mean_map[bootstrap_transcript_ids[ind]] = [np.mean(col), np.std(col)]
    return mean_map

def write_calculated_data_to_files(mean_map, corr, test_data_count):
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
    for key in mean_map.keys():
        ##### For running on smaller input size

        if key not in TranscriptInNumOfClassesDict:
            continue

        if key in transcript_truth_count:
            mu, sigma = mean_map[key] # mean and standard deviation
            '''s = np.random.normal(mu, sigma, 1000)
            count, bins, ignored = plt.hist(s, 30, normed=True)
            plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                            np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
                linewidth=1, color='r')'''
            meanValue = transcript_truth_count[key]
            '''if show_graph:
                plt.axvline(x=meanValue,linewidth=2, color='k')  
                plt.axvline(x=mu+2*sigma,linewidth=2, color='g')
                plt.axvline(x=mu-2*sigma,linewidth=2, color='g')
                plt.show()'''
            row = key + "\t" + str(meanValue) + "\t" + str(mu - 2 * sigma) \
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
            else:
               tfactor_count = 1
               tfactor_tpm = 1
               tfactor_no_of_reads = 1
               data_row = [
                        tfactor_count * int(TranscriptInNumOfClassesDict[key]),\
                        #float(transcript_quant[key][0]), \
                        #float(transcript_quant[key][1]), \
                        tfactor_tpm * float(transcript_quant[key][2]), \
                        tfactor_no_of_reads * float(transcript_quant[key][3]) \
                        ] 
            if meanValue > mu - 2*sigma and meanValue < mu + 2*sigma :
                valid_transcripts.write(row)
                success = [1]
                success.extend(data_row)
                writer.writerow(success)
            else:
                invalid_transcripts.write(row)
                failure = [0]
                failure.extend(data_row)
                writer.writerow(failure)

    data_file.close()

def apply_classification_model(test_data_count):
    characters = pd.read_csv("regression_data.csv")
    character_labels = characters.valid
    labels = list(set(character_labels))

    character_labels = np.array([labels.index(x) for x in character_labels])
    all_features = characters.iloc[:,1:]
    train_features = all_features[:-test_data_count]
    train_features = np.array(train_features)

    classifier = svm.SVC()
    classifier.fit(train_features, character_labels[:-test_data_count])

    results = classifier.predict(all_features[-test_data_count:])
    num_correct = (results == character_labels[-test_data_count:]).sum()
    recall = num_correct / len(character_labels[-test_data_count:])
    print("model accuracy (%): ", recall * 100, "%")

    # Plot outputs
    plt.scatter(all_features[-test_data_count:].iloc[:, :1], \
        characters.iloc[:,:1][-test_data_count:] ,  color='black')
    plt.plot(all_features[-test_data_count:].iloc[:, -3:].sum(axis=1), results, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show() 

def get_quant_map(quant_file):
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
            transcript_quant[row[0]] = (length, effective_length, tpm, num_of_reads)
    return transcript_quant

def get_bootstrap_transcript_info(boot_strap_file):
    print("Parsing Boot Strap File - ", boot_strap_file)
    transcript_ids = []
    count_matrix = {}
    firstLine = True
    for line in open(boot_strap_file, "r"):
        if firstLine:
            firstLine = False
            transcript_ids = [each.strip().replace("\n", "") for each in line.split("\t")]
        else:
            row = [float(each) for each in line.strip().replace("\n", "").split("\t")]
            for trans_ind in range(0, len(row)):
                if transcript_ids[trans_ind] not in count_matrix:
                    count_matrix[transcript_ids[trans_ind]] = []
                prev_rows = count_matrix[transcript_ids[trans_ind]]
                prev_rows.append(row[trans_ind])
                count_matrix[transcript_ids[trans_ind]] = prev_rows
        
    return transcript_ids, count_matrix

def get_poly_truth_count(truth_value_file):
    print("Parsing Truth Value File - ", truth_value_file)
    firstLine = True
    transcript_truth_count = {}
    for line in open(truth_value_file, "r"):
        if firstLine:
            # Ignore this value
            firstLine = False
        else:
            pair = [each.strip().replace("\n", "") for each in line.split("\t")]
            transcript_truth_count[pair[0]] = int(pair[1])
    return transcript_truth_count

#converts index given in equivalence classes to transcript ID
def ActualTranscriptMap(temp, Transcripts):
    for i in  range(1, len(temp)-1):
        temp[i] = Transcripts[int(temp[i])-3]
    return temp

def get_equivalence_class(equivalence_class_file):
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
        ActualTranscriptMap(temp, Transcripts)
        for i in range(1, len(temp)-1):
            if temp[i] in TranscriptInNumOfClassesDict:
                TranscriptInNumOfClassesDict[temp[i]] = TranscriptInNumOfClassesDict[temp[i]] +1
            else:
                TranscriptInNumOfClassesDict[temp[i]] = 1
        equiValenceClasses.append(temp)

    return(TranscriptInNumOfClassesDict,equiValenceClasses)


def main(bootstrap_transcript_ids, count_matrix, transcript_truth_count, \
        transcript_quant, show_graph, TranscriptInNumOfClassesDict):
    print("Number of transcripts : ", len(bootstrap_transcript_ids))
    print("Given number of the true reads", len(transcript_truth_count))

    mean_map = get_mean_and_standard_deviation(bootstrap_transcript_ids)
    corr = get_corelation(mean_map, transcript_quant, TranscriptInNumOfClassesDict)
    test_data_count = 2000
    write_calculated_data_to_files(mean_map, corr, test_data_count)
    apply_classification_model(test_data_count)


if __name__ == "__main__":
    
    ############ Input Files ######################
    boot_strap_file = "poly_ro/quant_bootstraps.tsv"
    truth_value_file = "poly_ro/poly_truth.tsv"
    quant_file = "poly_ro/quant.sf"
    equivalence_class_file = "poly_ro/eq_classes.txt"

    ########## Building dict by parsing input files ######################
    transcript_quant = get_quant_map(quant_file)
    bootstarp_transcript_ids, count_matrix = get_bootstrap_transcript_info(  \
        boot_strap_file)
    transcript_truth_count = get_poly_truth_count(truth_value_file)
    TranscriptInNumOfClassesDict,equiValenceClasses = get_equivalence_class( \
        equivalence_class_file)
   
    ## Show normal distribution of bootstrap experiment for each transcript_id ##
    show_graph = False
    if len(sys.argv) == 2:
        show_graph = True

    ############ Calling main #############
    main(bootstarp_transcript_ids, count_matrix, transcript_truth_count,  \
        transcript_quant, show_graph, TranscriptInNumOfClassesDict)


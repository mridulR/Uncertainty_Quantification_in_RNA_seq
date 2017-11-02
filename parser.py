import sys
import numpy as np
import matplotlib.pyplot as plt

def main(bootstrap_transcript_ids, count_matrix, transcript_truth_count, transcript_quant, show_graph):
    print("Number of transcripts : ", len(bootstrap_transcript_ids))
    print("Length of the true reads", len(transcript_truth_count))
    mean_map = {}
    for ind in range(0, len(bootstrap_transcript_ids)):
        col = count_matrix[bootstrap_transcript_ids[ind]]
        mean_map[bootstrap_transcript_ids[ind]] = [np.mean(col), np.std(col)]

    valid_transcripts = open("valid_transcripts", "w")
    invalid_transcripts = open("invalid_transcripts", "w")

    for key in mean_map.keys():
        if key in transcript_truth_count:
            mu, sigma = mean_map[key] # mean and standard deviation
            s = np.random.normal(mu, sigma, 1000)
            count, bins, ignored = plt.hist(s, 30, normed=True)
            plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                            np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
                linewidth=1, color='r')
            meanValue = transcript_truth_count[key]
            if show_graph:
                plt.axvline(x=meanValue,linewidth=2, color='k')  
                plt.axvline(x=mu+2*sigma,linewidth=2, color='g')
                plt.axvline(x=mu-2*sigma,linewidth=2, color='g')
                plt.show()
            row = key + "\t" + str(meanValue) + "\t" + str(mu - 2 * sigma) \
                    + "\t" + str(mu) + "\t" + str(mu + 2*sigma) + "\n"
            print("Running for key - ", key, " with row - ", row)
            if meanValue > mu - 2*sigma and meanValue < mu + 2*sigma :
                valid_transcripts.write(row)
            else:
                invalid_transcripts.write(row)

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
            tpm = float(row[2])
            num_of_reads = float(row[2])
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


if __name__ == "__main__":
    boot_strap_file = "poly_mo/quant_bootstraps.tsv"
    truth_value_file = "poly_mo/poly_truth.tsv"
    quant_file = "poly_mo/quant.sf"

    transcript_quant = get_quant_map(quant_file)
    bootstarp_transcript_ids, count_matrix = get_bootstrap_transcript_info(boot_strap_file)
    transcript_truth_count = get_poly_truth_count(truth_value_file)
    
    show_graph = False
    if len(sys.argv) == 2:
        show_graph = True

    main(bootstarp_transcript_ids, count_matrix, transcript_truth_count, transcript_quant, show_graph)

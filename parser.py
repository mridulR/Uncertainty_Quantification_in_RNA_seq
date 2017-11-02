import numpy as np
import matplotlib.pyplot as plt

def main():
    file_name = "poly_mo/quant_bootstraps.tsv"
    print("Parsing quant_bootstraps.tsv file .......")
    boot_strap_exp = -1
    transcript_ids = []
    count_matrix = {}
    mean_map = {}
    file_ = open(file_name, "r")
    for line in file_:
        boot_strap_exp += 1
        if boot_strap_exp == 0:
            transcript_ids = line.split("\t")
        else:
            row = [float(each) for each in line.strip().replace("\n", "").split("\t")]
            count_matrix[boot_strap_exp] = row
    print("Number of transcripts : ", len(transcript_ids))
    print("Number of bootstrap experiments : ", boot_strap_exp)
    print("transcripts = ",len(transcript_ids))



    file_name = "poly_mo/poly_truth.tsv"
    print("Parsing quant_bootstpoly_truth.tsv file .......")

    transcriptTrueReads={}
    file_ = open(file_name, "r")
    init=0
    for line in file_:
        
        if(init > 0):
            tReads = line.strip().split('\t')
            # print(tReads)
            transcriptTrueReads[tReads[0]] = int(tReads[1]) 
        init = init + 1
    print("length of the true reads", len(transcriptTrueReads))

    for ind in range(0, len(transcript_ids)):
        col = []
        for num in range(1, boot_strap_exp + 1):
            row = count_matrix[num]
            value = row[ind]
            col.append(value)
        mean_map[transcript_ids[ind]] = [np.mean(col), np.std(col)]

    
    for key in mean_map.keys():
        mu, sigma = mean_map[key] # mean and standard deviation
        s = np.random.normal(mu, sigma, 1000)
        count, bins, ignored = plt.hist(s, 30, normed=True)
        plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                            np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
                linewidth=1, color='r')
        if key in transcriptTrueReads:
            meanValue = transcriptTrueReads[key]
            plt.axvline(x=meanValue,linewidth=2, color='k')  
        plt.axvline(x=mu+2*sigma,linewidth=2, color='g')
        plt.axvline(x=mu-2*sigma,linewidth=2, color='g')
        plt.show()
    


    

if __name__ == "__main__":
    main()

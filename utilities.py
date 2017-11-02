#!/usr/bin/env python
"""
Usage:
from utilities import TranscriptRefData

t = TranscriptRefData ()
transcript_id = "ENST00000382410"
count = t.get_count(transcript_id)
len= t.get_len(transcript_id)
...
"""

class TranscriptRefData:

	def __init__(self, truthf= "poly_truth.tsv", quantf = "quant.sf"):
		print ("parsing: %s, %s"%(truthf, quantf))
		self.polytruth_file = truthf
		self.quant_file = quantf
		#{"transcript_id" : count}
		self.truth_val = {}
		#{"transcript_id" : (len, effective_len, tpm, num_reads)}
		self.quant_val = {}
		self.parse_polytruthfile()
		self.parse_quantfile()

	def parse_polytruthfile(self):
		with open (self.polytruth_file) as ip:
			f = ip.read().strip().split("\n")
			del f[0] #remove the header
			print ("input len: %s"%(len (f)))
			for each in f:
				tid = ((each.split())[0]).strip()
				value = int((each.split()[1]).strip())
				self.truth_val [tid] = value
			print ("parsed len: %d"%(len(self.truth_val)))


	def parse_quantfile(self):
		with open (self.quant_file) as ip:
			f = ip.read().strip().split("\n")
			del f[0] #remove header
			print (f[0], "\n", f[-1])
			for each in f:
				tid = ((each.split())[0]).strip()
				length = int((each.split()[1]).strip())
				eff_length = float((each.split()[2]).strip())
				tpm = float((each.split()[3]).strip())
				num_reads = float((each.split()[4]).strip())
				self.quant_val [tid] = (length, eff_length, tpm, num_reads)
			print (self.quant_val[f[0].split()[0].strip()], self.quant_val[f[-1].split()[0].strip()])
			print ("input len: %s"%(len (f)))
			print ("parsed len: %d"%(len(self.quant_val)))

	def get_count(self, tid):
		try:
			if tid in self.truth_val.keys():
				return self.truth_val[tid]
		except Exception as e:
			print (e)  


	def get_len(self, tid):
		try:
			if tid in self.quant_val.keys():
				return self.quant_val[tid][0]
		except Exception as e:
			print (e)  

	def get_efflen(self, tid):
		try:
			if tid in self.quant_val.keys():
				return self.quant_val[tid][1]
		except Exception as e:
			print (e) 

	def get_tpm(self, tid):
		try:
			if tid in self.quant_val.keys():
				return self.quant_val[tid][2]
		except Exception as e:
			print (e)  

	def get_numreads(self, tid):
		try:
			if tid in self.quant_val.keys():
				return self.quant_val[tid][3]
		except Exception as e:
			print (e)  
"""
	Observations: not all transcript IDs are present in both files.
		      quant has more transcripts, truth has lesser.
		      only a part of transcripts from both files are common
		      	
"""
if __name__=="__main__":
	print ("Parsing polytruth..")
	tf = input ("enter truth data file: ")
	qf = input ("enter quant data file: ")
	if not tf == " " or qf == " ":
		t = TranscriptRefData()
	else:
		t = TranscriptRefData(tf, qf)

	"""
	for each in t.truth_val.keys():
		print (t.get_count(each), t.get_len(each), t.get_efflen(each), t.get_tpm(each), t.get_numreads(each))
	"""
	tkeys = t.truth_val.keys()
	qkeys = t.quant_val.keys()
	diff = [k for k in qkeys if k not in tkeys]
#diff = set (tkeys).symmetric_difference(set (qkeys))
	print ("transcripts in truth file:%d"%len(tkeys))
	print ("transcripts in quants file:%d"%len(qkeys))
	print ("transcripts diff:%d"%(len(diff)))

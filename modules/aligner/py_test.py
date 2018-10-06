from ssw_aligner import ssw_api

alignment = ssw_api.Alignment()
filt = ssw_api.Filter()
aligner = ssw_api.Aligner(4, 2, 4, 2)
aligner.SetReferenceSequence("tttt", 4)
flag = aligner.Align_cpp("ttAtt", filt, alignment, 15)
print(alignment.__dir__)
print(flag)
print(alignment.cigar_string)
print(alignment.best_score)
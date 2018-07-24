input_path = "/home/ryan/data/Nanopore/BAM_VCF_concordance/chr19.vcf"
output_path = "/home/ryan/data/Nanopore/BAM_VCF_concordance/chr19_FN_only.vcf"

with open(input_path, 'r') as input_file:
    with open(output_path, 'w') as output_file:
        for line in input_file:
            if line.startswith("#"):
                output_file.write(line)

            elif "FN" in line:
                output_file.write(line)

            else:
                pass


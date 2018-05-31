# V.I.S.I.O.N.
Variant Identification System to Improve Outcomes in Nanopore (V.I.S.I.O.N)

# Calling Variants with Vision

## Set-up

Download and Install Vision from github:
```
mkdir vision
cd vision
git clone https://github.com/CGL-Deeplearning/vision.git
```
### Install Vision dependencies:
Vision uses python3 and specifically these packages: h5py, tqdm, pysam, pytorch, and pillow
All packages can be installed with pip (pip3 or python3 -m pip)

```
sudo -H pip3 install h5py tqdm pysam scipy pillow 
```


## Define path variables 
For convenience, we use bash variables define the paths to the files required for the Vision variant calling pipeline.

Ideally we will create one project directory located at the 'BASE' path variable. However if files are located elsewhere on disk, change path variables accordingly.
```
export BASE="${HOME}/<PROJECT NAME/PATH>"
export INPUT_DIR="${BASE}/input"

#path to bam, bam index (.bai) file must also be in the same directory
export BAM="${INPUT_DIR}/<BAM_FILE_NAME_WITH_SOURCE_AND_COVERAGE>.bam"

export REF="${INPUT_DIR}/<REFERENCE_FILE_NAME>fa.gz"

export CONF_VCF="${INPUT_DIR}/<REF_VCF_FILE_NAME_WITH_SOURCE_AND_INFO>.vcf"

#path to confident_bed file, bed file must be zipped and indexed
export CONF_BED="${INPUT_DIR}/<REF_BED_FILE_NAME_(ZIPPED)>bed.gz"

OUTPUT_DIR="${BASE}/output"
IMAGE_DIR="${OUTPUT_DIR}/images"
LOG_DIR="${OUTPUT_DIR}/logs"
```
#### Make directories
Based on defined bash variables, create the organized project direct and place/download respective files in defined locations. 
Note: creating a data directory might not be needed if mapping input files from existing locations.
```
mkdir -p "${DATA_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${IMAGE_DIR}"
mkdir -p "${LOG_DIR}"
```

Vision cannot read the compressed fa, so uncompress if necessary into a writable directory and index it.
```
export UNCOMPRESSED_REF="${INPUT_DIR}/<FILE_NAME UNCOMPRESSED REF>.fa"
zcat < "${REF}" >"${UNCOMPRESSED_REF}"
samtools faidx "${UNCOMPRESSED_REF}"
```

## Vision Labeled Image Generator (Training):
This command creates labeled training images from input BAM, Reference FASTA and truth VCF file. This command is used for model training purposes. The process is:
- Find candidates that can be variants
- Label candidates using the VCF
- Create images for each candidate

Input:
- BAM file: Alignment of a genome
- REF file: The reference FASTA file used in the alignment
- VCF file: A truth VCF file
- BED file: A confident bed file. If confident_bed is passed it will only generate train set for those region.

Output:
- H5PY files: Containing images and their label of the genome.
- CSV file: Containing records of images and their location in the H5PY file.

Note: to generate images for only one chromosome use flag ``--chromosome_name chr#`` or ``--chromosome_name #`` depending on how your input files are formatted. For a whole genome run, ommit flag completely.

Vision at this time cannot specify multiple chromosomes using the flag. 
Specify number of threads to use on your machine using ``--max_threads <num>``
The output is a csv file containing records of images and their location in the H5PY file. this csv file will be used in the next step. 

```
python3 labeled_image_generator.py \
--bam "${BAM}" \
--ref "${UNCOMPRESSED_REF}" \
--vcf "${CONF_VCF}" \
--confident_bed "${CONF_BED}" \
--output_dir "${IMAGE_DIR}" \
--chromosome_name chr19\
--max_threads 64
```

Predict Image Generator (Testing):
To call variants in testing mode(i.e when not training a model) use predict_image_generator.py instead of labeled_image_generator.py:


```
python3 predict_image_generator.py \
--bam "${BAM}" \
--ref "${UNCOMPRESSED_REF}" \
--vcf "${CONF_VCF}" \
--confident_bed "${CONF_BED}" \
--output_dir "${IMAGE_DIR}" \
--max_threads 64
```


## Predict/Call Variants Using Vision:
Vision's predict.py uses a trained model to call variants on a given set of images generated from the genome.
The process is:
- Create a prediction table/dictionary using a trained neural network
- Convert those predictions to a VCF file

INPUT:
- A trained model
- Set of images for prediction

Output:
- A VCF file containing all the variants.

To use GPU, use the flag `gpu_mode 1`, otherwise omit flag for CPU-only usage
Set num_workers (threads) and batch_size according to your machine.
{TBA: recommendations based on specs}
Use the nvidia-smi command to verify that the driver is running properly: `nvidia-smi`


```
PREDICT_CSV="<PATH TO CSV FROM IMAGE GENERATOR>"
MODEL="<PATH TO MODEL>"

python3 predict.py \
--test_file "${PREDICT_CSV}" \
--batch_size 512 \
--model_path "${MODEL}" \
--gpu_mode 1 \
--num_workers 32
```

## Evaluate Variant Calls using hap.py:
Use tool hap.py to compare Vision called variants with known variants. 
Note: to use hap.py, bed file needs to be zipped (bgzip) and indexed (tabix) and reference needs to be uncompressed.
To specify a chromosome region, use the flag ``-l chr19`` or ``-l 19``

Using hap.py installed locally:
```
OUTPUT_VCF="<PATH TO VCF GENERATED BY PREDICT.PY>"
HAPPY="<PATH_TO_HAP.PY>" 
python "${HAPPY}" \
"${CONF_VCF}" \
"${OUTPUT_VCF}" \
--preprocess-truth \
-f "${CONF_BED}" \
-o "${OUTPUT_DIR}/happy.output" \
-r "${UNCOMPRESSED_REF}" \
-l 19

```

Using hap.py docker image:
Alternatively, if you don't have hap.py installed locally, you can use the pre-complied docker image. 

```
sudo apt-get install docker.io
sudo docker pull pkrusche/hap.py
sudo docker run -it \
-v "${DATA_DIR}:${DATA_DIR}" \
-v "${OUTPUT_DIR}:${OUTPUT_DIR}" \
pkrusche/hap.py /opt/hap.py/bin/hap.py \
  "${CONF_VCF}" \
  "${OUTPUT_VCF}" \
  --preprocess-truth \
  -f "${TRUTH_BED}" \
  -o "${OUTPUT_DIR}/testhappy/happy.output" \
  -r "${UNCOMPRESSED_REF}" \
  -l 19
```

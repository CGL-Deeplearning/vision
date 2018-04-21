import argparse
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from modules.models.inception import Inception3
from modules.core.dataloader_predict import PileupDataset, TextColor
from collections import defaultdict
from modules.handlers.VcfWriter import VCFWriter
import operator
import os


def predict(test_file, batch_size, model_path, gpu_mode, num_workers):
    prediction_dict = defaultdict(list)
    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)

    test_dset = PileupDataset(test_file, transformations)
    testloader = DataLoader(test_dset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=gpu_mode # CUDA only
                            )

    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)
    if gpu_mode is False:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        # print('loaded state dict:', state_dict.keys())
        # print('\nIn state dict keys there is an extra word inserted by model parallel: "module.". We remove it here:')
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model = Inception3()
        model.load_state_dict(new_state_dict)
        model.cpu()
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model = Inception3()
        model.load_state_dict(new_state_dict)
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()

    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    for counter, (images, records) in enumerate(testloader):
        images = Variable(images, volatile=True)

        if gpu_mode:
            images = images.cuda()

        preds = model(images)
        m = nn.Softmax(dim=1)
        soft_probs = m(preds)
        preds = soft_probs.cpu()

        for i in range(0, preds.size(0)):
            rec = records[i]
            chr_name, pos_st, pos_end, ref, alt1, alt2, rec_type = rec.rstrip().split('\t')[0:7]
            probs = preds[i].data.numpy()
            prob_hom, prob_het, prob_hom_alt = probs
            prediction_dict[pos_st].append((chr_name, pos_st, pos_end, ref, alt1, alt2, rec_type, prob_hom, prob_het, prob_hom_alt))
            # print((chr_name, pos_st, pos_end, ref, alt1, alt2, rec_type, prob_hom, prob_het, prob_hom_alt))
        sys.stderr.write(TextColor.BLUE + " BATCHES DONE: " + str(counter+1) + "/" + str(len(testloader)) + "\n" + TextColor.END)

    return prediction_dict


def produce_vcf(prediction_dictionary, bam_file_path, sample_name, output_dir):
    vcf_writer = VCFWriter(bam_file_path, sample_name, output_dir)

    all_calls = []
    for pos in sorted(prediction_dictionary.keys()):
        records = prediction_dictionary[pos]
        if len(records) > 1:
            chrm, st_pos, end_pos, ref, alts, genotype, qual, gq = vcf_writer.get_genotype_for_multiple_allele(records)
        else:
            chrm, st_pos, end_pos, ref, alts, genotype, qual, gq = vcf_writer.get_genotype_for_single_allele(records)

        all_calls.append((chrm, int(st_pos), int(end_pos), ref, alts, genotype, qual, gq))

    all_calls.sort(key=operator.itemgetter(1))
    last_end = 0
    for record in all_calls:
        rec_filter = vcf_writer.get_filter(record, last_end)
        record = vcf_writer.get_proper_alleles(record)
        chrm, st_pos, end_pos, ref, alt_field, genotype, phred_qual, phred_gq = record
        if genotype != '0/0':
            last_end = end_pos
        vcf_writer.write_vcf_record(chrm, st_pos, end_pos, ref, alt_field, genotype, phred_qual, phred_gq, rec_filter)


def handle_output_directory(output_dir):
    """
    Process the output directory and return a valid directory where we save the output
    :param output_dir: Output directory path
    :return:
    """
    # process the output directory
    if output_dir[-1] != "/":
        output_dir += "/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    return output_dir


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Testing data description csv file."
    )
    parser.add_argument(
        "--bam_file",
        type=str,
        required=True,
        help="Path to the BAM file."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=100,
        help="Batch size for testing, default is 100."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=4,
        help="Batch size for testing, default is 100."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='./CNN.pkl',
        help="Saved model path."
    )
    parser.add_argument(
        "--gpu_mode",
        type=bool,
        default=False,
        help="If true then cuda is on."
    )
    parser.add_argument(
        "--sample_name",
        type=str,
        required=False,
        default='NA12878',
        help="Sample name of the sequence."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default='vcf_output',
        help="Output directory."
    )
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.output_dir = handle_output_directory(FLAGS.output_dir)
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "SAMPLE NAME: " + FLAGS.sample_name + "\n")
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "PLEASE USE --sample_name TO CHANGE SAMPLE NAME.\n")
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "OUTPUT DIRECTORY: " + FLAGS.output_dir + "\n")
    record_dict = predict(FLAGS.test_file, FLAGS.batch_size, FLAGS.model_path, FLAGS.gpu_mode, FLAGS.num_workers)
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "PREDICTION COMPLETED SUCCESSFULLY.\n")
    produce_vcf(record_dict, FLAGS.bam_file, FLAGS.sample_name, FLAGS.output_dir)
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "FINISHED CALLING VARIANT.\n")




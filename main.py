import logging
from curriculum_generator import CurriculumGenerator
import os
import shutil
import csv

from PIL import Image

# Simple demo for the curriculum generator class. It saves samples in three folders:
# - samples/sets/{train,val,test}/n: samples for tasks 0..n are extracted consecutively and sorted to different folders.
# - samples/curriculum: each sample is generated in teacher mode and numerated progressively in the same order the teacher would provide them.
# - samples/shuffled_curriculum: at each step a random task is selected and samples are extracted until every task is exhausted. There is a 0.3 probability of annotating samples with a corrupted task_id.

# In every case, the file stats.txt contains all the hyperparameters used during generation and annotations.csv contains annotations for each sample.
# Annotations.csv contains: filename (relative to the csv position), task_id, label (+1 positive, -1 negative, 0 unsupervised sample), symbolic representation (a python-interpretable string)
# In continual learning mode samples are meant to be observed in numerical order (the same as they appear in the annotations.csv file), in batch learning mode samples can be shuffled randomly.

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    cg = CurriculumGenerator("config.yml", "hierarchies.yml", logger=logger)

    shutil.rmtree("samples", ignore_errors=True)
    os.makedirs("samples/shuffled_curriculum", exist_ok=True)
    os.makedirs("samples/sets", exist_ok=True)
    os.makedirs("samples/sets/train", exist_ok=True)
    os.makedirs("samples/sets/val", exist_ok=True)
    os.makedirs("samples/sets/test", exist_ok=True)

    os.makedirs("samples/shuffled_curriculum/train/0", exist_ok=True)
    os.makedirs("samples/shuffled_curriculum/val/0", exist_ok=True)
    os.makedirs("samples/shuffled_curriculum/test/0", exist_ok=True)

    with open("samples/sets/stats.txt", "w", newline="\n", encoding="utf-8") as file:
        file.write("General parameters:\n")
        file.write("\tSeed: {}\n".format(cg.config["seed"]))
        file.write("\tCanvas size: {}\n".format(cg.config["canvas_size"]))
        file.write("\tPadding: {}\n".format(cg.config["padding"]))
        file.write("\tBackground color: {}\n".format(cg.config["bg_color"]))
        file.write("\tColors: {}\n".format(cg.config["colors"]))
        file.write("\tSizes: {}\n".format(cg.config["sizes"]))
        file.write("\tSize noise (uniform): {}\n".format(cg.config["size_noise"]))
        file.write("\tHue noise (Gaussian): mu=0, sigma={}\n".format(cg.config["h_sigma"]))
        file.write("\tSaturation noise (Gaussian): mu=0, sigma={}\n".format(cg.config["s_sigma"]))
        file.write("\tValue noise (Gaussian): mu=0, sigma={}\n".format(cg.config["v_sigma"]))
        file.write("\tTask id noise probability: {}\n".format(0.0))
        file.write("\tMode: independent datasets\n")
        file.write("\tTask type: binary classification\n")


        file.write("Task specific parameters:\n")
        for i in range(len(cg.tasks)):
            file.write("\tTask {}:\n".format(i))
            file.write("\t\tName: {}\n".format(cg.tasks[i].name))
            file.write("\t\tSeed: {}\n".format(cg.tasks[i].seed))
            file.write("\t\tPatience: {}\n".format(cg.tasks[i].patience))
            file.write("\t\tGamma: {}\n".format(cg.tasks[i].gamma))
            file.write("\t\tBeta: {}\n".format(cg.tasks[i].beta))
            file.write("\t\tInject color noise: {}\n".format(cg.tasks[i].noisy_color))
            file.write("\t\tInject size noise: {}\n".format(cg.tasks[i].noisy_size))
            file.write("\t\tTotal samples: {}\n".format(cg.tasks[i].total_samples))
            file.write("\t\tUnique samples: {}\n".format(len(cg.tasks[i].symbol_set)))

            pos_samples = len([x for x in cg.tasks[i].symbol_set if x[1] == 1])
            file.write("\t\tTotal positive samples (including unsupervised): {}\n".format(pos_samples))
            file.write("\t\tTotal negative samples (including unsupervised): {}\n".format(cg.tasks[i].total_samples - pos_samples))

            file.write("\t\tRejected samples:\n")
            file.write("\t\t\tPositives: {} (rule violation), {} (already sampled)\n".format(cg.tasks[i].rejected["positive"]["rule"], cg.tasks[i].rejected["positive"]["existing"]))
            file.write("\t\t\tNegatives: {} (rule violation), {} (already sampled)\n".format(
                cg.tasks[i].rejected["negative"]["rule"], cg.tasks[i].rejected["negative"]["existing"]))

            file.write("\t\tTrain samples ({}): expected {}, actual {}\n".format(cg.tasks[i].train_split, cg.tasks[i].requested_samples["train"], cg.tasks[i].samples["train"]))
            file.write("\t\tVal samples ({}): expected {}, actual {}\n".format(cg.tasks[i].val_split,
                                                                                 cg.tasks[i].requested_samples["val"],
                                                                                 cg.tasks[i].samples["val"]))
            test_split = 1.0 - cg.tasks[i].train_split - cg.tasks[i].val_split
            file.write("\t\tTest samples ({}): expected {}, actual {}\n".format(test_split,
                                                                                 cg.tasks[i].requested_samples["test"],
                                                                                 cg.tasks[i].samples["test"]))

            file.write("\t\tSampling with{} replacement\n".format("" if cg.tasks[i].with_replacement else "out"))


    for split in ["train", "val", "test"]:
        with open("samples/sets/{}/annotations.csv".format(split), "w", newline="\n", encoding="utf-8") as csvfile:
            fieldnames = ["filename", "task_id", "label", "supervised", "symbol"]
            fieldnames.extend(cg.config["concept_list"])

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()

            for i in range(len(cg.tasks)):
                os.mkdir("samples/sets/{}/{}".format(split, i))
                for j, sample in enumerate(cg.get_stream(i, split)): # NOTE: get_batch() can be called an infinite number of times, since it randomly samples the split sets.
                    sample_img, label, supervised, task_id, symbol, concepts = sample

                    row = {"filename": "{}/{:04d}.png".format(i, j), "task_id": task_id, "label": label, "supervised": supervised,
                                     "symbol": symbol}
                    for k, v in concepts.items():
                        row[k] = v

                    sample_img.save("samples/sets/{}/{}/{:04d}.png".format(split, i, j), "PNG")
                    writer.writerow(row)

    cg.reset()
    with open("samples/shuffled_curriculum/stats.txt", "w", newline="\n", encoding="utf-8") as file:
        file.write("General parameters:\n")
        file.write("\tSeed: {}\n".format(cg.config["seed"]))
        file.write("\tCanvas size: {}\n".format(cg.config["canvas_size"]))
        file.write("\tPadding: {}\n".format(cg.config["padding"]))
        file.write("\tBackground color: {}\n".format(cg.config["bg_color"]))
        file.write("\tColors: {}\n".format(cg.config["colors"]))
        file.write("\tSizes: {}\n".format(cg.config["sizes"]))
        file.write("\tSize noise (uniform): {}\n".format(cg.config["size_noise"]))
        file.write("\tHue noise (Gaussian): mu=0, sigma={}\n".format(cg.config["h_sigma"]))
        file.write("\tSaturation noise (Gaussian): mu=0, sigma={}\n".format(cg.config["s_sigma"]))
        file.write("\tValue noise (Gaussian): mu=0, sigma={}\n".format(cg.config["v_sigma"]))
        file.write("\tTask id noise probability: {}\n".format(0.3))
        file.write("\tMode: task-shuffled continual learning\n")
        file.write("\tTask type: binary classification\n")

        file.write("Task specific parameters:\n")
        for i in range(len(cg.tasks)):
            file.write("\tTask {}:\n".format(i))
            file.write("\t\tName: {}\n".format(cg.tasks[i].name))
            file.write("\t\tSeed: {}\n".format(cg.tasks[i].seed))
            file.write("\t\tPatience: {}\n".format(cg.tasks[i].patience))
            file.write("\t\tGamma: {}\n".format(cg.tasks[i].gamma))
            file.write("\t\tBeta: {}\n".format(cg.tasks[i].beta))
            file.write("\t\tInject color noise: {}\n".format(cg.tasks[i].noisy_color))
            file.write("\t\tInject size noise: {}\n".format(cg.tasks[i].noisy_size))
            file.write("\t\tTotal samples: {}\n".format(cg.tasks[i].total_samples))
            file.write("\t\tUnique samples: {}\n".format(len(cg.tasks[i].symbol_set)))

            pos_samples = len([x for x in cg.tasks[i].symbol_set if x[1] == 1])
            file.write("\t\tTotal positive samples (including unsupervised): {}\n".format(pos_samples))
            file.write("\t\tTotal negative samples (including unsupervised): {}\n".format(
                cg.tasks[i].total_samples - pos_samples))

            file.write("\t\tRejected samples:\n")
            file.write("\t\t\tPositives: {} (rule violation), {} (already sampled)\n".format(cg.tasks[i].rejected["positive"]["rule"], cg.tasks[i].rejected["positive"]["existing"]))
            file.write("\t\t\tNegatives: {} (rule violation), {} (already sampled)\n".format(
                cg.tasks[i].rejected["negative"]["rule"], cg.tasks[i].rejected["negative"]["existing"]))

            file.write("\t\tTrain samples ({}): expected {}, actual {}\n".format(cg.tasks[i].train_split,
                                                                                 cg.tasks[i].requested_samples["train"],
                                                                                 cg.tasks[i].samples["train"]))
            file.write("\t\tVal samples ({}): expected {}, actual {}\n".format(cg.tasks[i].val_split,
                                                                               cg.tasks[i].requested_samples["val"],
                                                                               cg.tasks[i].samples["val"]))
            test_split = 1.0 - cg.tasks[i].train_split - cg.tasks[i].val_split
            file.write("\t\tTest samples ({}): expected {}, actual {}\n".format(test_split,
                                                                                cg.tasks[i].requested_samples["test"],
                                                                                cg.tasks[i].samples["test"]))

            file.write("\t\tSampling with{} replacement\n".format("" if cg.tasks[i].with_replacement else "out"))

    for split in ["train", "val", "test"]:
        with open("samples/shuffled_curriculum/{}/annotations.csv".format(split), "w", newline="\n", encoding="utf-8") as csvfile:
            fieldnames = ["filename", "task_id", "label", "supervised", "symbol"]
            fieldnames.extend(cg.config["concept_list"])
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()
            for i, sample in enumerate(cg.generate_shuffled_curriculum(split, task_id_noise=0.3, batch_size=1)): # NOTE: generate_shuffled_curriculum() acts as a finite stream, but samples are still chosen randomly.
                sample_img, label, supervised, task_id, symbol, concepts = sample

                Image.fromarray(sample_img[0]).save("samples/shuffled_curriculum/{}/0/{:06d}.png".format(split, i), "PNG")

                row = {"filename": "0/{:06d}.png".format(i), "task_id": task_id[0], "label": label[0], "supervised": supervised[0], "symbol": symbol[0]}
                for k, v in concepts[0].items():
                    row[k] = v

                writer.writerow(row)
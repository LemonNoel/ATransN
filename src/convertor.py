import torch
import os
import shutil
import re

def get_metric(line, type="test"):
    matched = re.findall(r"%s (MR|MRR|HITS@[\d]+) at step (\d+): ([\d\.]+)" % (type), line)
    if matched:
        return (matched[0][0], int(matched[0][1]), float(matched[0][2]))
    else:
        return None

def get_best_steps(log_file):
    with open(log_file, "r") as f:
        valid_metric = dict()
        test_metric = dict()
        for line in f:
            x = get_metric(line, "valid")
            if x:
                if x[1] not in valid_metric:
                    valid_metric[x[1]] = dict()
                valid_metric[x[1]][x[0]] = x[2]
                continue
            x = get_metric(line, "test")
            if x:
                if x[1] not in test_metric:
                    test_metric[x[1]] = dict()
                test_metric[x[1]][x[0]] = x[2]
        valid_metric = sorted(valid_metric.items(), key=lambda x: x[1]["MRR"] + x[1]["HITS@3"] + x[1]["HITS@10"], reverse=True)
        test_metric = sorted(test_metric.items(), key=lambda x: x[1]["MRR"] + x[1]["HITS@3"] + x[1]["HITS@10"], reverse=True)

        return {"valid": valid_metric[0][0], "test": test_metric[0][0]}

def convert_model_file_to_embeddings(filename):
    state_dict = torch.load(filename, map_location=torch.device("cpu"))
    if "learner" not in state_dict:
        state_dict["learner"] = state_dict.pop("model")
    torch.save(state_dict, filename)
    dirname = os.path.dirname(filename)
    torch.save(state_dict["learner"]["entity_embedding"], os.path.join(dirname, "entity.pt"))
    torch.save(state_dict["learner"]["relation_embedding"], os.path.join(dirname, "relation.pt"))


if __name__ == "__main__":

    dumps_dir = "../dumps"
    for dirname in os.listdir(dumps_dir):
        if "CN3l" not in dirname:
            continue
        print(dirname)
        best_steps = get_best_steps(os.path.join(dumps_dir, dirname, "log.txt"))
        # if os.path.exists(os.path.join(dumps_dir, dirname, "checkpoint_%d.pt" % (best_steps["valid"]))):
        #     shutil.copy(
        #         os.path.join(dumps_dir, dirname, "checkpoint_%d.pt" % (best_steps["valid"])),
        #         os.path.join(dumps_dir, dirname, "checkpoint_valid.pt"))
        #     convert_model_file_to_embeddings(os.path.join(dumps_dir, dirname, "checkpoint_valid.pt"))
        # if os.path.exists(os.path.join(dumps_dir, dirname, "checkpoint_%d.pt" % (best_steps["test"]))):
        #     shutil.copy(
        #         os.path.join(dumps_dir, dirname, "checkpoint_%d.pt" % (best_steps["test"])),
        #         os.path.join(dumps_dir, dirname, "checkpoint_test.pt"))
        for pt_file in os.listdir(os.path.join(dumps_dir, dirname)):
            if not pt_file.startswith("checkpoint"):
                continue
            convert_model_file_to_embeddings(os.path.join(dumps_dir, dirname, pt_file))
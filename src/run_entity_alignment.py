import argparse
import re

def read_dictionary(file_name):
    with open(file_name, 'r', encoding='utf-8') as fp:
        data = dict([tuple(x.rstrip().split('\t')) for x in fp.readlines()])
        return data

def read_and_encode(teacher_dict_file, student_dict_file, save_file, mapping_file=None):
    teacher_entity_dict = read_dictionary(teacher_dict_file)
    student_entity_dict = read_dictionary(student_dict_file)
    
    if mapping_file is not None:
        with open(mapping_file, 'r', encoding='utf-8') as fp:
            mapping = dict([tuple(re.split(r"@@@", line.rstrip())) for line in fp])
    else:
        mapping = {k: k for k in set(teacher_entity_dict.keys()) & set(student_entity_dict.keys())}
    
    with open(save_file, 'w', encoding='utf-8') as wp:
        for t, s in mapping.items():
            t = teacher_entity_dict.get(t, -1)
            s = student_entity_dict.get(s, -1)
            if t == -1 or s == -1:
                continue
            wp.write('%s\t%s\n' % (t, s))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--mapping_file", type=str, default=None,
                        help="mapping file path")
    parser.add_argument("--teacher_dict_file", type=str,
                        help="teacher dict file path")
    parser.add_argument("--student_dict_file", type=str,
                        help="student dict file path")
    parser.add_argument("--save_file", type=str,
                        help="id mapping output file path")

    args = parser.parse_args()

    read_and_encode(args.teacher_dict_file, args.student_dict_file, args.save_file, args.mapping_file)

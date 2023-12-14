
def load_quadruples(inpath):
    with open(inpath, 'r') as f:
        quadrupleList = []
        for line in f:
            try:
                line_split = line.split()
                head = int(line_split[0])
                rel = int(line_split[1])
                tail = int(line_split[2])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
            except:
                print(line)
    return quadrupleList

def get_obj_seq(quadrupleList, time_intev=24):
    sub_rel_objseq = dict()
    for item in quadrupleList:
        key = (item[0], item[1])
        if key in sub_rel_objseq:
            sub_rel_objseq[key].append((item[2], item[3]/time_intev))
        else:
            sub_rel_objseq[key] = [(item[2], item[3]/time_intev)]
    return sub_rel_objseq

def get_object_list_length(obj_list):
    s = set()
    for item in obj_list:
        s.add(item[1])
    return len(s)

def is_available_sequence(obj_list, p=0.5):
    if get_object_list_length(obj_list) / len(obj_list) < p:
        return False
    return True

def entity_to_words(input_path):
    ent_dict, word_dict = {}, {}
    word_count = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            ent, id = line.strip().split("\t")
            two_parts = ent.strip().split("(")
            if len(two_parts) >= 2:
                part1 = two_parts[0].strip()
                part2 = two_parts[1][0:-1].strip()
            else:
                part1 = ent.strip()
                part2 = part1
            if part1 not in word_dict:
                word_dict[part1] = word_count
                word_count = word_count + 1
            if part2 not in word_dict:
                word_dict[part2] = word_count
                word_count = word_count + 1
            ent_dict[id] = [word_dict[part1], word_dict[part2]]

    return ent_dict, len(word_dict)

def get_stat_data(input_path='stat.txt'):
    stat = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            stat = line.split()
    return int(stat[0]), int(stat[1])

def get_score_from_models(entattr_model, ent_model, rel_model, batch_data, device):
    score1 = entattr_model(batch_data, device)
    score2 = ent_model(batch_data)
    score3 = rel_model(batch_data)
    score = score1 + 0.35 * score2 + 0.15 * score3
    return score
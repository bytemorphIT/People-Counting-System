
def average_confidence(conf_list):
    if not conf_list:
        return 0
    return sum(conf_list) / len(conf_list) * 100

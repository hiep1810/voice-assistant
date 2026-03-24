try:
    from editdistance_s import distance as eval
except ImportError:
    def eval(s1, s2):
        return 0

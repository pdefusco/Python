
class Lexicon(object):

    def __init__(self):
        self.direction = {'north':'north', 'south':'south', 'east':'east'}
        self.verb = {'go':'go', 'kill':'kill', 'eat':'eat'}

    def scan(self, words):

        

        split_words = words.split(' ')
        out_list = []
        for i in split_words:
            out_list.append(('direction',self.direction.get(i, None)))
        return out_list





'''def convert_numbers(s):
    try:
        return int(s)

    except ValueError:
        return None
'''

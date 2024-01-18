import json
import functools
import pickle
import os
import time
import threading
from nltk.stem import PorterStemmer
stem = functools.cache(PorterStemmer().stem)

print_progress = False
dprint = lambda *a, **k: print(*a, **k) if print_progress else None

ascii = {chr(i + 97) for i in range(26)}
stopwords = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'b', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'c', 'can', 'couldn', "couldn't", 'd', 'defn', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'e', 'each', 'etc', 'f', 'few', 'for', 'from', 'further', 'g', 'h', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'j', 'just', 'k', 'l', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'n', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'obs', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'p', 'q', 'r', 're', 's', 'same', 'shak', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'u', 'under', 'until', 'up', 'v', 've', 'very', 'w', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'x', 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 'z'}


def conditional_load(path, function=None, args=(), read=json.load, serialize=json.dump, read_method='r', write_method="x"):
    if os.path.isfile(path):  # if the data exists in a json file, then prefer loading that over redoing the work
        dprint(f"Reading data from '{path}' with function '{read.__name__}'")
        with open(path, read_method) as f:
            return read(f)
            
    if function is None:
        raise Exception(f"\"{path}\" did not exist")
            
    s_args = ', '.join([type(a).__name__ for a in args])
    dprint(f"Running '{function.__name__}({s_args})' instead of reading from '{path}'")

    ret = function(*args)
    try:
        with open(path, write_method) as f:
            serialize(ret, f)
    except Exception as e:
        os.remove(path)
        dprint(f"Serializing the return value with function 'serialize' failed. Ignoring step and returning without writing to file.")
        dprint(repr(e))
        
    return ret

class ThreadManager:
    NOT_FINISHED = 0xDEADBEEF
    
    def __init__(self, user_function, arguments, max_running=10, record_values=True):
        if record_values:
            self.values = [self.NOT_FINISHED]*len(arguments)
        
        self.record_values = record_values
        if self.record_values:
            def run(i): self.values[i] = user_function(*arguments[i])
        else:
            def run(i): user_function(*arguments[i])
            
        self.queue = [threading.Thread(target=run, args=(i,)) for i in range(len(arguments))][::-1]
        self.total_running = max_running

    def run(self, *, progress=False):
        threads = len(self.queue)
        
        running = []
        for _ in range(self.total_running):
            if not len(self.queue):
                break
            running.append(t := self.queue.pop())
            t.start()

        while self.queue:
            for t in running:
                if not t.is_alive():
                    running.remove(t)
                    running.append(t := self.queue.pop())
                    t.start()
                    break
            else:
                if progress:
                    dprint(f"\r{len(self.queue)} / {threads} threads waiting, {self.total_running} active", end=" " * 5)
                time.sleep(1)

        while running := [t for t in running if t.is_alive()]:
            if progress:
                dprint(f"\r0 / {threads} threads waiting, {len(running)} active", end=" " * 5)
            time.sleep(1)

        if progress:
            dprint("\rAll threads finished" + " " * 70)
            
        if self.record_values:
            assert all(x != self.NOT_FINISHED for x in self.values)
            return self.values

class DictionaryGroup:
    def __len__(self):
        return self.n
        
    def __init__(self, *dictionaries):           
        self.n = len(dictionaries)
        self.dictionaries = dictionaries

    def __getitem__(self, value):
        if isinstance(value, str):
            for d in self.dictionaries:
                if d.name == value:
                    return d
            raise KeyError(f"No dictionary named '{value}'")
        else:
            return self.dictionaries[value]
        
    # get the attribute 'name' from each dictionary in the group and return as a list
    def getattr(self, name):
        return [getattr(dictionary, name) for dictionary in self.dictionaries]

    # pair the name of the dictionary to the value of that dictionary
    def getpairs(self, key, value):
        return {getattr(dictionary, key): getattr(dictionary, value) for dictionary in self.dictionaries}

    # add a dictionary to the group and clear the cache for self.apply
    def add(self, d):
        self.n += 1
        self.dictionaries.append(d)
        self.apply.cache_clear()

    # remove a dictionary from the group and clear the cache for self.apply
    def remove(self, i):
        del self.dictionaries[i]
        self.apply.cache_clear()

    # apply a function to each dictionary in the group and return the results as a list
    @functools.cache
    def apply(self, func):
        return [func(d) for d in self.dictionaries]

    # return a string representation of the DictionaryGroup
    def __repr__(self):
        string = self.__class__.__name__ + "({\n"
        for d in self.dictionaries:
            string += f'    {repr(d)},\n'
        return string + "})"


class Dictionary:
    def __getitem__(self, word):
        if word in self.words:
            return self.words[word]
        elif word in self.stemmed_words:
            return self.words[self.stemmed_words[word]]
        assert word not in self
        raise KeyError(f'"{word}" is not a valid key')

    def __contains__(self, word):
        return word in self.stemmed_words or word in self.words
        
    def separate_words(self):
        # define a function 'f' to process word entries
        def f(entries):
            processed_entries = " ".join(entries).lower()  # lowercase and join all entries into one string
            all_stopwords = []
            all_words = []
            for word in re.findall(r"\w[\w']*", processed_entries):
                (all_words if word not in stopwords else all_stopwords).append(stem(word))
            return all_words, all_stopwords

        keys, text = zip(*self.definitions.items())  # unzip dictionary

        # process word entries in parallel using threads
        thread_manager = ThreadManager(f, [(arg,) for arg in text])
        separated = thread_manager.run()
        return dict(zip(keys, separated))

    # calculate word frequency and return as a dictionary
    def get_frequency(self, to_count):
        if not to_count: return {}
        all_values = functools.reduce(list.__add__, to_count.values())
        word_counter = Counter(all_values)  # count the words after joining all lists together using functools.reduce()
        return dict(word_counter)
        
    def __init__(self, root_dir, name):        
        dprint(f"Creating dictionary from '{root_dir}/{name}'")
        
        fp = f'{root_dir}/{name}/{name}'
        info = conditional_load(f'{fp}.metadata')
        self.definitions = conditional_load(f'{fp}.dictionary')

        words = conditional_load(f'{fp}.words', self.separate_words,
                                 read=pickle.load, serialize=lambda obj, f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL), 
                                 read_method="rb", write_method="xb")
        self.words, self.stopwords = {}, {}
        self.word_length = {}
        for word, (reg_words, stopwords) in words.items():
            self.words[word] = reg_words
            self.stopwords[word] = stopwords
            self.word_length[len(word)] = self.word_length.get(len(word),0) + 1
        
        self.most_common_words = conditional_load(f'{fp}.word_counts', self.get_frequency, args=(self.words,))
        self.most_common_stopwords = conditional_load(f'{fp}.stopword_counts', self.get_frequency, args=(self.stopwords,))
        
        self.stemmed_words = {stem(word): word for word in self.words}  # stem all the words in the dictionary so we can access it by either version
        self.all_words_used = set(self.words) | set(a for b in self.words.values() for a in b)
        self.word_to_number = {w: i for i, w in enumerate(self.all_words_used)}
        self.number_to_word = {v: k for k, v in self.word_to_number.items()}
        
        self.total_words = sum(len(entry) for entry in self.words.values())
        self.total_stopwords = sum(len(entry) for entry in self.stopwords.values())
        self.total_words_all = self.total_words + self.total_stopwords

        self.name = info["name"]
        self.short_name = fp.replace('/', '\\').split('\\')[-1]
        self.source = info["source"]
        self.total_entries = self.headwords = info["headwords"]
        self.most_definitions = info["most_defns"]
        self.average_definition_length = info["average_words_per_defn"]
        self.longest_definition = info["most_words_in_defn"]

    def write_basic_info(self):
        l = sum(i*c for i, c in self.word_length.items()) / sum(self.word_length.values())
        a = ',\n\t'.join(f'{w}: {c}' for w, c in self.most_definitions)
        b = len(self.all_words_used)
        return f'# of words defined: {self.total_entries}\nAverage length of word defined: {l}\nAverage Length of Definition: {self.average_definition_length}\nLargest # of defns: [\n\t{a}\n]\nUnique words: {b}'

    # return a string representation of the dictionary
    def __repr__(self):
        return f'{self.__class__.__name__}("{self.name}", words={self.headwords})'
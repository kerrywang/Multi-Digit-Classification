from DataPreprocessing.ProcessMethod import *

class Preprocessor(object):
    def __init__(self):
        self.processors = []

    def register_processor(self, processors):
        for processor in processors:
            assert isinstance(processor, ProcessMethod)
            self.processors.append(processor)

    def Process(self, data):
        for processor in self.processors:
            data = processor.Process(data)
        return data


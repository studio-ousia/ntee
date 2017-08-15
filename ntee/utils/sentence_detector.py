# -*- coding: utf-8 -*-

import pkg_resources

java_initialized = False


def init_java():
    global java_initialized

    if not java_initialized:
        import jnius_config

        jnius_config.set_classpath(
            pkg_resources.resource_filename(
                'ntee.utils', '/resources/opennlp-tools-1.5.3.jar'
            ),
        )

        java_initialized = True


class OpenNLPSentenceDetector(object):
    def __init__(self):
        init_java()

        from jnius import autoclass

        File = autoclass('java.io.File')
        SentenceModel = autoclass('opennlp.tools.sentdetect.SentenceModel')
        SentenceDetectorME = autoclass('opennlp.tools.sentdetect.SentenceDetectorME')

        sentence_model_file = pkg_resources.resource_filename(
            __name__, 'opennlp/en-sent.bin'
        )
        sentence_model = SentenceModel(File(sentence_model_file))
        self._detector = SentenceDetectorME(sentence_model)

    def sent_pos_detect(self, text):
        return [(span.getStart(), span.getEnd())
                for span in self._detector.sentPosDetect(text)]

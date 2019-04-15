import json
import logging
from typing import Any, Dict, List, Tuple

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("coqa")
class QuACReader(DatasetReader):
    """
    Reads a JSON-formatted Quesiton Answering in Context (QuAC) data file
    and returns a ``Dataset`` where the ``Instances`` have four fields: ``question``, a ``ListField``,
    ``passage``, another ``TextField``, and ``span_start`` and ``span_end``, both ``ListField`` composed of
    IndexFields`` into the ``passage`` ``TextField``.
    Two ``ListField``, composed of ``LabelField``, ``yesno_list`` and  ``followup_list`` is added.
    We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_text_lists'] and ``metadata['token_offsets']``.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    num_context_answers : ``int``, optional
        How many previous question answers to consider in a context.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 num_context_answers: int = 0) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._num_context_answers = num_context_answers

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)

            dataset = list()
            for i in range(20):
                dataset.append(dataset_json["data"][i])

            # dataset = dataset_json['data']
        logger.info("Reading the dataset")
        # for article in dataset:
        #     for paragraph_json in article['paragraphs']:
        #         paragraph = paragraph_json["context"]
        #         tokenized_paragraph = self._tokenizer.tokenize(paragraph)
        #         qas = paragraph_json['qas']
        #         metadata = {}
        #         metadata["instance_id"] = [qa['id'] for qa in qas]
        #         question_text_list = [qa["question"].strip().replace("\n", "") for qa in qas]
        #         answer_texts_list = [[answer['text'] for answer in qa['answers']] for qa in qas]
        #         metadata["question"] = question_text_list
        #         metadata['answer_texts_list'] = answer_texts_list
        #         span_starts_list = [[answer['answer_start'] for answer in qa['answers']] for qa in qas]
        #         span_ends_list = []
        #         for answer_starts, an_list in zip(span_starts_list, answer_texts_list):
        #             span_ends = [start + len(answer) for start, answer in zip(answer_starts, an_list)]
        #             span_ends_list.append(span_ends)
        #         yesno_list = [str(qa['yesno']) for qa in qas]
        #         followup_list = [str(qa['followup']) for qa in qas]
        #         instance = self.text_to_instance(question_text_list,
        #                                          paragraph,
        #                                          span_starts_list,
        #                                          span_ends_list,
        #                                          tokenized_paragraph,
        #                                          yesno_list,
        #                                          followup_list,x
        #                                          metadata)
        #         yield instance

        for article in dataset:
            paragraph = article['story']
            n_paragraph, padding = self.delete_leading_tokens_of_paragraph(paragraph)
            paragraph_length = len(n_paragraph) + 1
            # n_paragraph += " CANNOTANSWER"
            tokenized_paragraph = self._tokenizer.tokenize(n_paragraph)
            questions = article['questions']
            metadata = dict()
            metadata['instance_id'] = [question['turn_id'] for question in questions]
            question_text_list = [question['input_text'].strip().replace("\n", "") for question in questions]

            answer_texts_list = list()
            span_starts_list = list()
            span_ends_list = list()
            yesno_list = list()
            followup_list = list()
            answers = article["answers"]
            i = 0
            for answer in answers:

                answer_text_list = list()
                span_start_list = list()
                span_end_list = list()

                tmp = answer["span_text"]
                before = self.get_front_blanks(tmp, padding)
                answer_text = answer["span_text"].strip().replace("\n", "")

                span_start, span_end, yesno = self.deal_answer(answer_text, answer["span_start"], answer["input_text"],
                                                               paragraph_length, before)

                if answer_text.lower() == "unknown":
                    # answer_text = "CANNOTANSWER"
                    answer_text = n_paragraph[0]

                answer_text_list.append(answer_text)
                span_start_list.append(span_start)
                span_end_list.append(span_end)

                if "additional_answers" in article:
                    additional_answers = article["additional_answers"]
                    for index in additional_answers:
                        tmp = additional_answers[index][i]["span_text"]
                        before = self.get_front_blanks(tmp, padding)
                        answer_text = additional_answers[index][i]["span_text"].strip().replace("\n", "")

                        span_start, span_end, yesno = self.deal_answer(answer_text,
                                                                       additional_answers[index][i]["span_start"],
                                                                       additional_answers[index][i]["input_text"],
                                                                       paragraph_length, before)

                        if answer_text.lower() == "unknown":
                            # answer_text = "CANNOTANSWER"
                            answer_text = n_paragraph[0]

                        answer_text_list.append(answer_text)
                        span_start_list.append(span_start)
                        span_end_list.append(span_end)

                yesno_list.append(yesno)
                if i == len(answers) - 1:
                    followup_list.append("n")
                else:
                    followup_list.append("y")

                i += 1

                answer_texts_list.append(answer_text_list)
                span_starts_list.append(span_start_list)
                span_ends_list.append(span_end_list)

            metadata['question'] = question_text_list
            metadata['answer_texts_list'] = answer_texts_list
            # metadata['span_starts_list'] = span_starts_list

            instance = self.text_to_instance(question_text_list,
                                             paragraph,
                                             span_starts_list,
                                             span_ends_list,
                                             tokenized_paragraph,
                                             yesno_list,
                                             followup_list,
                                             metadata)
            yield instance

    def deal_answer(self, answer_text, answer_span_start, answer_input_text, paragraph_length, before):
        if answer_text.lower() == "unknown":
            # span_start = paragraph_length
            # span_end = span_start + len("CANNOTANSWER")
            span_start = span_end = 0
            yesno = "x"
        else:
            span_start = answer_span_start + before
            span_end = span_start + len(answer_text)
            if answer_input_text.lower() == "yes":
                yesno = "y"
            elif answer_input_text.lower() == "no":
                yesno = "n"
            else:
                yesno = "x"

        return span_start, span_end, yesno

    def get_front_blanks(self, answer, padding):
        answer = answer.replace("\n", "")
        before = 0
        for i in range(len(answer)):
            if answer[i] == ' ':
                before += 1
            else:
                break
        return before - padding

    def delete_leading_tokens_of_paragraph(self, paragraph):
        before = 0
        for i in range(len(paragraph)):
            if paragraph[i] == ' ' or paragraph[i] == '\n':
                before += 1
            else:
                break

        nparagraph = paragraph[before:]
        return nparagraph, before

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text_list: List[str],
                         passage_text: str,
                         start_span_list: List[List[int]] = None,
                         end_span_list: List[List[int]] = None,
                         passage_tokens: List[Token] = None,
                         yesno_list: List[int] = None,
                         followup_list: List[int] = None,
                         additional_metadata: Dict[str, Any] = None) -> Instance:
        # pylint: disable=arguments-differ
        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        answer_token_span_list = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        for start_list, end_list in zip(start_span_list, end_span_list):
            token_spans: List[Tuple[int, int]] = []
            for char_span_start, char_span_end in zip(start_list, end_list):
                (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                             (char_span_start, char_span_end))
                if error:
                    logger.debug("Passage: %s", passage_text)
                    logger.debug("Passage tokens: %s", passage_tokens)
                    logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                    logger.debug("Token span: (%d, %d)", span_start, span_end)
                    logger.debug("Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
                    logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
                token_spans.append((span_start, span_end))
            answer_token_span_list.append(token_spans)
        question_list_tokens = [self._tokenizer.tokenize(q) for q in question_text_list]
        # Map answer texts to "CANNOTANSWER" if more than half of them marked as so.
        additional_metadata['answer_texts_list'] = [util.handle_cannot(ans_list) for ans_list \
                                                    in additional_metadata['answer_texts_list']]
        return util.make_reading_comprehension_instance_quac(question_list_tokens,
                                                             passage_tokens,
                                                             self._token_indexers,
                                                             passage_text,
                                                             answer_token_span_list,
                                                             yesno_list,
                                                             followup_list,
                                                             additional_metadata,
                                                             self._num_context_answers)

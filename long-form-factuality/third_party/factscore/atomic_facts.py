# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Atomic fact generator."""

import json
import os
import re
import string
from typing import Optional

from absl import app
import nltk
from nltk import tokenize
import numpy as np
import rank_bm25
import spacy

# pylint: disable=g-bad-import-order
from common import modeling
from common import shared_config
from common import utils
# pylint: enable=g-bad-import-order

nltk.download('punkt', quiet=True)

MONTHS = [
    m.lower()
    for m in [
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
        'July',
        'August',
        'September',
        'October',
        'November',
        'December',
    ]
]
SPACY_MODEL = spacy.load('en_core_web_sm')
DEMON_DIR = 'third_party/factscore/demos/'
ATOMIC_FACT_INSTRUCTION = """

[ATOMIC FACT EXTRACTOR : {prompt}]

# Your role:
You are given a SENTENCE and its CONTEXT (CONTEXT includes a PROMPT and its RESPONSE where the SENTENCE originates). Your task is to break the SENTENCE down into a list of atomic facts. An atomic fact is a statement containing a singular piece of information that specifies a fact and can be fact-checked against evidence from the real world. Each atomic fact in the output list should contain a different piece of information. The atomic fact should be well contexualized, i.e the extracted atomic fact should contain all the information it needs from its sourrounding to be understood without the original sourrounding text. Like pronouns should be replaced by person's actual name, title, organization, place, events, etc, and correct time, geographic region or additional necessary context should be added.

# Your task:

Inputs:

```
# PROMPT
{prompt}
```

```
# RESPONSE
{response}
```

```
# SENTENCE
{sentence}
```

Based on the PROMPT and RESPONSE above, your task is to generate list of contextualized atomic facts from the input SENTENCE. Only output the atomic facts as a list, with each item starting with "- ". Do not include other formatting. Do not repeat atomic facts. Only include atomic facts that is literally part of the input sentence below, else refrain from including the fact. Return literal string "no atomic facts" if no atomic facts are found.
"""

FILTER_REDUNDANT_FACTS_INSTRUCTION = """
[ATOMIC FACT FILTER]

# Your role:
You are an intelligent information extractor and fact checker. 

# Your task:
Your task is to filter out redundant atomic facts from a list of atomic facts, that are repetition of another fact, generalization of another fact, specialization of another fact or paraphrase of another fact.

# Input:
For the following atomic facts listed below, remove all the redundant facts.

```
# Atomic facts
{atomic_facts}
```

# Output format:
Only output the filtered list. Each item should start with "- ". Do not include any other text apart from filtered atomic facts.
"""

class AtomicFactGenerator(object):
  """Atomic fact generator."""

  def __init__(
      self,
      api_key: str,
      demon_dir: Optional[str] = DEMON_DIR,
      gpt3_cache_file: Optional[str] = None,
      other_lm: Optional[modeling.Model] = None,
  ):
    self.nlp = SPACY_MODEL
    self.is_bio = True
    self.demon_path = os.path.join(
        demon_dir, 'demons.json' if self.is_bio else 'demons_complex.json'
    )
    self.other_lm = other_lm
    self.api_key = api_key
    self.gpt3_cache_file = gpt3_cache_file

    # get the demos
    with utils.open_file_wrapped(self.demon_path, mode='r') as f:
      self.demons = json.load(f)

    tokenized_corpus = [doc.split(' ') for doc in self.demons.keys()]
    self.bm25 = rank_bm25.BM25Okapi(tokenized_corpus)

  def run(self, prompt: str, generation: str, cost_estimate: Optional[bool] = None):
    """Convert the generation into a set of atomic facts."""
    assert isinstance(generation, str), 'generation must be a string'
    paragraphs = [
        para.strip() for para in generation.split('\n') if para.strip()
    ]
    return self.get_atomic_facts_from_paragraph(
        prompt, paragraphs, cost_estimate=cost_estimate
    )

  def get_atomic_facts_from_paragraph(self, prompt, paragraphs, cost_estimate=None):
    """Get the atomic facts from the paragraphs."""
    sentences, para_breaks = [], []

    for para_idx, paragraph in enumerate(paragraphs):
      if para_idx > 0:
        para_breaks.append(len(sentences))

      initials = detect_initials(paragraph)
      curr_sentences = tokenize.sent_tokenize(paragraph)
      curr_sentences_2 = tokenize.sent_tokenize(paragraph)
      curr_sentences = fix_sentence_splitter(curr_sentences, initials)
      curr_sentences_2 = fix_sentence_splitter(curr_sentences_2, initials)
      # ensure the crediability of the sentence splitter fixing algorithm
      assert curr_sentences == curr_sentences_2, (
          paragraph,
          curr_sentences,
          curr_sentences_2,
      )
      sentences += curr_sentences

    atoms_or_estimate = self.get_init_atomic_facts_from_sentence(
        prompt,
        [
            sent
            for i, sent in enumerate(sentences)
            if not (
                not self.is_bio
                and (
                    (
                        i == 0
                        and (
                            sent.startswith('Sure')
                            or sent.startswith('Here are')
                        )
                    )
                    or (
                        i == len(sentences) - 1
                        and (
                            sent.startswith('Please')
                            or sent.startswith('I hope')
                            or sent.startswith('Here are')
                        )
                    )
                )
            )
        ],
        cost_estimate=cost_estimate,
    )

    if cost_estimate:
      return atoms_or_estimate
    else:
      atoms = atoms_or_estimate

    atomic_facts_pairs = []

    for i, sent in enumerate(sentences):
      if not self.is_bio and (
          (i == 0 and (sent.startswith('Sure') or sent.startswith('Here are')))
          or (
              i == len(sentences) - 1
              and (
                  sent.startswith('Please')
                  or sent.startswith('I hope')
                  or sent.startswith('Here are')
              )
          )
      ):
        atomic_facts_pairs.append((sent, []))
      elif self.is_bio and sent.startswith(
          'This sentence does not contain any facts'
      ):
        atomic_facts_pairs.append((sent, []))
      elif (
          sent.startswith('Sure')
          or sent.startswith('Please')
          or (i == 0 and sent.startswith('Here are'))
      ):
        atomic_facts_pairs.append((sent, []))
      else:
        atomic_facts_pairs.append((sent, atoms[sent]))

    # postprocess_atomic_facts will fix minor issues from InstructGPT
    # it is supposed to handle sentence splitter issue too, but since here
    # we fixed sentence splitter issue already,
    # the new para_breaks should be identical to the original para_breaks
    if self.is_bio:
      atomic_facts_pairs, para_breaks = postprocess_atomic_facts(
          atomic_facts_pairs, list(para_breaks), self.nlp
      )

    return atomic_facts_pairs, para_breaks

  def get_init_atomic_facts_from_sentence(self, sentence_prompt, sentences, cost_estimate=None):
    """Get the initial atomic facts from the sentences."""
    is_bio, demons = self.is_bio, self.demons
    prompts, prompt_to_sent, atoms = [], {}, {}
    k = 1 if is_bio else 0
    n = 7 if is_bio else 8

    response = "\n".join(sentences)

    for sentence in sentences:
      if sentence in atoms:
        continue
      
      prompt = ATOMIC_FACT_INSTRUCTION.format(
        prompt=sentence_prompt, response=response, sentence=sentence
      )
      prompts.append(prompt)
      prompt_to_sent[prompt] = sentence

    if cost_estimate:
      total_words_estimate = 0

      for prompt in prompts:
        total_words_estimate += len(prompt.split())

      return total_words_estimate
    else:
      for prompt in prompts:
        if self.other_lm is not None:
          atomic_facts = self.other_lm.generate(prompt, temperature=0)

          if "no atomic facts" not in atomic_facts:
            output = self.other_lm.generate(
              FILTER_REDUNDANT_FACTS_INSTRUCTION.format(atomic_facts=atomic_facts),
              temperature=0
            )
          else:
            output = ""
        else:
          raise ValueError('other_lm is None')

        sentences_from_output = text_to_sentences(output)

        if not sentences_from_output:  # account for markdown lists
          sentences_from_output = text_to_sentences(output, separator='* ')

        atoms[prompt_to_sent[prompt]] = sentences_from_output

      for key, value in demons.items():
        if key not in atoms:
          atoms[key] = value

      return atoms


def best_demos(query, bm25, demons_sents, k):
  tokenized_query = query.split(' ')
  top_machings = bm25.get_top_n(tokenized_query, demons_sents, k)
  return top_machings


def text_to_sentences(text: str, separator: str = '- ') -> list[str]:
  """Transform InstructGPT output into sentences."""
  sentences = text.split(separator)[1:]
  sentences = [
      sentence[:sentence.find('\n')] if '\n' in sentence else sentence
      for sentence in sentences
  ]
  sentences = [
      sent.strip()[:-1] if sent.strip()[-1] == '\n' else sent.strip()
      for sent in sentences
  ]

  if sentences:
    if sentences[-1][-1] != '.':
      sentences[-1] = sentences[-1] + '.'
  else:
    sentences = []

  return sentences


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def is_num(text):
  try:
    _ = int(text)
    return True
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def is_date(text):
  text = normalize_answer(text)

  for token in text.split(' '):
    if (not is_num(token)) and token not in MONTHS:
      return False

  return True


def extract_numeric_values(text):
  pattern = r'\b\d+\b'  # regular expression pattern for integers
  numeric_values = re.findall(
      pattern, text
  )  # find all numeric values in the text
  return set(
      [value for value in numeric_values]
  )  # convert the values to float and return as a list


def detect_entities(text, nlp):
  """Detect entities from the text."""
  doc, entities = nlp(text), set()

  def _add_to_entities(text):
    if '-' in text:
      for each in text.split('-'):
        entities.add(each.strip())
    else:
      entities.add(text)

  for ent in doc.ents:
    # spacy often has errors with other types of entities
    if ent.label_ in [
        'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'
    ]:
      if is_date(ent.text):
        _add_to_entities(ent.text)
      else:
        for token in ent.text.split():
          if is_date(token):
            _add_to_entities(token)

  for new_ent in extract_numeric_values(text):
    if not np.any([new_ent in ent for ent in entities]):
      entities.add(new_ent)

  return entities


def postprocess_atomic_facts(in_atomic_facts, para_breaks, nlp):
  """Postprocess atomic facts."""
  verbs = [
      'born.',
      ' appointed.',
      ' characterized.',
      ' described.',
      ' known.',
      ' member.',
      ' advocate.',
      'served.',
      'elected.',
  ]
  permitted_verbs = ['founding member.']
  atomic_facts, new_atomic_facts, new_para_breaks = [], [], []

  for i, (sent, facts) in enumerate(in_atomic_facts):
    sent = sent.strip()

    if len(sent.split()) == 1 and i not in para_breaks and i > 0:
      assert i not in para_breaks
      atomic_facts[-1][0] += ' ' + sent
      atomic_facts[-1][1] += facts
    else:
      if i in para_breaks:
        new_para_breaks.append(len(atomic_facts))

      atomic_facts.append([sent, facts])

  for _, (sent, facts) in enumerate(atomic_facts):
    entities = detect_entities(sent, nlp)
    covered_entities, new_facts = set(), []

    for i, fact in enumerate(facts):
      if any([fact.endswith(verb) for verb in verbs]) and not any(
          [fact.endswith(verb) for verb in permitted_verbs]
      ):
        if any([
            fact[:-1] in other_fact
            for j, other_fact in enumerate(facts)
            if j != i
        ]):
          continue

      sent_entities = detect_entities(fact, nlp)
      covered_entities |= set([e for e in sent_entities if e in entities])
      new_entities = sent_entities - entities

      if new_entities:
        do_pass = False

        for new_ent in new_entities:
          pre_ent = None

          for ent in entities:
            if ent.startswith(new_ent):
              pre_ent = ent
              break

          if pre_ent is None:
            do_pass = True
            break

          fact = fact.replace(new_ent, pre_ent)
          covered_entities.add(pre_ent)

        if do_pass:
          continue

      if fact in new_facts:
        continue

      new_facts.append(fact)

    # there is a bug in spacy entity linker, so just go with the previous facts
    try:
      assert entities == covered_entities
    except AssertionError:
      new_facts = facts

    new_atomic_facts.append((sent, new_facts))

  return new_atomic_facts, new_para_breaks


def is_integer(s):
  try:
    _ = int(s)
    return True
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def detect_initials(text):
  pattern = r'[A-Z]\. ?[A-Z]\.'
  match = re.findall(pattern, text)
  return [m for m in match]


def fix_sentence_splitter(curr_sentences, initials):
  """Fix sentence splitter issues."""
  for initial in initials:
    if not np.any([initial in sent for sent in curr_sentences]):
      alpha1, alpha2 = [t.strip() for t in initial.split('.') if t.strip()]

      for i, (sent1, sent2) in enumerate(
          zip(curr_sentences, curr_sentences[1:])
      ):
        if sent1.endswith(alpha1 + '.') and sent2.startswith(alpha2 + '.'):
          # merge sentence i and i+1
          curr_sentences = (
              curr_sentences[:i]
              + [curr_sentences[i] + ' ' + curr_sentences[i + 1]]
              + curr_sentences[i + 2 :]
          )
          break

  sentences, combine_with_previous = [], None

  for sent_idx, sent in enumerate(curr_sentences):
    if len(sent.split()) <= 1 and sent_idx == 0:
      assert not combine_with_previous
      combine_with_previous = True
      sentences.append(sent)
    elif len(sent.split()) <= 1:
      assert sent_idx > 0
      sentences[-1] += ' ' + sent
    elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
      assert sent_idx > 0, curr_sentences
      sentences[-1] += ' ' + sent
      combine_with_previous = False
    elif combine_with_previous:
      assert sent_idx > 0
      sentences[-1] += ' ' + sent
      combine_with_previous = False
    else:
      assert not combine_with_previous
      sentences.append(sent)

  return sentences


def main(_) -> None:
  generator = AtomicFactGenerator(
      shared_config.openai_api_key, 'demos', gpt3_cache_file=None
  )
  atomic_facts, para_breaks = generator.run(
      'Thierry Henry (born 17 August 1977) is a French professional football'
      ' coach, pundit, and former player. He is considered one of the greatest'
      ' strikers of all time, and one the greatest players of the Premier'
      " League history. He has been named Arsenal F.C's greatest ever"
      ' player.\n\nHenry made his professional debut with Monaco in 1994 before'
      ' signing for defending Serie A champions Juventus. However, limited'
      " playing time, coupled with disagreements with the club's hierarchy, led"
      ' to him signing for Premier League club Arsenal for £11 million in 1999.'
  )
  print(atomic_facts)
  print(para_breaks)


if __name__ == '__main__':
  app.run(main)
